"""
평가 실행 진입점 — 평가셋 전체를 answer_pipeline으로 돌리고
summary.json / details.jsonl을 남긴다 (팀 규약 결과 파일 형식).

기대하는 평가셋 형식 (JSONL, 한 줄=문항 하나):
{"question_id": "q001", "question": "...", "task_type": "qa"}

사용 예:
  python run_eval.py \
    --evalset /srv/rfp/shared_data/processed/evalset_v1/questions.jsonl \
    --index /srv/rfp/shared_data/processed/index_v1 \
    --extraction-table /srv/rfp/shared_data/processed/rfp_extraction_table_v2/table.jsonl \
    --out /srv/rfp/shared_data/results/tm001
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from config import load_config
from git_info import get_git_info, warn_if_dirty
from vector_store import VectorStore
from embedding_client import EmbeddingClient
from generation_client import GenerationClient
from table_query import load_extraction_table
from deadline_metadata import load_deadline_by_document_id
from answer_pipeline import answer


def load_evalset(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--extraction-table", required=False)
    parser.add_argument("--registry", required=False,
                         help="document_registry_v2.json 경로 — 마감 필터용 CSV 매핑에 필요")
    parser.add_argument("--deadline-csv", required=False,
                         help="data_list.csv 경로 — 있어야 선별형 마감 필터(4-10-2)가 켜짐")
    parser.add_argument("--out", required=True, help="결과 저장 폴더")
    parser.add_argument("--experiment-config", required=False)
    args = parser.parse_args()

    cfg = load_config(args.experiment_config)
    warn_if_dirty(purpose="평가 실행")

    store = VectorStore.load(Path(args.index))
    table = load_extraction_table(Path(args.extraction_table)) if args.extraction_table else []

    deadline_map = None
    if args.deadline_csv and args.registry:
        deadline_map = load_deadline_by_document_id(
            Path(args.deadline_csv), Path(args.registry), cfg,
        )
    elif cfg.get("deadline_filter_default", {}).get("select", False):
        print("⚠️  --deadline-csv/--registry 미지정 — 선별형 마감 필터가 base.yaml엔 "
              "켜져 있는데 이 평가는 필터 없이 돕니다. 채점 결과에 마감 지난 사업이 "
              "섞여 들어갈 수 있습니다.")

    # message.txt 8번 확정 — select·no_search_needed만 있는 평가셋이면
    # OpenAI 클라이언트를 한 번도 안 만들 수도 있다. 지연 생성으로 통일.
    _cache: dict = {}

    def get_embed_client() -> EmbeddingClient:
        if "embed" not in _cache:
            _cache["embed"] = EmbeddingClient(cfg)
        return _cache["embed"]

    def get_gen_client() -> GenerationClient:
        if "gen" not in _cache:
            _cache["gen"] = GenerationClient(cfg)
        return _cache["gen"]

    items = load_evalset(Path(args.evalset))
    print(f"평가 문항 {len(items)}개 로드 완료")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    details = []
    task_counts: dict[str, int] = {}
    abstain_count = 0
    fallback_count = 0

    for i, item in enumerate(items, 1):
        q = item["question"]
        try:
            result = answer(q, store, get_embed_client, get_gen_client, table, cfg,
                             deadline_map=deadline_map)
            # ⚠️ 버그 수정(리뷰 반영): answer()는 내부에서 예외를 이미 잡아서
            # error_stage/error_detail로 반환한다 — 그래서 이 try/except는
            # answer() 호출 자체가 실패하는 극히 드문 경우(예: 인자 오류)만
            # 잡는다. result.error_stage를 확인 안 하면 실제 검색·조건질의·
            # 생성 실패가 있어도 error=None으로 조용히 기록돼 error_count=0으로
            # 나올 수 있었다.
            record = {
                "question_id": item.get("question_id", f"q{i:04d}"),
                "question": q,
                "expected_task_type": item.get("task_type"),
                "actual_task_type": result.task_type,
                "route_matched_rule": result.route_matched_rule,
                "route_is_fallback": result.route_is_fallback,
                "answer": result.text,
                "sources": result.sources,
                "abstained": result.abstained,
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
                "retrieved_scores": result.retrieved_scores,
                "condition_query": result.condition_query,
                "condition_result_doc_ids": result.condition_result_doc_ids,
                "error_stage": result.error_stage,
                "error": (
                    f"[{result.error_stage}] {result.error_detail}"
                    if result.error_stage else None
                ),
            }
        except Exception as e:
            record = {
                "question_id": item.get("question_id", f"q{i:04d}"),
                "question": q,
                "expected_task_type": item.get("task_type"),
                "actual_task_type": None,
                "route_matched_rule": None,
                "route_is_fallback": None,
                "answer": None,
                "sources": [],
                "abstained": None,
                "retrieved_chunk_ids": [],
                "retrieved_scores": [],
                "condition_query": None,
                "condition_result_doc_ids": [],
                "error_stage": "pipeline_call",
                "error": str(e),
            }

        details.append(record)
        task_counts[record["actual_task_type"] or "error"] = (
            task_counts.get(record["actual_task_type"] or "error", 0) + 1
        )
        if record["abstained"]:
            abstain_count += 1
        if record["route_is_fallback"]:
            fallback_count += 1

        if i % 10 == 0:
            print(f"  {i}/{len(items)} 처리 중...")

    with open(out_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    summary = {
        "config": cfg,
        "git": get_git_info(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(items),
        "task_type_distribution": task_counts,
        "abstain_count": abstain_count,
        "abstain_rate": abstain_count / len(items) if items else 0,
        "routing_fallback_count": fallback_count,
        "error_count": sum(1 for d in details if d["error"]),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ 평가 완료: {out_dir}")
    print(f"   기권율 {summary['abstain_rate']:.1%}, 분기 폴백 {fallback_count}건, "
          f"에러 {summary['error_count']}건")


if __name__ == "__main__":
    main()
