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
import sys
from pathlib import Path
from datetime import datetime, timezone

# scripts/run_eval.py 기준 ../rag를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from config import load_config
from git_info import get_git_info, warn_if_dirty
from vector_store import VectorStore
from embedding_client import EmbeddingClient
from generation_client import GenerationClient
from table_query import load_extraction_table
from deadline_metadata import load_deadline_by_document_id, load_org_index
from answer_pipeline import answer, SessionState  # 같은 scripts/ 폴더라 경로 추가 불필요


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
    parser.add_argument(
        "--continuous-session", action="store_true",
        help="평가셋에 session_id가 없을 때, 파일 전체를 하나의 이어지는 대화로 "
             "취급한다(anaphora 문항이 앞 문항 활성 문서를 이어받음). 명시적으로 "
             "켜야만 이렇게 동작한다 — 기본값은 문항마다 독립(세션 리셋)이다."
    )
    parser.add_argument(
        "--allow-no-deadline-filter", action="store_true",
        help="base.yaml에 선별형 마감 필터가 켜져 있는데 --deadline-csv/--registry를 "
             "안 줄 때, 에러 대신 필터 없이 진행하도록 명시적으로 허용한다. "
             "테스트 목적 외에는 쓰지 않는다."
    )
    parser.add_argument(
        "--allow-errors", action="store_true",
        help="문항 중 오류가 있어도 실패 종료코드 대신 성공으로 끝내는 것을 "
             "명시적으로 허용한다. 기본값은 오류 1건 이상이면 실패 종료."
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment_config)
    warn_if_dirty(purpose="평가 실행")

    store = VectorStore.load(Path(args.index))
    table = load_extraction_table(Path(args.extraction_table)) if args.extraction_table else []

    deadline_map = None
    org_index = None
    deadline_filter_should_be_on = cfg.get("deadline_filter_default", {}).get("select", False)
    if args.deadline_csv and args.registry:
        deadline_map = load_deadline_by_document_id(
            Path(args.deadline_csv), Path(args.registry), cfg,
        )
        org_index = load_org_index(Path(args.deadline_csv), Path(args.registry))
    elif deadline_filter_should_be_on and not args.allow_no_deadline_filter:
        # ⚠️ 정정(리뷰 반영): 예전엔 경고만 하고 필터 없이 계속 진행했다 —
        # base.yaml이 "필터를 켜라"고 확정해뒀는데 실행이 조용히 그걸 어긴
        # 셈이었다. 마감 지난 사업이 결과에 섞이는 건 안전 문제라 하드블록한다.
        raise RuntimeError(
            "base.yaml의 deadline_filter_default.select=true인데 --deadline-csv/"
            "--registry가 없습니다. 마감 지난 사업이 결과에 섞일 수 있어 중단합니다. "
            "두 인자를 채우거나, 의도적으로 우회하려면 --allow-no-deadline-filter를 "
            "명시적으로 주세요."
        )

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
    if not items:
        # ⚠️ 버그 수정(리뷰 반영): 빈 평가셋도 예전엔 그냥 "평가 완료"로
        # 끝났다 — 파일 경로를 잘못 넘겼거나 필터링 실수로 0문항이 된
        # 경우를 놓칠 수 있어 명시적으로 막는다.
        print("❌ 평가셋에 문항이 0개입니다 — 경로가 맞는지 확인하세요.")
        sys.exit(1)

    # 4-14(활성 문서 상태) — 평가셋 문항이 'session_id' 필드로 대화 단위를
    # 밝히고 있으면 그룹이 바뀔 때마다 세션을 리셋한다.
    # ⚠️ 정정(리뷰 반영): 예전엔 session_id가 하나도 없으면 파일 전체를 한
    # 대화로 취급했는데, 이게 위험한 기본값이었다 — 서로 무관한 문항인데
    # 우연히 활성 문서가 섞이는 게, 이어지는 문항인데 안 이어지는 것보다
    # 훨씬 위험한 실패다. 기본값을 뒤집는다: session_id가 없으면 문항마다
    # 독립(매번 세션 리셋)이 기본이고, --continuous-session을 명시적으로
    # 줘야만 파일 전체를 한 대화로 취급한다.
    has_session_ids = any(item.get("session_id") for item in items)
    if has_session_ids:
        continuity_mode = "session_id 그룹별"
    elif args.continuous_session:
        continuity_mode = "파일 전체(하나의 대화, --continuous-session 명시)"
    else:
        continuity_mode = "문항마다 독립(기본값)"
    print(f"세션 연속성 모드: {continuity_mode}")

    session = SessionState()
    prev_session_id = items[0].get("session_id") if items else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    details = []
    task_counts: dict[str, int] = {}
    abstain_count = 0
    fallback_count = 0

    for i, item in enumerate(items, 1):
        q = item["question"]
        cur_session_id = item.get("session_id")
        if has_session_ids:
            if not cur_session_id:
                # ⚠️ 버그 수정(리뷰 반영): 예전엔 session_id가 없는 문항끼리
                # cur_session_id(None) == prev_session_id(None)로 같다고
                # 판정돼서 서로 이어졌다 — 다른 문항엔 session_id가 있는데
                # 이 문항만 없다는 건 "독립 문항"이라는 뜻으로 보고, 항상
                # 새 세션으로 처리한다(다른 무관한 문항으로 절대 안 새게).
                session = SessionState()
                prev_session_id = None
            elif cur_session_id != prev_session_id:
                session = SessionState()  # session_id 그룹 전환 — 이전 문항 상태 안 이어받음
                prev_session_id = cur_session_id
            # else: 같은 session_id가 이어짐 — 세션 유지
        elif not args.continuous_session:
            session = SessionState()  # 기본값 — 매 문항 독립, 이전 문항 상태 절대 안 섞임
        # else: --continuous-session이고 session_id도 없음 → 세션을 리셋하지 않고 이어감

        # 문항 자체에 active_document_id가 박혀 있으면, 직전 문항 이어받기와
        # 무관하게 그 상태로 이 문항을 시작하라는 뜻(테스트 하네스가 이 문항
        # 전용 전제조건을 직접 주입하는 방식) — 이어받기 로직보다 우선한다.
        if item.get("active_document_id"):
            session.active_document_id = item["active_document_id"]

        try:
            result = answer(q, store, get_embed_client, get_gen_client, table, cfg,
                             deadline_map=deadline_map, session=session, org_index=org_index)
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
                "session_id": cur_session_id,
                "active_document_after": session.active_document_id,
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

    if summary["error_count"] > 0 and not args.allow_errors:
        # ⚠️ 버그 수정(리뷰 반영): 예전엔 오류가 몇 건이든 종료코드가 항상
        # 성공(0)이라, 자동 실행 파이프라인이 "에러 3건 있어도 평가 정상
        # 완료"로 오판할 수 있었다. 의도적으로 오류를 허용하려면
        # --allow-errors를 명시해야 한다.
        print(f"❌ 오류 {summary['error_count']}건 있어 실패로 종료합니다. "
              f"의도적으로 넘기려면 --allow-errors를 명시하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
