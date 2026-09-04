"""
평가 실행 진입점 — 평가셋 전체를 answer_pipeline으로 돌리고
summary.json / details.jsonl / responses.jsonl / api_cost_summary.json을 남긴다.

기대 평가셋 형식 (JSONL, 한 줄=문항 하나). id 키는 "id" 또는 "question_id":
{"id": "PRAC-EXT-001", "question": "...", "task_type": "extraction",
 "session_id": "s1"(선택), "active_document_id": "RFP-000001"(선택)}

사용 예:
  source /srv/rfp/venv/bin/activate
  export RAG_ROOT=/srv/rfp
  python3 run_eval.py --evalset /srv/rfp/evalset/practice_items.jsonl \
      --index /path/to/index --out /path/to/결과폴더

[2026-09-02 수정]
- 문항 시작마다 API 사용량을 초기화한다. 이전 문항의 토큰·비용이 다음 문항으로
  복사되던 버그(5-1)를 막는다. 생성 비용과 임베딩 비용을 분리해 기록한다.
- 추출표·identity_v2만 쓴 문항은 생성 API 비용이 0으로 남는다.
- 문항당 자동 재시도는 최대 1회(기본). 원인 확인 없는 무제한 재시도 금지.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from config import load_config  # noqa: E402
from git_info import get_git_info, warn_if_dirty  # noqa: E402
from embedding_client import EmbeddingClient  # noqa: E402
from generation_client import GenerationClient  # noqa: E402
from pricing import Usage, compute_cost  # noqa: E402
from answer_pipeline import (  # noqa: E402
    answer, answer_to_response, sanitize_error, SessionState,
    build_runtime, add_common_args,
)


def load_evalset(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def item_id(item: dict, i: int) -> str:
    return str(item.get("id") or item.get("question_id") or f"q{i:04d}")


def _merge_usage(*clients) -> Usage:
    total = Usage()
    for client in clients:
        if client is None:
            continue
        u = client.usage
        total.generation_requests += u.generation_requests
        total.generation_input_tokens += u.generation_input_tokens
        total.generation_cached_input_tokens += u.generation_cached_input_tokens
        total.generation_output_tokens += u.generation_output_tokens
        total.embedding_requests += u.embedding_requests
        total.embedding_tokens += u.embedding_tokens
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--out", required=True, help="결과 저장 폴더")
    add_common_args(parser)
    parser.add_argument(
        "--continuous-session", action="store_true",
        help="평가셋에 session_id가 없을 때 파일 전체를 하나의 이어지는 대화로 취급한다. "
             "기본값은 문항마다 독립(세션 리셋).")
    parser.add_argument(
        "--allow-errors", action="store_true",
        help="문항 오류가 있어도 실패 종료코드 대신 성공으로 끝내는 것을 명시적으로 허용.")
    parser.add_argument(
        "--max-item-retries", type=int, default=1,
        help="문항당 자동 재시도 횟수(기본 1). 원인 확인 없는 무제한 재시도 금지.")
    args = parser.parse_args()

    if args.max_item_retries < 0 or args.max_item_retries > 1:
        print("❌ --max-item-retries는 0 또는 1만 허용합니다(무제한 재시도 금지).")
        sys.exit(2)

    cfg = load_config(args.experiment_config)
    warn_if_dirty(purpose="평가 실행")
    rt = build_runtime(args, cfg)
    store, table, identity = rt["store"], rt["table"], rt["identity"]
    locator = rt["locator"]
    registry_scope = rt["registry_scope"]
    print(f"인덱스 꼬리표 검증 통과: {rt['index_check']['index_dir']} "
          f"(dim={rt['index_check']['vector_dimension']})")

    cache: dict = {}

    def get_embed_client() -> EmbeddingClient:
        if "embed" not in cache:
            cache["embed"] = EmbeddingClient(cfg)
        return cache["embed"]

    def get_gen_client() -> GenerationClient:
        if "gen" not in cache:
            cache["gen"] = GenerationClient(cfg)
        return cache["gen"]

    items = load_evalset(Path(args.evalset))
    print(f"평가 문항 {len(items)}개 로드 완료")
    if not items:
        print("❌ 평가셋에 문항이 0개입니다 — 경로가 맞는지 확인하세요.")
        sys.exit(1)

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

    details, responses = [], []
    task_counts: dict[str, int] = {}
    abstain_count = fallback_count = 0
    total_usage = Usage()
    total_retries = 0
    t_all = time.perf_counter()

    for i, item in enumerate(items, 1):
        q = item["question"]
        qid = item_id(item, i)
        cur_session_id = item.get("session_id")
        if has_session_ids:
            if not cur_session_id:
                session = SessionState()
                prev_session_id = None
            elif cur_session_id != prev_session_id:
                session = SessionState()
                prev_session_id = cur_session_id
        elif not args.continuous_session:
            session = SessionState()

        if item.get("active_document_id"):
            session.active_document_id = item["active_document_id"]

        # ⭐ 문항 시작 — 이전 문항 사용량이 절대 넘어오지 않게 초기화
        for key in ("gen", "embed"):
            if key in cache:
                cache[key].reset_usage()

        retries_used = 0
        result = None
        last_error = None
        t0 = time.perf_counter()
        for attempt in range(args.max_item_retries + 1):
            if attempt:
                retries_used += 1
                for key in ("gen", "embed"):
                    if key in cache:
                        cache[key].reset_usage()
            try:
                result = answer(q, store, get_embed_client, get_gen_client,
                                table, cfg, identity=identity, session=session,
                                locator=locator, registry_scope=registry_scope)
                last_error = None
            except Exception as e:  # noqa: BLE001
                last_error = sanitize_error(f"{type(e).__name__}: {e}")
                result = None
            if result is not None and not result.error_stage:
                break
            if result is not None and result.error_stage:
                last_error = result.error_detail
        latency_ms = round((time.perf_counter() - t0) * 1000)
        total_retries += retries_used

        item_usage = _merge_usage(cache.get("gen"), cache.get("embed"))
        cost_usd, cost_detail = compute_cost(cfg, item_usage)
        for f in ("generation_requests", "generation_input_tokens",
                  "generation_cached_input_tokens", "generation_output_tokens",
                  "embedding_requests", "embedding_tokens"):
            setattr(total_usage, f, getattr(total_usage, f) + getattr(item_usage, f))

        if result is None:
            record = {
                "question_id": qid, "question": q,
                "expected_task_type": item.get("task_type"),
                "expected_answer_type": item.get("answer_type"),
                "actual_task_type": None, "route": None,
                "route_matched_rule": None, "route_is_fallback": None,
                "answer": None, "sources": [], "abstained": None,
                "retrieved_chunk_ids": [], "retrieved_scores": [],
                "condition_query": None, "condition_result_doc_ids": [],
                "error_stage": "pipeline_call", "error": last_error,
                "session_id": cur_session_id,
                "active_document_after": session.active_document_id,
                "latency_ms": latency_ms, "cost_usd": cost_usd,
                "cost_detail": cost_detail, "retries": retries_used,
                "called_generation_api": item_usage.generation_requests > 0,
                "called_embedding_api": item_usage.embedding_requests > 0,
            }
            responses.append({
                "id": qid, "answer": None, "structured_answer": None,
                "contexts": [], "retrieved": [], "citations": [],
                "selected_document_ids": [], "abstained": None, "route": None,
                "failure": last_error, "latency_ms": latency_ms, "cost_usd": cost_usd,
            })
        else:
            result.latency_ms = latency_ms
            result.cost_usd = cost_usd
            result.cost_detail = cost_detail
            record = {
                "question_id": qid, "question": q,
                "expected_task_type": item.get("task_type"),
                "expected_answer_type": item.get("answer_type"),
                "actual_task_type": result.task_type,
                "route": result.route,
                "route_matched_rule": result.route_matched_rule,
                "route_is_fallback": result.route_is_fallback,
                "answer": result.text,
                "sources": result.sources,
                "used_sources": result.used_sources,
                "resolution": result.resolution,
                "abstained": result.abstained,
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
                "retrieved_scores": result.retrieved_scores,
                "condition_query": result.condition_query,
                "condition_result_doc_ids": result.condition_result_doc_ids,
                "citation_count": len(result.citations),
                "citation_diagnostics": result.citation_diagnostics,
                "error_stage": result.error_stage,
                "error": (f"[{result.error_stage}] {sanitize_error(result.error_detail)}"
                          if result.error_stage else None),
                "session_id": cur_session_id,
                "active_document_after": session.active_document_id,
                "latency_ms": latency_ms, "cost_usd": cost_usd,
                "cost_detail": cost_detail, "retries": retries_used,
                "called_generation_api": item_usage.generation_requests > 0,
                "called_embedding_api": item_usage.embedding_requests > 0,
            }
            responses.append(answer_to_response(qid, result))

        details.append(record)
        key = record["actual_task_type"] or "error"
        task_counts[key] = task_counts.get(key, 0) + 1
        if record["abstained"]:
            abstain_count += 1
        if record["route_is_fallback"]:
            fallback_count += 1
        if i % 10 == 0:
            print(f"  {i}/{len(items)} 처리 중...")

    total_elapsed_ms = round((time.perf_counter() - t_all) * 1000)

    with open(out_dir / "details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_cost, total_cost_detail = compute_cost(cfg, total_usage)
    api_summary = {
        "evalset": str(args.evalset),
        "generation_model": cfg.get("generation_model"),
        "embedding_model": cfg.get("embedding_model"),
        "pricing_unit": cfg.get("pricing_unit"),
        "pricing": cfg.get("pricing"),
        "total_requests": {
            "generation": total_usage.generation_requests,
            "embedding": total_usage.embedding_requests,
        },
        "tokens": total_usage.as_dict(),
        "total_cost_usd": total_cost,
        "cost_breakdown": total_cost_detail,
        "total_elapsed_ms": total_elapsed_ms,
        "retries": total_retries,
        "per_item": [
            {"id": d["question_id"], "latency_ms": d["latency_ms"],
             "cost_usd": d["cost_usd"], "retries": d["retries"],
             "called_generation_api": d["called_generation_api"],
             "called_embedding_api": d["called_embedding_api"],
             "tokens": (d.get("cost_detail") or {}).get("usage")}
            for d in details
        ],
    }
    with open(out_dir / "api_cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(api_summary, f, ensure_ascii=False, indent=2)

    summary = {
        "config": cfg,
        "input_paths": rt["paths"],
        "index_check": rt["index_check"],
        "git": get_git_info(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_datetime": cfg.get("reference_datetime"),
        # 실제로 실린 생성 프롬프트 파일 — 설정과 실행이 어긋나면 여기서 드러난다
        "prompt_generate_configured": cfg.get("prompt_generate"),
        "prompt_generate_loaded": getattr(cache.get("gen"), "prompt_file", None),
        "total_questions": len(items),
        "task_type_distribution": task_counts,
        "route_distribution": {
            r: sum(1 for d in details if d["route"] == r)
            for r in sorted({d["route"] for d in details if d["route"]})
        },
        "abstain_count": abstain_count,
        "abstain_rate": abstain_count / len(items) if items else 0,
        "routing_fallback_count": fallback_count,
        "error_count": sum(1 for d in details if d["error"]),
        "total_cost_usd": total_cost,
        "total_elapsed_ms": total_elapsed_ms,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ 평가 완료: {out_dir}")
    print(f"   기권율 {summary['abstain_rate']:.1%}, 분기 폴백 {fallback_count}건, "
          f"에러 {summary['error_count']}건, 총비용 {total_cost}")

    if summary["error_count"] > 0 and not args.allow_errors:
        print(f"❌ 오류 {summary['error_count']}건 있어 실패로 종료합니다. "
              f"의도적으로 넘기려면 --allow-errors를 명시하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
