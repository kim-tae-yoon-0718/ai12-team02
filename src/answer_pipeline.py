"""
질문 답하기 진입점 — F-0(분기) → G/G-2(검색·조건질의) → I(컨텍스트 구성)
→ J(생성) → K(출처 부착) → K-2(형식화).

사용 예:
  export RAG_ROOT=/srv/rfp
  export OPENAI_API_KEY=...
  python answer_pipeline.py --question "5억 이상인 사업 알려줘" \
      --index /srv/rfp/shared_data/processed/index_v1 \
      --extraction-table /srv/rfp/shared_data/processed/rfp_extraction_table_v2/table.jsonl
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import load_config
from vector_store import VectorStore, ChunkMetadata
from embedding_client import EmbeddingClient
from generation_client import GenerationClient
from router import route, RouteResult
from table_query import (
    load_extraction_table, parse_conditions, run_conditions_query, QueryResult,
    detect_field, detect_document_id, detect_document_ids, needs_explanation, lookup_field,
)
from deadline_metadata import load_deadline_by_document_id, is_before_deadline


@dataclass
class Answer:
    text: str
    task_type: str
    route_matched_rule: str | None
    route_is_fallback: bool
    sources: list[str]
    abstained: bool
    # message.txt 10번 — 하루님 채점기가 검색/문서특정/생성 실패를 분리하는 데 씀
    retrieved_chunk_ids: list[str] = None      # type: ignore[assignment]
    retrieved_scores: list[float] = None       # type: ignore[assignment]
    condition_query: list[dict] | None = None
    condition_result_doc_ids: list[str] = None  # type: ignore[assignment]
    error_stage: str | None = None
    error_detail: str | None = None

    def __post_init__(self):
        if self.retrieved_chunk_ids is None:
            self.retrieved_chunk_ids = []
        if self.retrieved_scores is None:
            self.retrieved_scores = []
        if self.condition_result_doc_ids is None:
            self.condition_result_doc_ids = []


def format_source(m: ChunkMetadata) -> str:
    """4-13 확정 형식: 문서명 > 장절 > 표/문단번호."""
    parts = [m.document_name, m.chapter]
    if m.chunk_type == "table" and m.table_idx is not None:
        label = f"표 {m.table_idx}"
        if m.part is not None and m.of is not None:
            label += f" ({m.part}/{m.of})"
        parts.append(label)
    return " > ".join(p for p in parts if p)


def answer_qa_or_extract_by_search(
    question: str, store: VectorStore, embed_client: EmbeddingClient,
    gen_client: GenerationClient, cfg: dict[str, Any],
    document_id: str | None = None,
) -> Answer:
    """G(검색) → I(컨텍스트) → J(생성) → K(출처) 경로.
    document_id를 주면 그 문서로 검색 범위를 좁힌다(4-14, 추출형의
    '원문 설명 필요' 하위경로에서 사용)."""
    query_vec = embed_client.embed_query(question)
    top_k = cfg.get("top_k", 5)
    results = store.search(query_vec, top_k=top_k, active_only=True, document_id=document_id)

    if not results:
        # 4-12-1 확정: 검색 결과가 아예 없으면 이유 추측 없이 "모릅니다"만
        return Answer(
            text="확인할 수 없습니다.", task_type="qa",
            route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
        )

    # ⚠️ 버그 수정(리뷰 반영): 4-7-1 확정 규칙 — "빈 셀 60% 초과 표(degraded)는
    # 값을 생성하지 않고 원문 위치만 안내"가 지금까지 생성 단계에 전혀
    # 연결돼 있지 않았다. table_degraded=true인 청크를 그대로 LLM 컨텍스트에
    # 넣으면 손상된 표에서 값을 지어낼 위험이 있다 — 생성용 컨텍스트에서
    # 빼고 원문 위치 안내로만 따로 붙인다.
    normal_results = [(m, s) for m, s in results if not m.table_degraded]
    degraded_results = [(m, s) for m, s in results if m.table_degraded]

    if not normal_results:
        # 검색된 게 전부 손상된 표뿐 — LLM 호출 자체를 안 하고 위치만 안내
        text = (
            "이 항목은 표가 손상되어(빈 셀 60% 초과) 값을 추정하지 않습니다. "
            "아래 원문 위치를 직접 확인해주세요:\n"
            + "\n".join(f"- {format_source(m)}" for m, _ in degraded_results)
        )
        return Answer(
            text=text, task_type="qa", route_matched_rule=None,
            route_is_fallback=False,
            sources=[format_source(m) for m, _ in degraded_results],
            abstained=True,
            retrieved_chunk_ids=[m.chunk_id for m, _ in degraded_results],
            retrieved_scores=[s for _, s in degraded_results],
        )

    context_texts = [f"[출처: {format_source(m)}]\n{m.text}" for m, _ in normal_results]

    answer_text = gen_client.generate(question, context_texts)
    sources = [format_source(m) for m, _ in normal_results]

    if degraded_results:
        answer_text += (
            "\n\n[표 손상으로 값 생성 안 함 — 원문 위치만 안내]\n"
            + "\n".join(f"- {format_source(m)}" for m, _ in degraded_results)
        )
        sources += [format_source(m) for m, _ in degraded_results]

    return Answer(
        text=answer_text, task_type="qa", route_matched_rule=None,
        route_is_fallback=False, sources=sources, abstained=False,
        retrieved_chunk_ids=[m.chunk_id for m, _ in results],
        retrieved_scores=[s for _, s in results],
    )


def apply_deadline_filter(
    doc_ids: list[str], deadline_map: dict[str, datetime | None] | None,
    cfg: dict[str, Any],
) -> tuple[list[str], list[str], str | None]:
    """선별형 결과에 마감 필터 적용(4-10-2 확정). deadline_map이 없으면
    필터를 아예 건너뛴다(CSV 미연결 상태) — 조용히 통과시키되 호출측이
    필터가 안 걸렸다는 걸 알 수 있게 표시는 따로 한다.
    반환: (마감 필터 통과 문서 ID, 미상 안내 문구 목록, 위험 고지 문구)"""
    if deadline_map is None:
        return doc_ids, [], None
    if not cfg.get("deadline_filter_default", {}).get("select", False):
        return doc_ids, [], None

    ref = datetime.strptime(cfg["reference_datetime"], "%Y-%m-%d")
    missing_policy = cfg.get("deadline_missing_policy", "show_as_unknown")
    kept, unknown_notes = [], []
    for doc_id in doc_ids:
        ok, note = is_before_deadline(doc_id, deadline_map, ref)
        if ok is False:
            continue  # 마감 지남 — 결과에서 제외
        if ok is None:
            # 4-10-2 확정: 미상은 제외하지 않고 "미상"으로 표시 후 통과
            if missing_policy == "show_as_unknown":
                unknown_notes.append(f"- {doc_id}: {note}")
        kept.append(doc_id)

    disclosure = (
        cfg.get("deadline_filter_disclosure_message")
        if cfg.get("deadline_filter_disclosure") else None
    )
    return kept, unknown_notes, disclosure


def answer_select_by_table(
    question: str, table: list[dict], cfg: dict[str, Any],
    deadline_map: dict[str, datetime | None] | None = None,
) -> Answer:
    """선별형 — G-2(조건 질의) → K-2(코드로 결과 조립) 경로. 생성 단계 안 태움(4-9-8 확정).
    deadline_map을 주면 마감 필터(4-10-2)까지 적용 — base.yaml
    deadline_filter_default.select=true가 기본값."""
    conditions, fully_matched = parse_conditions(question)

    if not conditions:
        return Answer(
            text="조건을 이해하지 못했습니다. 더 구체적으로 말씀해주세요.",
            task_type="select", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True, condition_query=[],
        )
    if not fully_matched:
        # message.txt 7번 확정: 조건 일부만 읽고 나머지를 조용히 무시하지 않는다
        return Answer(
            text=(
                "조건 일부를 인식하지 못했습니다. 인식한 조건: "
                + ", ".join(f"{c.field}{c.operator}{c.value}" for c in conditions)
                + " — 나머지 조건을 더 구체적으로 말씀해주시겠어요?"
            ),
            task_type="select", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
            condition_query=[c.__dict__ for c in conditions],
        )

    results, total, condition_warnings = run_conditions_query(table, conditions)

    doc_ids = [r.document_id for r in results]
    kept_ids, deadline_notes, deadline_disclosure = apply_deadline_filter(doc_ids, deadline_map, cfg)
    kept_set = set(kept_ids)
    filtered_out = len(results) - len(kept_set)
    results = [r for r in results if r.document_id in kept_set]

    lines = [f"- {r.document_id}: {r.value}" for r in results]
    body = "\n".join(lines) if lines else "조건에 맞는 문서가 없습니다."
    if deadline_map is None:
        body += "\n\n[알림] 마감 필터가 연결돼 있지 않습니다 — 마감 지난 사업이 섞여 있을 수 있습니다."
    elif filtered_out:
        body += f"\n\n(마감 지난 사업 {filtered_out}건 제외됨 — 기준일 {cfg['reference_datetime']})"
    if deadline_notes:
        if deadline_disclosure:
            body += f"\n\n[마감일 미상 — {deadline_disclosure}]\n" + "\n".join(deadline_notes)
        else:
            body += "\n\n[마감일 미상]\n" + "\n".join(deadline_notes)
    if condition_warnings:
        body += "\n\n[확인 필요]\n" + "\n".join(f"- {w}" for w in condition_warnings)
    if total > len(results) + filtered_out:
        body += f"\n\n(조건 매칭 전체 {total}건 중 일부만 표시)"

    return Answer(
        text=body, task_type="select", route_matched_rule=None,
        route_is_fallback=False, sources=[], abstained=False,
        condition_query=[c.__dict__ for c in conditions],
        condition_result_doc_ids=[r.document_id for r in results],
    )


def answer_extract_by_table(
    question: str, table: list[dict], store: VectorStore,
    get_embed_client: "Callable[[], EmbeddingClient]",
    get_gen_client: "Callable[[], GenerationClient]",
    cfg: dict[str, Any],
) -> Answer:
    """12필드 추출형 — 기본은 F-0 → G-2 → K-2(값만, 코드로 조립, LLM 안 태움).
    질문이 원문 설명·근거까지 요구하면 F-0 → G → I → J → K-2로 넘어간다
    (message.txt 8번 세 번째 경로). 둘 다 field·document_id가 명확해야만
    타고, 애매하면 QA로 조용히 새지 않고 확인 질문으로 되묻는다."""
    field = detect_field(question)
    doc_id = detect_document_id(question)

    if field is None:
        return Answer(
            text="어느 항목을 확인하고 싶으신가요? (예: 예산, 지역제한, 사업기간, 참가자격 등)",
            task_type="extract", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
        )
    if doc_id is None:
        # 4-14(활성 문서 상태)가 answer() 오케스트레이션에 아직 연결 안 돼 있어
        # (README 확인) 지금은 질문에 문서 ID가 명시돼야만 G-2로 특정할 수 있다.
        return Answer(
            text=f"'{field}'를 어느 문서에서 확인할까요? 문서 ID(예: RFP-000001)를 "
                 f"알려주시면 바로 찾아드릴게요.",
            task_type="extract", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
            condition_query=[{"field": field, "document_id": None}],
        )

    if needs_explanation(question):
        # 값 + 원문 설명 둘 다 필요 — G→I→J→K 경로(검색+생성)로, 검색 범위는
        # 이미 특정된 문서로 좁힌다
        embed_client = get_embed_client()
        gen_client = get_gen_client()
        result = answer_qa_or_extract_by_search(
            question, store, embed_client, gen_client, cfg, document_id=doc_id,
        )
        result.condition_query = [{"field": field, "document_id": doc_id, "route": "G→I→J→K"}]
        return result

    row = lookup_field(table, doc_id, field)
    if row is None:
        return Answer(
            text=f"{doc_id} 문서에서 '{field}' 항목을 찾을 수 없습니다.",
            task_type="extract", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
            condition_query=[{"field": field, "document_id": doc_id}],
        )

    status = row["status"]
    if status == "value_present":
        text = f"{doc_id}의 '{field}': {row.get('answer_normalized') or row.get('answer_raw')}"
        abstained = False
    elif status == "field_absent":
        # 4-12-1 확정: field_absent를 "없음/제한 없음"으로 추정하지 않는다 —
        # 원문에 항목 자체가 없다는 사실만 전달한다.
        text = f"{doc_id} 문서 원문에는 '{field}' 항목 자체가 없습니다."
        abstained = False
    elif status == "not_disclosed":
        text = f"{doc_id} 문서는 '{field}'를 비공개로 명시하고 있습니다."
        abstained = False
    elif status == "external_reference":
        text = (f"{doc_id} 문서의 '{field}'는 외부 공고문·붙임을 확인하라고 "
                f"안내돼 있습니다(원문에 직접 값이 없음).")
        abstained = False
    elif status == "conflict":
        text = (f"{doc_id} 문서의 '{field}'는 원문 내에서 서로 다른 값이 확인됩니다 — "
                f"단정하지 않고 원문을 직접 확인해야 합니다: {row.get('answer_raw')}")
        abstained = True  # 자동으로 하나를 고르지 않음(4-9-8) — 사람 확인 필요
    else:
        text = f"{doc_id} 문서의 '{field}' 상태를 확인하지 못했습니다({status})."
        abstained = True

    return Answer(
        text=text, task_type="extract", route_matched_rule=None, route_is_fallback=False,
        sources=[f"{doc_id} (추출표: {field})"], abstained=abstained,
        condition_query=[{"field": field, "document_id": doc_id, "status": status, "route": "G-2"}],
        condition_result_doc_ids=[doc_id],
    )


def answer_compare_by_table(question: str, table: list[dict]) -> Answer:
    """비교형 — 필드 × 문서 구조로 코드가 조립한다(message.txt 8번 확정).
    LLM이 값을 다시 쓰지 않는다 — G-2 조회 결과를 표로 나열만 한다."""
    field = detect_field(question)
    doc_ids = detect_document_ids(question)

    if field is None:
        return Answer(
            text="어느 항목을 비교할까요? (예: 예산, 지역제한, 사업기간, 참가자격 등)",
            task_type="compare", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
        )
    if len(doc_ids) < 2:
        return Answer(
            text="비교하려면 문서가 두 건 이상 필요합니다. 문서 ID를 두 개 이상 "
                 "알려주시겠어요? (예: RFP-000001, RFP-000002)",
            task_type="compare", route_matched_rule=None, route_is_fallback=False,
            sources=[], abstained=True,
            condition_query=[{"field": field, "document_ids": doc_ids}],
        )

    lines = []
    for doc_id in doc_ids:
        row = lookup_field(table, doc_id, field)
        if row is None:
            lines.append(f"- {doc_id}: 항목을 찾을 수 없음")
            continue
        status = row["status"]
        if status == "value_present":
            lines.append(f"- {doc_id}: {row.get('answer_normalized') or row.get('answer_raw')}")
        elif status == "field_absent":
            lines.append(f"- {doc_id}: 원문에 항목 자체가 없음")
        elif status == "not_disclosed":
            lines.append(f"- {doc_id}: 비공개로 명시됨")
        elif status == "external_reference":
            lines.append(f"- {doc_id}: 외부 공고문·붙임 확인 필요(원문에 직접 값 없음)")
        elif status == "conflict":
            lines.append(f"- {doc_id}: 원문 내 값 상충 — 직접 확인 필요 ({row.get('answer_raw')})")
        else:
            lines.append(f"- {doc_id}: 상태 미확인({status})")

    body = f"[{field}] 비교\n" + "\n".join(lines)

    return Answer(
        text=body, task_type="compare", route_matched_rule=None, route_is_fallback=False,
        sources=[], abstained=False,
        condition_query=[{"field": field, "document_ids": doc_ids, "route": "G-2"}],
        condition_result_doc_ids=doc_ids,
    )


def answer(
    question: str, store: VectorStore,
    get_embed_client: "Callable[[], EmbeddingClient]",
    get_gen_client: "Callable[[], GenerationClient]",
    table: list[dict], cfg: dict[str, Any],
    deadline_map: dict[str, datetime | None] | None = None,
) -> Answer:
    """embed_client/gen_client는 즉시 인스턴스가 아니라 지연 생성 콜러블로 받는다
    (message.txt 8번 확정) — no_search_needed·select 경로는 OpenAI 클라이언트가
    아예 필요 없으므로 호출 자체를 하지 않는다.
    deadline_map은 선별형 마감 필터(4-10-2)용 — 없으면 필터 없이 동작(경고만)."""
    r: RouteResult = route(question, cfg)

    try:
        if r.task_type == "no_search_needed":
            result = Answer(
                text="안녕하세요! RFP 관련 질문을 도와드릴게요.",
                task_type=r.task_type, route_matched_rule=r.matched_rule,
                route_is_fallback=r.is_fallback, sources=[], abstained=False,
            )
        elif r.task_type == "select":
            result = answer_select_by_table(question, table, cfg, deadline_map=deadline_map)
        elif r.task_type == "extract":
            # 이전엔 QA와 같은 검색 경로로 뭉뚱그려져 있었음 — message.txt 8번
            # 확정대로 G-2(코드 조회) 우선, 원문 설명 필요시에만 검색 경로로 이관
            result = answer_extract_by_table(
                question, table, store, get_embed_client, get_gen_client, cfg,
            )
        elif r.task_type == "compare":
            # message.txt 8번 확정 — 필드×문서 구조로 코드가 조립, LLM 안 태움
            result = answer_compare_by_table(question, table)
        elif r.task_type == "qa":
            embed_client = get_embed_client()
            gen_client = get_gen_client()
            result = answer_qa_or_extract_by_search(question, store, embed_client, gen_client, cfg)
        else:
            result = Answer(
                text="확인할 수 없습니다.", task_type=r.task_type,
                route_matched_rule=r.matched_rule, route_is_fallback=r.is_fallback,
                sources=[], abstained=True,
            )
    except Exception as e:
        result = Answer(
            text="처리 중 오류가 발생했습니다.", task_type=r.task_type,
            route_matched_rule=r.matched_rule, route_is_fallback=r.is_fallback,
            sources=[], abstained=True,
            error_stage=r.task_type, error_detail=str(e),
        )

    result.task_type = r.task_type
    result.route_matched_rule = r.matched_rule
    result.route_is_fallback = r.is_fallback
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--index", required=True, help="index_vN 폴더 경로")
    parser.add_argument("--extraction-table", required=False)
    parser.add_argument("--registry", required=False,
                         help="document_registry_v2.json 경로 — 마감 필터용 CSV 매핑에 필요")
    parser.add_argument("--deadline-csv", required=False,
                         help="data_list.csv 경로 — 있어야 선별형 마감 필터(4-10-2)가 켜짐")
    parser.add_argument("--experiment-config", required=False)
    args = parser.parse_args()

    cfg = load_config(args.experiment_config)
    store = VectorStore.load(Path(args.index))
    table = load_extraction_table(Path(args.extraction_table)) if args.extraction_table else []

    deadline_map = None
    if args.deadline_csv and args.registry:
        deadline_map = load_deadline_by_document_id(
            Path(args.deadline_csv), Path(args.registry), cfg,
        )
    elif cfg.get("deadline_filter_default", {}).get("select", False):
        print("⚠️  --deadline-csv/--registry 미지정 — 선별형 마감 필터가 base.yaml엔 "
              "켜져 있는데 이 실행에선 꺼진 채로 돕니다(마감 지난 사업이 섞일 수 있음).")

    # 지연 생성 — no_search_needed·select 경로에서는 아예 호출되지 않는다
    _cache: dict[str, Any] = {}

    def get_embed_client() -> EmbeddingClient:
        if "embed" not in _cache:
            _cache["embed"] = EmbeddingClient(cfg)
        return _cache["embed"]

    def get_gen_client() -> GenerationClient:
        if "gen" not in _cache:
            _cache["gen"] = GenerationClient(cfg)
        return _cache["gen"]

    result = answer(args.question, store, get_embed_client, get_gen_client, table, cfg,
                     deadline_map=deadline_map)

    print(json.dumps(
        {
            "question": args.question,
            "task_type": result.task_type,
            "route_matched_rule": result.route_matched_rule,
            "route_is_fallback": result.route_is_fallback,
            "answer": result.text,
            "sources": result.sources,
            "abstained": result.abstained,
        },
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
