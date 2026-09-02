"""
질문 답하기 진입점 — F-0(분기) → G/G-2(검색·조건질의/구조화 조회)
→ I(컨텍스트) → J(생성) → K(출처) → K-2(형식화).

사용 예:
  source /srv/rfp/venv/bin/activate
  export RAG_ROOT=/srv/rfp
  export OPENAI_API_KEY=...        # 값은 출력하지 않는다(존재 여부만 확인)
  python3 answer_pipeline.py --question "5억 이상인 사업 알려줘"

[2026-09-02 확정 정책]
- 문서 특정: doc_resolver의 5단계 우선순위(정확한 ID → active_document_id →
  기관명+사업명 → 기관명 → 사업명 단일후보). 모호하면 임의로 고르지 않고 되묻는다.
- QA형이라도 질문에 확정된 구조화 필드가 있으면 구조화 자료를 **함께** 쓴다.
  마감일 → identity_v2, 12필드 → extraction_table_v3, 설명 보완 → chunks_v3.
- 구조화 값은 공식 확정 값이고, 청크는 설명·문맥 보완 근거다. 모델이 청크를
  보고 확정 값을 바꾸지 못하게 프롬프트에서 구분한다. 단, 구조화 자료 자체가
  conflict면 하나를 고르지 않고 충돌로 답한다.
- 응답의 최상위 구조는 질문 유형과 무관하게 동일하다(하루님 채점기 스키마).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from config import (  # noqa: E402
    load_config, index_dir, extraction_table_path, extraction_metadata_path,
    document_registry_path, identity_path, chunks_path,
)
from vector_store import (  # noqa: E402
    VectorStore, ChunkMetadata, validate_index_tag, IndexTagError,
)
from chunk_locator import ChunkLocator  # noqa: E402
from embedding_client import EmbeddingClient  # noqa: E402
from generation_client import GenerationClient  # noqa: E402
from router import route, RouteResult, NO_SEARCH_SYSTEM_HELP  # noqa: E402
from table_query import (  # noqa: E402
    load_extraction_table, parse_conditions, run_conditions_query, QueryResult,
    detect_field, detect_fields, needs_explanation, lookup_field,
    detect_deadline_question, detect_deadline_eligibility_question, needs_deadline_data,
    classify_region_restriction, NON_VALUE_STATUS,
)
from identity_metadata import (  # noqa: E402
    IdentityIndex, load_identity, is_before_deadline, deadline_citation,
    deadline_evidence, reference_datetime_from_config, DEADLINE_COLUMN,
)
from doc_resolver import (  # noqa: E402
    DocumentResolution, resolve_document, resolve_documents_for_compare,
    detect_unknown_orgs, has_anaphora, detect_document_id, detect_document_ids,
    RESOLVE_NONE,
)
from pricing import Usage  # noqa: E402

# ---------------------------------------------------------------------------
# 내부 실행 경로(route) — 평가셋 라벨(task_type)과 별개인 기록용 값
# ---------------------------------------------------------------------------
ROUTE_SELECT = "추출테이블_문서선별"
ROUTE_EXTRACT_VALUE = "추출테이블_값조회"
ROUTE_COMPARE = "추출테이블_비교조립"
ROUTE_SEARCH_LLM = "chunks검색_LLM답변"
ROUTE_STRUCTURED_LLM = "구조화자료_결합_LLM답변"
ROUTE_IDENTITY_VALUE = "identity_v2_값조회"
ROUTE_CLARIFY = "애매_되묻기"
ROUTE_GREETING = "검색불필요_인사응답"
ROUTE_SYSTEM_HELP = "검색불필요_사용법안내"

ALL_ROUTES = (
    ROUTE_SELECT, ROUTE_EXTRACT_VALUE, ROUTE_COMPARE, ROUTE_SEARCH_LLM,
    ROUTE_STRUCTURED_LLM, ROUTE_IDENTITY_VALUE, ROUTE_CLARIFY,
    ROUTE_GREETING, ROUTE_SYSTEM_HELP,
)

SYSTEM_HELP_TEXT = (
    "이 시스템은 RFP(제안요청서) 100건을 근거로 답하는 검색·질의 도구입니다. "
    "다음처럼 물어보실 수 있습니다.\n"
    "1. 조건에 맞는 공고 찾기 — 예: \"예산 5억 이상, 지역제한 없는 사업 알려줘\"\n"
    "2. 특정 문서의 12필드 조회 — 예: \"RFP-000038의 예산 알려줘\" "
    "(12필드: 사업 개요·사업분야·공고일·사업기간·예산·참가 자격(면허·실적)·"
    "지역제한·컨소시엄 요건·평가 배점·제출 방식·필수 제출 서류·과업 범위)\n"
    "3. 문서 비교 — 예: \"RFP-000038이랑 RFP-000043 예산이랑 사업기간 비교해줘\"\n"
    "4. 일반 문서 질문 — 예: \"이 사업의 추진배경을 설명해줘\" "
    "(원문 청크를 근거로 답하고 출처를 함께 표시합니다)\n"
    "5. 모호할 때 재확인 — 기관명이나 사업명이 여러 문서에 걸리면 임의로 고르지 않고 "
    "후보를 보여드리며 다시 여쭤봅니다. 문서 ID(RFP-000001 형식)로 알려주시면 가장 정확합니다.\n"
    "마감일은 identity_v2, 12필드 값은 추출표 v3, 원문 설명은 청크 v3를 근거로 씁니다."
)

_ANAPHORA_MARKERS_DOC = "그 사업 / 이 사업 / 거기 / 그거"


@dataclass
class SessionState:
    active_document_id: str | None = None


@dataclass
class Answer:
    """모든 질문 유형이 공유하는 응답 구조(질문마다 다른 구조를 만들지 않는다)."""
    text: str
    task_type: str
    route_matched_rule: str | None = None
    route_is_fallback: bool = False
    sources: list[str] = field(default_factory=list)
    abstained: bool = False
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    condition_query: list[dict] | None = None
    condition_result_doc_ids: list[str] = field(default_factory=list)
    error_stage: str | None = None
    error_detail: str | None = None
    route: str | None = None
    structured_answer: Any = None
    contexts: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    selected_document_ids: list[str] = field(default_factory=list)
    failure: str | None = None
    # 어떤 공식 자료를 실제로 썼는지(버전 포함) — 답변 근거의 출처 추적용
    used_sources: list[dict] = field(default_factory=list)
    # 결함 1-4 — citations 를 어떻게 골랐는지(모델 선언·미지 ID 등) 진단용
    citation_diagnostics: dict | None = None
    resolution: dict | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    cost_detail: dict | None = None


# ---------------------------------------------------------------------------
# 값 표시 정리 — 최종 답변 텍스트(answer)는 "값 자체"여야 한다
# ---------------------------------------------------------------------------

# 원문에서 딸려온 문서 서식 흔적(항목 번호·레이블). 값의 일부가 아니다.
_ENUM_PREFIX_RE = re.compile(
    r"^\s*(?:[가-힣]\s*[.)]|\(?\d{1,2}\s*[.)]|[①-⑳]|[ⅰⅱⅲⅳⅴ]\s*[.)])\s*"
)
# "예산소요액 :", "사업금액:" 처럼 값 앞에 붙은 짧은 레이블
_LABEL_PREFIX_RE = re.compile(r"^\s*[가-힣A-Za-z0-9 ()·]{1,14}\s*[:：]\s*")
# 금액 앞의 "금"
_AMOUNT_WORD_RE = re.compile(r"^\s*금\s+(?=[\d￦₩])")


def clean_value_text(raw: Any) -> str:
    """추출된 원문 값에서 **문서 서식 흔적만** 떼어낸 최종 답변 텍스트.

    ⚠️ 결함 수정(2026-09-02): 예전엔 answer 에 우리가 만든 장식
    ("RFP-000043의 '예산': ")을 붙여서, 응답 계약상 "최종 답변 텍스트"여야 할 값이
    아니게 됐다(하루님 채점기 grade_short_answer 는 정규화 정확 일치를 본다).
    장식을 없애고, 원문에서 딸려온 항목 번호("다."), 레이블("예산소요액 :"),
    금액 앞 "금" 만 제거한다.

    ★ 값의 내용은 바꾸지 않는다 — 단위·괄호 부연·문구는 원문 그대로 남긴다.
    ★ 제거 후 빈 문자열이 되면 원문을 그대로 돌려준다(과잉 제거 방지).

    >>> clean_value_text("다. 예산소요액 : 금 248,796천원(부가세 포함)")
    '248,796천원(부가세 포함)'
    >>> clean_value_text("352,000,000원(부가가치세 포함)")
    '352,000,000원(부가가치세 포함)'
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    for _ in range(3):  # "다. 예산소요액 : 금 " 처럼 겹쳐 있을 수 있다
        before = text
        text = _ENUM_PREFIX_RE.sub("", text)
        text = _LABEL_PREFIX_RE.sub("", text)
        text = _AMOUNT_WORD_RE.sub("", text)
        if text == before:
            break
    text = text.strip()
    return text or str(raw).strip()


def as_list_value(row: dict) -> list[str] | None:
    """추출표 값이 목록형이면 문자열 배열로. 아니면 None.

    공식 추출표는 목록형 필드(필수 제출 서류 등)의 answer_normalized 를 JSON 배열
    문자열로 담는다. 응답 계약상 목록형은 structured_answer 가 배열이어야 한다."""
    norm = row.get("answer_normalized")
    if isinstance(norm, list):
        return [str(x) for x in norm]
    if isinstance(norm, str):
        t = norm.strip()
        if t.startswith("[") and t.endswith("]"):
            try:
                parsed = json.loads(t)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
    raw = row.get("answer_raw")
    if isinstance(raw, str) and "\n" in raw.strip():
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        if len(parts) >= 2:
            return parts
    return None


# ---------------------------------------------------------------------------
# 근거 위치 → citations
# ---------------------------------------------------------------------------

def _deepest_section(loc: dict) -> str:
    """가장 구체적인 실제 절 이름. section_path의 마지막 항목 > heading 순."""
    sp = loc.get("section_path") or []
    if isinstance(sp, list) and sp:
        last = sp[-1]
        if isinstance(last, dict) and last.get("title"):
            return str(last["title"])
        if isinstance(last, str) and last:
            return last
    return str(loc.get("heading") or "")


def location_to_citation(document_id: str, loc: dict, source: str) -> dict:
    """추출표 위치 객체 → {document, section, ref_no} (+ 재탐색용 부가 정보).

    ref_no에는 문단·표·항목 번호와 줄 위치를 함께 넣어 원문에서 다시 찾을 수
    있게 한다(4-3 확정). block_index를 비우지 않는다.
    """
    section = _deepest_section(loc)
    block_type = loc.get("block_type") or "block"
    block_index = loc.get("block_index")
    line = loc.get("line", loc.get("line_start"))
    line_end = loc.get("line_end")

    parts: list[str] = []
    if block_index is not None:
        parts.append(f"{block_type} {block_index}")
    else:
        parts.append(str(block_type))
    if line is not None:
        parts.append(f"line {line}" if not line_end or line_end == line
                     else f"line {line}-{line_end}")
    return {
        "document": document_id,
        "section": section,
        "ref_no": " · ".join(p for p in parts if p),
        "line": line,
        "block_type": block_type,
        "block_index": block_index,
        "source": source,
    }


def row_citations(document_id: str, row: dict | None, source: str = "extraction_table_v3") -> list[dict]:
    """representative_location과 additional_locations를 **모두** 근거로 만든다.
    첫 위치만 남기지 않는다(4-3 확정). conflict 행에서 특히 중요."""
    if row is None:
        return []
    out: list[dict] = []
    rep = row.get("representative_location")
    if isinstance(rep, dict) and rep:
        out.append(location_to_citation(document_id, rep, source))
    for loc in row.get("additional_locations") or []:
        if isinstance(loc, dict) and loc:
            out.append(location_to_citation(document_id, loc, source))
    if not out:
        out.append({
            "document": document_id,
            "section": "",
            "ref_no": row.get("field_name", ""),
            "line": None,
            "block_type": None,
            "block_index": None,
            "source": source,
        })
    # 같은 위치가 여러 번 나오면 한 번만
    seen: set[tuple] = set()
    deduped = []
    for c in out:
        key = (c["document"], c["section"], c["ref_no"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def format_source(m: ChunkMetadata) -> str:
    """4-13 형식: 문서명 > 장절 > 표/문단 번호."""
    parts = [m.document_name, m.chapter]
    if m.chunk_type == "table" and m.table_idx is not None:
        label = f"표 {m.table_idx}"
        if m.part is not None and m.of is not None:
            label += f" ({m.part}/{m.of})"
        parts.append(label)
    elif m.location_label:
        parts.append(m.location_label)
    return " > ".join(p for p in parts if p)


def chunk_to_record(m: ChunkMetadata) -> dict:
    """contexts/retrieved에 넣는 청크 정보 — 실제 청크 값을 그대로 보존한다.
    ⚠️ 텍스트 청크의 block_index를 None으로 지우지 않는다(4-3 확정)."""
    return {
        "chunk_id": m.chunk_id,
        "document_id": m.document_id,
        "section_path": m.section_path,
        "section_paths": m.section_paths,
        "block_type": m.chunk_type,
        "block_index": m.block_index,
        "table_idx": m.table_idx,
        "part": m.part,
        "of": m.of,
        "md_line_start": m.md_line_start,
        "md_line_end": m.md_line_end,
        "location_label": m.location_label,
        "search_text": m.text,
    }


def chunk_to_citation(m: ChunkMetadata) -> dict:
    """검색 경로 출처 — {document, section, ref_no} + 재탐색 정보."""
    section = ""
    if m.section_path:
        last = m.section_path[-1]
        section = last.get("title") if isinstance(last, dict) else str(last)
    section = section or m.chapter
    if m.chunk_type == "table" and m.table_idx is not None:
        ref = f"table {m.table_idx}"
        if m.part and m.of:
            ref += f" ({m.part}/{m.of})"
    else:
        ref = f"{m.chunk_type} {m.block_index}" if m.block_index is not None else m.chunk_type
    if m.md_line_start is not None:
        ref += (f" · line {m.md_line_start}"
                if m.md_line_end in (None, m.md_line_start)
                else f" · line {m.md_line_start}-{m.md_line_end}")
    return {
        "document": m.document_id,
        "section": section,
        "ref_no": ref,
        "line": m.md_line_start,
        "block_type": m.chunk_type,
        "block_index": m.block_index,
        "source": "chunks_v3",
    }


# 하위호환 별칭 (기존 테스트·외부 호출용)
def table_row_to_citation(document_id: str, row: dict) -> dict:
    cites = row_citations(document_id, row)
    return cites[0]


# ---------------------------------------------------------------------------
# 구조화 근거 조립
# ---------------------------------------------------------------------------

@dataclass
class StructuredEvidence:
    kind: str                      # "deadline" | "field"
    document_id: str
    field_name: str | None
    status: str
    value_raw: Any = None
    value_normalized: Any = None
    answer_text: str = ""          # 코드가 확정한 문구(모델이 바꾸면 안 되는 부분)
    short_text: str = ""           # 비교표 한 줄용 요약(상태 정보를 잃지 않는다)
    prompt_text: str = ""          # LLM에 넣을 "공식 확정 값" 블록
    citations: list[dict] = field(default_factory=list)
    abstained: bool = False
    structured: dict = field(default_factory=dict)
    used_source: dict = field(default_factory=dict)


def _source_tag(cfg: dict[str, Any], name: str) -> dict:
    return {
        "identity_v2": {
            "source": "identity_v2",
            "file": "document_identity_v2.csv",
            "registry_version": cfg.get("document_registry_version", cfg.get("corpus")),
            "column": DEADLINE_COLUMN,
        },
        "extraction_table": {
            "source": "extraction_table",
            "extraction_version": cfg.get("extraction_version"),
            "schema_version": cfg.get("schema_version"),
            "corpus_version": cfg.get("corpus"),
            "registry_version": cfg.get("document_registry_version", cfg.get("corpus")),
        },
        "chunks": {
            "source": "chunks",
            "chunking_version": cfg.get("chunking_version"),
            "corpus_version": cfg.get("corpus"),
        },
    }[name]


def build_deadline_evidence(
    document_id: str, identity: IdentityIndex, cfg: dict[str, Any],
    eligibility: bool = False,
) -> StructuredEvidence:
    """identity_v2의 마감일 값·근거 위치·기준 시각 정책까지 담은 구조화 근거.

    ⚠️ answer 는 "최종 답변 텍스트"다 — 우리가 만든 장식(문서ID·필드명 접두)을 붙이지
    않는다. 안내 문구·원문 설명은 structured_answer 쪽에 담는다(응답 계약).
    """
    ev = deadline_evidence(identity, document_id)
    rec = identity.get(document_id)
    cite = deadline_citation(document_id)
    used = _source_tag(cfg, "identity_v2")

    if rec is None:
        return StructuredEvidence(
            kind="deadline", document_id=document_id, field_name="입찰 참여 마감일",
            status="identity_missing",
            short_text="identity_v2에 문서 없음(확인 불가)",
            answer_text=(f"{document_id}는 identity_v2에 없어 마감일을 확인할 수 없습니다."),
            prompt_text=f"- {document_id} 마감일: identity_v2에 문서 없음(확인 불가)",
            citations=[cite], abstained=True, structured=ev, used_source=used,
        )
    if rec.bid_deadline is None:
        return StructuredEvidence(
            kind="deadline", document_id=document_id, field_name="입찰 참여 마감일",
            status="not_disclosed",
            short_text="identity_v2에 값 없음(미상)",
            answer_text="입찰 참여 마감일이 identity_v2에 없습니다(미상).",
            prompt_text=f"- {document_id} 마감일: identity_v2에 값 없음(미상)",
            citations=[cite], abstained=True, structured=ev, used_source=used,
        )

    ref_dt = reference_datetime_from_config(cfg)
    deadline_str = rec.bid_deadline.strftime("%Y-%m-%d %H:%M")
    structured = dict(ev)
    structured.update({
        "reference_datetime": ref_dt.strftime("%Y-%m-%d"),
        "reference_datetime_source": cfg.get("reference_datetime_source", "external"),
    })

    if eligibility:
        ok, note = is_before_deadline(document_id, identity, ref_dt)
        structured["is_before_deadline"] = ok
        if ok is True:
            text = f"참여 가능합니다(마감 {deadline_str})."
        elif ok is False:
            text = f"참여할 수 없습니다(마감 {deadline_str})."
        else:
            text = f"마감일이 미상이라 참여 가능 여부를 단정할 수 없습니다({note})."
        disclosure = cfg.get("deadline_filter_disclosure_message")
        if disclosure and cfg.get("deadline_filter_disclosure"):
            # ⚠️ 고지 문구는 answer 본문이 아니라 구조화 응답에 싣는다 — answer 는
            #    채점·비교 대상인 "최종 답변 텍스트"라 값 외 문구가 섞이면 안 된다.
            structured["notice"] = disclosure
    else:
        text = deadline_str

    return StructuredEvidence(
        kind="deadline", document_id=document_id, field_name="입찰 참여 마감일",
        status="value_present", value_raw=rec.bid_deadline_raw,
        value_normalized=rec.bid_deadline.strftime("%Y-%m-%d"),
        short_text=deadline_str, answer_text=text,
        prompt_text=(f"- {document_id} 입찰 참여 마감일(공식 확정): {deadline_str} "
                     f"[identity_v2 · {DEADLINE_COLUMN}]\n"
                     f"- 평가 기준 시각(고정): {ref_dt:%Y-%m-%d} "
                     f"(reference_datetime, now() 사용 안 함)"),
        citations=[cite], abstained=False, structured=structured, used_source=used,
    )


_STATUS_TEXT = {
    "field_absent": "원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음).",
    "external_reference": "외부 공고문·붙임을 직접 확인해야 합니다(원문에 직접 값이 없음).",
    "not_disclosed": "원문에서 비공개로 명시하고 있습니다.",
    "extraction_failed": "값을 확인하지 못했습니다(추출 실패) — 정상 값으로 취급하지 않습니다.",
    "review_required": "검수 대기 상태입니다 — 정상 값으로 취급하지 않습니다.",
}


def build_field_evidence(
    table: list[dict], document_id: str, field_name: str, cfg: dict[str, Any],
    locator: ChunkLocator | None = None,
) -> StructuredEvidence:
    """추출표 12필드 값·상태·위치를 담은 구조화 근거. 상태를 무시하지 않는다.

    [2026-09-02 결함 1-4] 근거는 **실제로 답에 쓴 위치만** 남긴다.
      - 값을 답한 경우  → representative_location 하나
      - conflict 인 경우 → 서로 다른 값의 위치를 모두(두 문구가 모두 답의 일부)
    좌표는 팀 확정 규약(청크 location_label·line)으로 변환한다 — 변환에 실패하면
    근거를 지어내지 않고 비운다.
    """
    used = _source_tag(cfg, "extraction_table")
    row = lookup_field(table, document_id, field_name)
    if row is None:
        return StructuredEvidence(
            kind="field", document_id=document_id, field_name=field_name,
            status="row_missing", short_text="추출표에 행 없음",
            answer_text=f"{document_id} 문서에서 '{field_name}' 항목을 찾을 수 없습니다.",
            prompt_text=f"- {document_id} / {field_name}: 추출표에 행 없음",
            citations=[], abstained=True, used_source=used,
        )

    status = row["status"]
    value_raw = row.get("answer_raw")
    value_norm = row.get("answer_normalized")
    structured: dict[str, Any] = {
        "document_id": document_id, "field_name": field_name, "status": status,
        "answer_raw": value_raw, "answer_normalized": value_norm,
    }

    def _cite(loc: Any) -> list[dict]:
        """추출표 위치 → 인용 좌표.

        1순위: 청크 좌표(팀 확정 규약 — ref_no=location_label, line 범위)
        2순위: 추출표 위치 자체(청크 매핑 실패 시). **실재하는 위치는 버리지 않는다** —
               특히 conflict 의 두 위치는 답의 일부라 반드시 보존해야 한다.
        위치 정보 자체가 없으면 빈 목록(근거를 지어내지 않는다)."""
        if not isinstance(loc, dict) or not loc:
            return []
        if locator is not None:
            c = locator.citation_for_location(document_id, loc, field=field_name)
            if c:
                return [c]
        line = loc.get("line", loc.get("line_start"))
        section = _deepest_section(loc)
        if line is None and not section:
            return []
        block_type = loc.get("block_type") or "block"
        block_index = loc.get("block_index")
        ref = f"{block_type} {block_index}" if block_index is not None else str(block_type)
        return [{
            "document": document_id, "section": section, "ref_no": ref,
            "line": line, "line_end": loc.get("line_end", line),
            "field": field_name, "source": "extraction_table_v3(raw_location)",
        }]

    if status == "conflict":
        pieces = [p.strip() for p in str(value_raw or "").split("---") if p.strip()]
        # 충돌은 서로 다른 두 문구가 모두 답의 일부다 → 두 위치 모두 실제 근거
        cites = _cite(row.get("representative_location"))
        for loc in (row.get("additional_locations") or []):
            cites.extend(_cite(loc))
        seen: set[tuple] = set()
        cites = [c for c in cites
                 if not ((c["document"], c["ref_no"], c["line"]) in seen
                         or seen.add((c["document"], c["ref_no"], c["line"])))]
        loc_lines = [f"    · {c['section'] or '(절 미상)'} / {c['ref_no']}" for c in cites]
        body = "\n".join(f"  - 내용 {i + 1}: {p}" for i, p in enumerate(pieces)) or f"  - {value_raw}"
        text = ("원문 안에서 서로 다른 내용이 확인됩니다 — 하나로 단정하지 않습니다.\n"
                + body + ("\n  근거 위치(모두 보존):\n" + "\n".join(loc_lines) if loc_lines else ""))
        structured["conflict_values"] = pieces
        structured["conflict_locations"] = cites
        return StructuredEvidence(
            kind="field", document_id=document_id, field_name=field_name,
            status=status, value_raw=value_raw, value_normalized=value_norm,
            short_text="원문 내 값 상충 — 직접 확인 필요 (" + " / ".join(pieces) + ")",
            answer_text=text,
            prompt_text=(f"- {document_id} / {field_name}: **상충(conflict)**. 서로 다른 "
                         f"내용이 두 곳 이상에 있습니다. 하나를 고르지 말고 둘 다 제시하세요.\n"
                         + "\n".join(f"    {p}" for p in pieces)),
            citations=cites, abstained=True, structured=structured, used_source=used,
        )

    if status in NON_VALUE_STATUS:
        note = _STATUS_TEXT.get(status, f"상태를 확인하지 못했습니다({status}).")
        abstain = status in ("extraction_failed", "review_required")
        # 값이 없다고 답한 근거 위치는 있으면 남기고, 없으면 비운다(지어내지 않음)
        cites = _cite(row.get("representative_location"))
        return StructuredEvidence(
            kind="field", document_id=document_id, field_name=field_name,
            status=status, value_raw=value_raw, value_normalized=value_norm,
            short_text=note, answer_text=note,
            prompt_text=f"- {document_id} / {field_name}: 값 없음 — 상태 {status} ({note})",
            citations=cites, abstained=abstain, structured=structured, used_source=used,
        )

    display = clean_value_text(value_raw if value_raw not in (None, "") else value_norm)
    list_value = as_list_value(row)
    if list_value:
        structured["items"] = list_value
    short = display
    if field_name == "지역제한":
        verdict, reason = classify_region_restriction(row)
        structured["region_restriction"] = verdict
        structured["region_restriction_reason"] = reason
        short = f"{display} [판정: {verdict}]"
    return StructuredEvidence(
        kind="field", document_id=document_id, field_name=field_name,
        status=status, value_raw=value_raw, value_normalized=value_norm,
        short_text=short, answer_text=display,
        prompt_text=f"- {document_id} / {field_name}(공식 확정 값): {display}",
        citations=_cite(row.get("representative_location")),
        abstained=False, structured=structured, used_source=used,
    )


def detect_structured_need(question: str, table: list[dict] | None) -> tuple[bool, str | None]:
    """QA형 질문에 확정된 구조화 필드가 들어 있는지. (마감일 필요?, 12필드 이름)"""
    return needs_deadline_data(question), (detect_field(question) if table else None)


# ---------------------------------------------------------------------------
# 검색 경로
# ---------------------------------------------------------------------------

def _search(
    question: str, store: VectorStore, embed_client: EmbeddingClient,
    cfg: dict[str, Any], document_id: str | None,
) -> list[tuple[ChunkMetadata, float]]:
    if store is None:
        return []
    query_vec = embed_client.embed_query(question)
    return store.search(query_vec, top_k=cfg.get("top_k", 5),
                        active_only=True, document_id=document_id)


def answer_qa_or_extract_by_search(
    question: str, store: VectorStore, embed_client: EmbeddingClient,
    gen_client: GenerationClient, cfg: dict[str, Any],
    document_id: str | None = None,
    structured: list[StructuredEvidence] | None = None,
    task_type: str = "qa",
) -> Answer:
    """G(검색) → I(컨텍스트) → J(생성) → K(출처).
    structured가 있으면 공식 확정 값을 프롬프트에 함께 넣고, 코드가 만든
    확정 문구를 답변 맨 앞에 고정한다(모델이 값을 바꾸지 못하게)."""
    structured = structured or []
    used_sources = [ev.used_source for ev in structured if ev.used_source]
    structured_citations: list[dict] = []
    for ev in structured:
        structured_citations.extend(ev.citations)

    results = _search(question, store, embed_client, cfg, document_id)

    normal = [(m, s) for m, s in results if not m.table_degraded]
    degraded = [(m, s) for m, s in results if m.table_degraded]
    if results:
        used_sources.append(_source_tag(cfg, "chunks"))

    lead = "\n".join(ev.answer_text for ev in structured if ev.answer_text)
    any_structured_abstain = any(ev.abstained for ev in structured)

    def _no_llm_route() -> str:
        """생성 API를 안 태웠을 때의 실제 경로 이름."""
        if structured and all(ev.kind == "deadline" for ev in structured):
            return ROUTE_IDENTITY_VALUE
        return ROUTE_EXTRACT_VALUE

    if not normal:
        # 원문 근거가 없다 — 구조화 값만으로 답하거나(있으면), 모른다고 한다
        if structured:
            body = lead
            if degraded:
                body += ("\n\n[표 손상으로 값 생성 안 함 — 원문 위치만 안내]\n"
                         + "\n".join(f"- {format_source(m)}" for m, _ in degraded))
            return Answer(
                text=body, task_type=task_type, abstained=any_structured_abstain,
                sources=[f"{ev.document_id} ({ev.used_source.get('source')})" for ev in structured],
                route=_no_llm_route(),   # 생성 호출이 없었으니 LLM 경로로 기록하지 않는다
                structured_answer=_merge_structured(structured),
                retrieved=[chunk_to_record(m) for m, _ in results],
                retrieved_chunk_ids=[m.chunk_id for m, _ in results],
                retrieved_scores=[s for _, s in results],
                contexts=[],
                # 손상 표는 값 생성에 쓰지 않았으므로 인용하지 않는다(위치만 sources 로 안내)
                citations=structured_citations,
                citation_diagnostics={"protocol": "no_generation",
                                      "structured_citations": len(structured_citations),
                                      "chunk_citations": 0},
                selected_document_ids=[ev.document_id for ev in structured],
                used_sources=used_sources,
            )
        if degraded:
            # 손상된 표(빈 셀 60% 초과)만 검색된 경우 — 값을 만들지 않는다(4-7-1 확정).
            # ⚠️ 결함 수정(2026-09-02): 값을 만들지 않았는데도 손상 청크 전체를
            #    citations 에 넣고 있었다. citations 는 "답의 실제 근거"만 담는 자리다
            #    (결함 1-4와 같은 원칙) — 여기서는 답 자체가 "값을 만들지 않았다"이므로
            #    근거가 없다. 원문 확인 위치는 sources 로, 검색 기록은 retrieved 로 남긴다.
            text = ("이 항목은 표가 손상되어(빈 셀 60% 초과) 값을 추정하지 않습니다. "
                    "아래 원문 위치를 직접 확인해주세요:\n"
                    + "\n".join(f"- {format_source(m)}" for m, _ in degraded))
            return Answer(
                text=text, task_type=task_type, abstained=True,
                sources=[format_source(m) for m, _ in degraded],
                route=ROUTE_SEARCH_LLM,
                retrieved=[chunk_to_record(m) for m, _ in results],
                retrieved_chunk_ids=[m.chunk_id for m, _ in results],
                retrieved_scores=[s for _, s in results],
                contexts=[],
                citations=[],
                citation_diagnostics={
                    "protocol": "no_generation_degraded_only",
                    "reason": ("정상 청크 0개 · 손상 표만 검색됨 — 값을 만들지 않았으므로 "
                               "인용할 근거가 없다. 확인 위치는 sources, 검색 기록은 retrieved."),
                    "degraded_chunks": len(degraded),
                    "chunk_citations": 0, "structured_citations": 0,
                },
                used_sources=used_sources,
            )
        return Answer(
            text="확인할 수 없습니다.", task_type=task_type, abstained=True,
            route=ROUTE_SEARCH_LLM, used_sources=used_sources,
        )

    context_texts = [f"[출처: {format_source(m)}]\n{m.text}" for m, _ in normal]
    structured_context = "\n".join(ev.prompt_text for ev in structured if ev.prompt_text) or None
    generated = gen_client.generate(
        question, context_texts, structured_context=structured_context,
    )

    # ⭐ 결함 1-4: 검색된 청크를 전부 citations 로 복사하지 않는다.
    #   모델이 USED_EVIDENCE 줄로 밝힌 근거 번호만 실제 청크 좌표로 바꿔 인용한다.
    #   contexts/retrieved 는 진단용으로 그대로 보존한다(역할이 다르다).
    used_ids = getattr(gen_client, "last_used_evidence", None)
    chunk_citations: list[dict] = []
    unknown_ids: list[int] = []
    if used_ids:
        for eid in used_ids:
            if 1 <= eid <= len(normal):
                chunk_citations.append(chunk_to_citation(normal[eid - 1][0]))
            else:
                unknown_ids.append(eid)   # 존재하지 않는 근거 ID는 인용으로 인정하지 않는다
    citation_diagnostics = {
        "protocol": ("missing" if used_ids is None
                     else ("none_declared" if used_ids == [] else "ok")),
        "declared_evidence_ids": used_ids,
        "context_count": len(normal),
        "unknown_evidence_ids": unknown_ids,
        "chunk_citations": len(chunk_citations),
        "structured_citations": len(structured_citations),
    }

    if structured:
        # ⚠️ 확정된 구조화 값이 있으면 **그 값이 최종 답변**이다. 모델이 쓴 원문 설명은
        #    답변 본문에 덧붙이지 않고 structured_answer.explanation 으로 함께 전달한다.
        #    (응답 계약: answer 는 채점·비교 대상인 "최종 답변 텍스트" — 값 외의 문장이
        #     섞이면 값이 맞아도 "정답+덧붙임"으로 취급된다.)
        answer_text = lead or generated
        explanation = generated if lead else None
    else:
        answer_text = generated
        explanation = None

    sources = [format_source(m) for m, _ in normal]
    if degraded:
        # 손상된 표는 값을 만들지 않고 위치만 안내한다 — 인용(citations)에는 넣지 않는다.
        sources += [format_source(m) for m, _ in degraded]

    return Answer(
        text=answer_text, task_type=task_type,
        abstained=any_structured_abstain,
        sources=sources,
        route=ROUTE_STRUCTURED_LLM if structured else ROUTE_SEARCH_LLM,
        retrieved_chunk_ids=[m.chunk_id for m, _ in results],
        retrieved_scores=[s for _, s in results],
        retrieved=[chunk_to_record(m) for m, _ in results],
        contexts=[chunk_to_record(m) for m, _ in normal],
        citations=structured_citations + chunk_citations,
        citation_diagnostics=citation_diagnostics,
        structured_answer=(_merge_structured(structured, degraded,
                                            explanation=explanation)
                           if structured else None),
        selected_document_ids=[ev.document_id for ev in structured] or (
            [document_id] if document_id else []),
        used_sources=used_sources,
    )


def _merge_structured(structured: list[StructuredEvidence],
                      degraded: list | None = None,
                      explanation: str | None = None) -> Any:
    """구조화 응답 조립 — 목록형 필드는 배열 그대로(응답 계약).

    필드가 하나뿐이고 그 값이 목록이면 structured_answer 는 그 **배열**이어야 한다
    (하루님 check_format: list 답변은 structured_answer 가 배열)."""
    if len(structured) == 1 and not explanation and not degraded:
        st = structured[0].structured or {}
        items = st.get("items")
        if isinstance(items, list) and items:
            return items
    out: dict[str, Any] = {ev.field_name: ev.structured for ev in structured}
    if explanation:
        out["explanation"] = explanation
    if degraded:
        out["degraded_tables"] = [format_source(m) for m, _ in degraded]
    return out


# ---------------------------------------------------------------------------
# 선별형
# ---------------------------------------------------------------------------

def apply_deadline_filter(
    doc_ids: list[str], identity: IdentityIndex | None, cfg: dict[str, Any],
) -> tuple[list[str], list[str], str | None]:
    if identity is None:
        return doc_ids, [], None
    if not cfg.get("deadline_filter_default", {}).get("select", False):
        return doc_ids, [], None

    ref = reference_datetime_from_config(cfg)
    missing_policy = cfg.get("deadline_missing_policy", "show_as_unknown")
    kept, unknown_notes = [], []
    for doc_id in doc_ids:
        ok, note = is_before_deadline(doc_id, identity, ref)
        if ok is False:
            continue
        if ok is None and missing_policy == "show_as_unknown":
            unknown_notes.append(f"- {doc_id}: {note}")
        kept.append(doc_id)

    disclosure = (cfg.get("deadline_filter_disclosure_message")
                  if cfg.get("deadline_filter_disclosure") else None)
    return kept, unknown_notes, disclosure


def answer_select_by_table(
    question: str, table: list[dict], cfg: dict[str, Any],
    identity: IdentityIndex | None = None,
    locator: ChunkLocator | None = None,
) -> Answer:
    """선별형 — G-2(조건 질의) → K-2(코드 조립). 생성 단계 안 태움."""
    conditions, fully_matched = parse_conditions(question)

    if not conditions:
        return Answer(
            text="조건을 이해하지 못했습니다. 더 구체적으로 말씀해주세요.",
            task_type="select", abstained=True, condition_query=[], route=ROUTE_CLARIFY,
        )
    if not fully_matched:
        return Answer(
            text=("조건 일부를 인식하지 못했습니다. 인식한 조건: "
                  + ", ".join(f"{c.field}{c.operator}{c.value}" for c in conditions)
                  + " — 나머지 조건을 더 구체적으로 말씀해주시겠어요?"),
            task_type="select", abstained=True,
            condition_query=[c.__dict__ for c in conditions], route=ROUTE_CLARIFY,
        )

    results, total, condition_warnings = run_conditions_query(table, conditions)

    doc_ids = [r.document_id for r in results]
    kept_ids, deadline_notes, disclosure = apply_deadline_filter(doc_ids, identity, cfg)
    kept_set = set(kept_ids)
    filtered_out = len(results) - len(kept_set)
    results = [r for r in results if r.document_id in kept_set]

    lines = [f"- {r.document_id}: {r.value}" for r in results]
    body = "\n".join(lines) if lines else "조건에 맞는 문서가 없습니다."
    if identity is None:
        body += "\n\n[알림] 마감 필터가 연결돼 있지 않습니다 — 마감 지난 사업이 섞여 있을 수 있습니다."
    elif filtered_out:
        body += f"\n\n(마감 지난 사업 {filtered_out}건 제외됨 — 기준일 {cfg['reference_datetime']})"
    if deadline_notes:
        head = f"[마감일 미상 — {disclosure}]" if disclosure else "[마감일 미상]"
        body += f"\n\n{head}\n" + "\n".join(deadline_notes)
    if condition_warnings:
        body += "\n\n[확인 필요]\n" + "\n".join(f"- {w}" for w in condition_warnings)
    if total > len(results) + filtered_out:
        body += f"\n\n(조건 매칭 전체 {total}건 중 일부만 표시)"

    # 선별형 근거 — 실제 선택 조건별로, 문서마다 근거를 남긴다(첫 조건만 X)
    citations: list[dict] = []
    for r in results:
        for cq in conditions:
            ev = build_field_evidence(table, r.document_id, cq.field, cfg, locator=locator)
            citations.extend(ev.citations)

    return Answer(
        text=body, task_type="select",
        sources=[f"{r.document_id} (추출표: {'/'.join(c.field for c in conditions)})"
                 for r in results],
        abstained=False,
        condition_query=[c.__dict__ for c in conditions],
        condition_result_doc_ids=[r.document_id for r in results],
        route=ROUTE_SELECT,
        structured_answer=[r.document_id for r in results],
        selected_document_ids=[r.document_id for r in results],
        citations=citations,
        used_sources=[_source_tag(cfg, "extraction_table")]
        + ([_source_tag(cfg, "identity_v2")] if identity is not None else []),
    )


# ---------------------------------------------------------------------------
# 되묻기
# ---------------------------------------------------------------------------

def _clarify(
    text: str, task_type: str, resolution: DocumentResolution | None = None,
    condition_query: list[dict] | None = None,
) -> Answer:
    return Answer(
        text=text, task_type=task_type, abstained=True, route=ROUTE_CLARIFY,
        condition_query=condition_query,
        resolution=({"method": resolution.method,
                     "candidates": resolution.candidates,
                     "unknown_orgs": resolution.unknown_orgs} if resolution else None),
        selected_document_ids=(resolution.candidates if resolution else []),
        # 후보 정보는 구조화 결과에도 보존한다(결함 1-2)
        structured_answer=({"clarification_needed": True,
                            "candidates": resolution.candidates,
                            "candidate_labels": resolution.candidate_labels,
                            "unknown_orgs": resolution.unknown_orgs,
                            "resolution_method": resolution.method}
                           if resolution else None),
    )


def _ambiguous_text(what: str, resolution: DocumentResolution) -> str:
    labels = resolution.candidate_labels or resolution.candidates
    return (f"{what}에 해당하는 문서가 여러 건입니다. 어느 문서인지 알려주시겠어요?\n"
            + "\n".join(f"- {lab}" for lab in labels))


# ---------------------------------------------------------------------------
# 추출형
# ---------------------------------------------------------------------------

def answer_extract_by_table(
    question: str, table: list[dict], store: VectorStore,
    get_embed_client: "Callable[[], EmbeddingClient]",
    get_gen_client: "Callable[[], GenerationClient]",
    cfg: dict[str, Any],
    session: SessionState | None = None,
    identity: IdentityIndex | None = None,
    locator: ChunkLocator | None = None,
) -> Answer:
    """12필드 추출형. 마감일은 12필드 밖이라 identity_v2 전용 분기로 처리."""
    active = session.active_document_id if session else None

    if detect_deadline_question(question):
        resolution = resolve_document(question, identity, active_document_id=active)
        if resolution.document_id is None:
            if resolution.candidates:
                return _clarify(_ambiguous_text("마감일을 물으신 조건", resolution),
                                "extract", resolution)
            return _clarify(
                "마감일을 어느 문서에서 확인할까요? 문서 ID(예: RFP-000001), "
                "발주기관명 또는 사업명을 알려주시면 찾아드릴게요.",
                "extract", resolution)
        doc_id = resolution.document_id
        if session is not None:
            session.active_document_id = doc_id
        if identity is None:
            return _clarify(
                "마감일 자료(identity_v2)가 이 실행에 연결돼 있지 않습니다 — "
                "--identity를 함께 넘겨야 확인할 수 있습니다.", "extract", resolution)
        ev = build_deadline_evidence(doc_id, identity, cfg, eligibility=False)
        return Answer(
            text=ev.answer_text, task_type="extract", abstained=ev.abstained,
            sources=[f"{doc_id} (identity_v2: {DEADLINE_COLUMN})"],
            condition_query=[{"field": "입찰 참여 마감일", "document_id": doc_id,
                              "route": "identity_v2", "resolved_by": resolution.method}],
            condition_result_doc_ids=[doc_id],
            route=ROUTE_IDENTITY_VALUE,
            structured_answer=ev.structured,
            citations=ev.citations,
            selected_document_ids=[doc_id],
            used_sources=[ev.used_source],
            resolution={"method": resolution.method, "candidates": resolution.candidates},
        )

    field_name = detect_field(question)
    if field_name is None:
        return _clarify(
            "어느 항목을 확인하고 싶으신가요? (예: 예산, 지역제한, 사업기간, 참가자격 등)",
            "extract")

    resolution = resolve_document(question, identity, active_document_id=active)
    if resolution.document_id is None:
        if resolution.candidates:
            return _clarify(_ambiguous_text(f"'{field_name}'를 물으신 조건", resolution),
                            "extract", resolution,
                            [{"field": field_name, "document_id": None,
                              "candidates": resolution.candidates}])
        return _clarify(
            f"'{field_name}'를 어느 문서에서 확인할까요? 문서 ID(예: RFP-000001), "
            f"발주기관명 또는 사업명을 알려주시면 찾아드릴게요.",
            "extract", resolution, [{"field": field_name, "document_id": None}])

    doc_id = resolution.document_id
    if session is not None:
        session.active_document_id = doc_id

    ev = build_field_evidence(table, doc_id, field_name, cfg, locator=locator)

    if needs_explanation(question):
        embed_client = get_embed_client()
        gen_client = get_gen_client()
        result = answer_qa_or_extract_by_search(
            question, store, embed_client, gen_client, cfg,
            document_id=doc_id, structured=[ev], task_type="extract",
        )
        result.condition_query = [{"field": field_name, "document_id": doc_id,
                                   "route": "G-2 + G→I→J→K",
                                   "resolved_by": resolution.method}]
        result.condition_result_doc_ids = [doc_id]
        return result

    return Answer(
        text=ev.answer_text, task_type="extract", abstained=ev.abstained,
        sources=[f"{doc_id} (추출표 {cfg.get('extraction_version')}: {field_name})"],
        condition_query=[{"field": field_name, "document_id": doc_id,
                          "status": ev.status, "route": "G-2",
                          "resolved_by": resolution.method}],
        condition_result_doc_ids=[doc_id],
        route=ROUTE_EXTRACT_VALUE,
        structured_answer=_merge_structured([ev]),
        citations=ev.citations,
        selected_document_ids=[doc_id],
        used_sources=[ev.used_source],
        resolution={"method": resolution.method, "candidates": resolution.candidates},
    )


# ---------------------------------------------------------------------------
# 비교형
# ---------------------------------------------------------------------------

def answer_compare_by_table(
    question: str, table: list[dict], cfg: dict[str, Any],
    identity: IdentityIndex | None = None,
    locator: ChunkLocator | None = None,
) -> Answer:
    """비교형 — 필드×문서 구조로 코드가 조립. LLM이 값을 다시 쓰지 않는다."""
    fields = detect_fields(question)
    doc_ids, unknown_orgs = resolve_documents_for_compare(question, identity)

    if not fields:
        return _clarify(
            "어느 항목을 비교할까요? (예: 예산, 지역제한, 사업기간, 참가자격 등)",
            "compare")
    if len(doc_ids) < 2:
        extra = ""
        if unknown_orgs:
            extra = ("\n확인되지 않은 기관: " + ", ".join(unknown_orgs)
                     + " — 공식 목록에 없는 기관입니다.")
        return _clarify(
            "비교하려면 문서가 두 건 이상 필요합니다. 문서 ID 또는 발주기관·사업명을 "
            "두 개 이상 알려주시겠어요? (예: RFP-000001, RFP-000002)" + extra,
            "compare", condition_query=[{"fields": fields, "document_ids": doc_ids}])

    body_parts: list[str] = []
    # 응답 계약: 비교형 structured_answer 는 {문서ID: {필드: 값 문자열}}
    structured: dict[str, dict[str, Any]] = {d: {} for d in doc_ids}
    evidence_detail: dict[str, dict[str, Any]] = {d: {} for d in doc_ids}
    citations: list[dict] = []
    abstained = False
    for field_name in fields:
        lines = []
        for doc_id in doc_ids:
            ev = build_field_evidence(table, doc_id, field_name, cfg, locator=locator)
            lines.append(f"| {doc_id} | {ev.short_text or ev.answer_text} |")
            structured[doc_id][field_name] = ev.answer_text
            evidence_detail[doc_id][field_name] = ev.structured
            # 실제로 비교한 문서×필드의 근거만 기록(첫 문서·첫 위치만 남기지 않는다)
            citations.extend(ev.citations)
            abstained = abstained or ev.abstained
        body_parts.append(f"[{field_name}] 비교\n| 문서 | 값 |\n|---|---|\n"
                          + "\n".join(lines))

    body = "\n\n".join(body_parts)
    if unknown_orgs:
        body += ("\n\n[확인 필요] 공식 목록에 없는 기관: " + ", ".join(unknown_orgs))

    return Answer(
        text=body, task_type="compare",
        sources=[f"{doc_id} (추출표 {cfg.get('extraction_version')}: {f})"
                 for f in fields for doc_id in doc_ids],
        abstained=abstained,
        condition_query=[{"fields": fields, "document_ids": doc_ids, "route": "G-2"}],
        condition_result_doc_ids=doc_ids,
        route=ROUTE_COMPARE,
        structured_answer=structured,
        citation_diagnostics={"protocol": "structured_only",
                              "evidence_detail": evidence_detail},
        selected_document_ids=doc_ids,
        citations=citations,
        used_sources=[_source_tag(cfg, "extraction_table")],
    )


# ---------------------------------------------------------------------------
# QA형 (구조화 자료 결합)
# ---------------------------------------------------------------------------

def answer_qa(
    question: str, store: VectorStore, table: list[dict],
    get_embed_client: "Callable[[], EmbeddingClient]",
    get_gen_client: "Callable[[], GenerationClient]",
    cfg: dict[str, Any],
    session: SessionState | None = None,
    identity: IdentityIndex | None = None,
    locator: ChunkLocator | None = None,
) -> Answer:
    active = session.active_document_id if session else None
    resolution = resolve_document(question, identity, active_document_id=active)

    # 존재하지 않는 기관을 물으면 검색·생성을 태우지 않는다(비용·환각 방지)
    if resolution.unknown_orgs and resolution.document_id is None and not resolution.candidates:
        return Answer(
            text=("'" + ", ".join(resolution.unknown_orgs) + "'에 해당하는 사업을 "
                  "찾을 수 없습니다. 사업명이나 발주기관명을 다시 확인해주세요."),
            task_type="qa", abstained=True, route=ROUTE_CLARIFY,
            resolution={"method": resolution.method,
                        "unknown_orgs": resolution.unknown_orgs},
        )

    # ⭐ 결함 1-2: 문서 후보가 2개 이상인데 하나로 확정되지 않았으면, 임의로 첫 문서를
    #    고르거나 코퍼스 전체 검색으로 넘어가지 않는다. 검색·LLM 호출 **전에** 되묻는다.
    if resolution.document_id is None and len(resolution.candidates) > 1:
        return _clarify(_ambiguous_text("물으신 조건", resolution), "qa", resolution)

    doc_id = resolution.document_id
    if doc_id and session is not None:
        session.active_document_id = doc_id

    # 확정된 구조화 필드가 질문에 있으면 구조화 자료를 함께 쓴다(3-2 확정)
    structured: list[StructuredEvidence] = []
    if doc_id:
        if needs_deadline_data(question) and identity is not None:
            structured.append(build_deadline_evidence(
                doc_id, identity, cfg,
                eligibility=detect_deadline_eligibility_question(question),
            ))
        field_name = detect_field(question)
        if field_name and table:
            structured.append(build_field_evidence(table, doc_id, field_name, cfg,
                                                   locator=locator))

    embed_client = get_embed_client()
    gen_client = get_gen_client()
    result = answer_qa_or_extract_by_search(
        question, store, embed_client, gen_client, cfg,
        document_id=doc_id, structured=structured, task_type="qa",
    )
    if resolution.unknown_orgs:
        result.text += ("\n\n[확인 필요] 공식 목록에 없는 기관: "
                        + ", ".join(resolution.unknown_orgs))
    result.resolution = {"method": resolution.method,
                         "candidates": resolution.candidates,
                         "unknown_orgs": resolution.unknown_orgs}
    return result


# ---------------------------------------------------------------------------
# 오케스트레이션
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{6}[A-Za-z0-9_\-]+"), "sk-******(마스킹됨)"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "(이메일 마스킹됨)"),
    (re.compile(r"\b0\d{1,2}-?\d{3,4}-?\d{4}\b"), "(전화번호 마스킹됨)"),
]


def sanitize_error(detail: str | None) -> str | None:
    """오류 문구에서 API 키·개인정보 후보를 제거한다."""
    if not detail:
        return detail
    out = str(detail)
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def answer(
    question: str, store: VectorStore,
    get_embed_client: "Callable[[], EmbeddingClient]",
    get_gen_client: "Callable[[], GenerationClient]",
    table: list[dict], cfg: dict[str, Any],
    identity: IdentityIndex | None = None,
    session: SessionState | None = None,
    locator: ChunkLocator | None = None,
) -> Answer:
    r: RouteResult = route(question, cfg)

    try:
        if r.task_type == "no_search_needed":
            if r.no_search_kind == NO_SEARCH_SYSTEM_HELP:
                result = Answer(text=SYSTEM_HELP_TEXT, task_type=r.task_type,
                                route=ROUTE_SYSTEM_HELP)
            else:
                result = Answer(
                    text=("안녕하세요! RFP 관련 질문을 도와드릴게요. "
                          "무엇을 물어볼 수 있는지 궁금하시면 \"사용법 알려줘\"라고 "
                          "말씀해 주세요."),
                    task_type=r.task_type, route=ROUTE_GREETING)
        elif r.task_type == "select":
            result = answer_select_by_table(question, table, cfg, identity=identity,
                                            locator=locator)
        elif r.task_type == "extract":
            result = answer_extract_by_table(
                question, table, store, get_embed_client, get_gen_client, cfg,
                session=session, identity=identity, locator=locator)
        elif r.task_type == "compare":
            result = answer_compare_by_table(question, table, cfg, identity=identity,
                                             locator=locator)
        elif r.task_type == "qa":
            result = answer_qa(question, store, table, get_embed_client,
                               get_gen_client, cfg, session=session, identity=identity,
                               locator=locator)
        else:
            result = Answer(text="확인할 수 없습니다.", task_type=r.task_type,
                            abstained=True, route=ROUTE_CLARIFY)
    except Exception as e:  # noqa: BLE001
        detail = sanitize_error(f"{type(e).__name__}: {e}")
        result = Answer(
            text="처리 중 오류가 발생했습니다.", task_type=r.task_type,
            abstained=True, error_stage=r.task_type,
            error_detail=detail, failure=detail,
        )

    result.task_type = r.task_type
    result.route_matched_rule = r.matched_rule
    result.route_is_fallback = r.is_fallback
    return result


def answer_to_response(question_id: str, result: Answer) -> dict:
    """하루님 채점기 responses.jsonl 한 줄 — 질문 유형과 무관하게 동일 구조."""
    return {
        "id": question_id,
        "answer": result.text,
        "structured_answer": result.structured_answer,
        "contexts": result.contexts,
        "retrieved": result.retrieved,
        "citations": result.citations,
        "selected_document_ids": result.selected_document_ids,
        "abstained": result.abstained,
        "route": result.route,
        "failure": sanitize_error(result.failure),
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _try_derive(explicit: str | None, deriver, cfg, label: str) -> Path | None:
    if explicit:
        return Path(explicit)
    try:
        return deriver(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  {label} 자동 조립 실패({e}) — 이 값 없이 진행합니다.", file=sys.stderr)
        return None


def build_runtime(args, cfg) -> dict[str, Any]:
    """CLI 진입점 두 곳(answer_pipeline / run_eval)이 공유하는 준비 절차."""
    index_path = _try_derive(args.index, index_dir, cfg, "--index")
    if index_path is None:
        raise RuntimeError(
            "--index를 자동 조립하지 못했고 명시적으로도 안 주셨습니다 — "
            "인덱스 없이는 실행할 수 없습니다. --index를 직접 주거나 RAG_ROOT를 설정하세요.")
    table_arg = _try_derive(args.extraction_table, extraction_table_path, cfg,
                            "--extraction-table")
    meta_arg = _try_derive(getattr(args, "extraction_metadata", None),
                           extraction_metadata_path, cfg, "--extraction-metadata")
    identity_arg = _try_derive(args.identity, identity_path, cfg, "--identity")
    registry_arg = _try_derive(args.registry, document_registry_path, cfg, "--registry")
    chunks_arg = _try_derive(getattr(args, "chunks", None), chunks_path, cfg, "--chunks")

    # ⭐ 결함 1-1: 인덱스를 **불러오기 전에** 꼬리표를 검증한다. 여기서 실패하면
    #    임베딩·생성 클라이언트를 만들지 않으므로 API 호출이 0회다.
    #    build_index.py 와 같은 함수(validate_index_tag)를 써서 기준을 이중화하지 않는다.
    index_check = validate_index_tag(index_path, cfg)

    store = VectorStore.load(index_path)

    if not (table_arg and Path(table_arg).exists()):
        raise RuntimeError(
            f"공식 추출표를 찾지 못했습니다({table_arg}). 선별형·추출형·비교형이 "
            f"동작하지 않으므로 중단합니다.")
    table = load_extraction_table(
        Path(table_arg), cfg=cfg, metadata_path=meta_arg,
        official=not getattr(args, "allow_unofficial_table", False),
    )

    identity = None
    if identity_arg and Path(identity_arg).exists():
        identity = load_identity(identity_arg)
    elif not getattr(args, "allow_no_identity", False):
        raise RuntimeError(
            f"identity_v2를 찾지 못했습니다({identity_arg}). 마감일·발주기관·사업명의 "
            f"유일한 공식 출처라 없으면 문서 특정과 마감일 답변이 불가능합니다. "
            f"테스트 목적이면 --allow-no-identity를 명시하세요.")

    deadline_filter_on = cfg.get("deadline_filter_default", {}).get("select", False)
    if identity is None and deadline_filter_on and not args.allow_no_deadline_filter:
        raise RuntimeError(
            "base.yaml의 deadline_filter_default.select=true인데 identity_v2가 "
            "없습니다. 마감 지난 사업이 결과에 섞일 수 있어 중단합니다. "
            "--allow-no-deadline-filter로만 우회하세요.")

    # 근거 좌표 변환용 청크 색인 (팀 확정 좌표 규약: ref_no=location_label, line 범위).
    # 검색 제외 청크도 좌표는 존재하므로 chunks.jsonl 전체를 쓴다.
    locator = None
    locator_source = None
    if chunks_arg and Path(chunks_arg).exists():
        locator = ChunkLocator.from_chunks_jsonl(chunks_arg)
        locator_source = str(chunks_arg)
    elif not getattr(args, "allow_no_chunks", False):
        raise RuntimeError(
            f"청크 파일을 찾지 못했습니다({chunks_arg}). 근거 좌표를 팀 확정 규약"
            f"(location_label·line)으로 변환할 수 없어 중단합니다. --chunks 로 지정하거나, "
            f"테스트 목적이면 --allow-no-chunks 를 명시하세요.")
    else:
        locator = ChunkLocator.from_vector_store(store)
        locator_source = f"{index_path} (인덱스 메타데이터 — 검색 제외 청크 없음)"

    return {
        "store": store, "table": table, "identity": identity, "locator": locator,
        "index_check": {"index_dir": index_check["index_dir"],
                        "vector_dimension": index_check["vector_dimension"],
                        "tag": index_check["tag"],
                        "validated": True},
        "paths": {
            "index": str(index_path), "extraction_table": str(table_arg),
            "extraction_metadata": str(meta_arg), "identity": str(identity_arg),
            "registry": str(registry_arg), "chunks": locator_source,
        },
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", required=False)
    parser.add_argument("--extraction-table", required=False)
    parser.add_argument("--extraction-metadata", required=False)
    parser.add_argument("--identity", required=False,
                        help="document_identity_v2.csv 경로(마감일·기관명·사업명)")
    parser.add_argument("--registry", required=False)
    parser.add_argument("--chunks", required=False,
                        help="chunks.jsonl 경로 — 근거 좌표를 팀 확정 규약으로 변환하는 데 씀. "
                             "생략하면 base.yaml의 chunking_version 으로 자동 조립")
    parser.add_argument("--allow-no-chunks", action="store_true",
                        help="청크 파일 없이 실행(테스트 목적). 좌표 변환 정확도가 떨어진다.")
    parser.add_argument("--experiment-config", required=False)
    parser.add_argument("--allow-no-deadline-filter", action="store_true")
    parser.add_argument("--allow-no-identity", action="store_true",
                        help="identity_v2 없이 실행(테스트 목적).")
    parser.add_argument("--allow-unofficial-table", action="store_true",
                        help="공식 규모(1,200행·100문서·12필드)·버전 검증을 건너뛴다(테스트 목적).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--active-document", required=False)
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.experiment_config)
    rt = build_runtime(args, cfg)
    session = SessionState(active_document_id=args.active_document)

    cache: dict[str, Any] = {}

    def get_embed_client() -> EmbeddingClient:
        if "embed" not in cache:
            cache["embed"] = EmbeddingClient(cfg)
        cache["embed"].reset_usage()
        return cache["embed"]

    def get_gen_client() -> GenerationClient:
        if "gen" not in cache:
            cache["gen"] = GenerationClient(cfg)
        cache["gen"].reset_usage()
        return cache["gen"]

    import time
    from pricing import compute_cost
    t0 = time.perf_counter()
    result = answer(args.question, rt["store"], get_embed_client, get_gen_client,
                    rt["table"], cfg, identity=rt["identity"], session=session,
                    locator=rt["locator"])
    result.latency_ms = round((time.perf_counter() - t0) * 1000)

    usage = Usage()
    for key in ("gen", "embed"):
        client = cache.get(key)
        if client is not None:
            u = client.usage
            usage.generation_requests += u.generation_requests
            usage.generation_input_tokens += u.generation_input_tokens
            usage.generation_cached_input_tokens += u.generation_cached_input_tokens
            usage.generation_output_tokens += u.generation_output_tokens
            usage.embedding_requests += u.embedding_requests
            usage.embedding_tokens += u.embedding_tokens
    result.cost_usd, result.cost_detail = compute_cost(cfg, usage)

    payload = answer_to_response(args.question, result)
    payload.update({
        "question": args.question,
        "task_type": result.task_type,
        "route_matched_rule": result.route_matched_rule,
        "route_is_fallback": result.route_is_fallback,
        "sources": result.sources,
        "used_sources": result.used_sources,
        "resolution": result.resolution,
        "active_document_after": session.active_document_id,
        "error_stage": result.error_stage,
        "error_detail": sanitize_error(result.error_detail),
        "cost_detail": result.cost_detail,
        "citation_diagnostics": result.citation_diagnostics,
        "index_check": rt["index_check"],
    })
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if result.error_stage:
        sys.exit(1)


if __name__ == "__main__":
    main()
