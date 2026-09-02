"""
G-2 — 조건 질의 및 공식 추출표(rfp_extraction_table_v3) 조회.

공식 추출 테이블 형식 (extraction_table_v3.json, JSONL 아님. 최상위 'rows' 키):
{
  "document_id": "RFP-000001",
  "field_name": <공식 12필드 중 하나>,
  "status": "value_present" | "field_absent" | "external_reference"
            | "not_disclosed" | "conflict" | "extraction_failed" | "review_required",
  "answer_raw": "...", "answer_normalized": "...",
  "representative_location": {...}, "additional_locations": [ {...}, ... ],
  "active": "true" | "false"  (문자열!),
  "schema_version", "extraction_version", "corpus_version", "registry_version", ...
}
⚠️ 사업명·발주기관·마감일시는 최종 12필드에서 제거됨 — identity_v2 담당.

[2026-09-02 수정]
① 숫자 안의 쉼표 버그: "49,500"이 "49"와 "500"으로 쪼개지던 문제. 조건을
   나누기 **전에** 천 단위 쉼표 숫자를 자리표시자로 보호한다.
② 비교 연산자 전체 지원: 이상(>=)·초과/넘는(>)·이하(<=)·미만(<)이 서로 다른
   연산자로 동작하고, 네 연산자 모두 실제 금액 값까지 파싱한다(예전엔
   >=,<= 에서만 값을 채우고 >,< 는 value=None이라 항상 거짓이었다).
③ 지역제한을 뜻으로 판정: 빈칸(field_absent)을 '제한 없음'으로 보지 않고,
   value_present라는 이유만으로 '제한 있음'으로 보지도 않는다.
④ 공식 추출표 검증 강화: 1,200행·100문서·문서당 12필드·필드명 일치·
   (document_id, field_name) 중복 0·스키마/버전 일치·메타데이터 대조.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

State = Literal[
    "value_present", "field_absent", "external_reference", "not_disclosed",
    "conflict",
]
# 공식표에 현재 0건이지만 스키마상 허용되는 상태
_FUTURE_ALLOWED_STATUS = {"extraction_failed", "review_required"}
ALL_ALLOWED_STATUS = set(State.__args__) | _FUTURE_ALLOWED_STATUS

# 정상 값처럼 답하면 안 되는 상태 (4-9-8 / 3-2 확정)
NON_VALUE_STATUS = {
    "field_absent", "external_reference", "not_disclosed", "conflict",
    "extraction_failed", "review_required",
}

# 공식 12필드 — extraction_table_v3.json의 "fields"와 정확히 같아야 한다
OFFICIAL_FIELDS: tuple[str, ...] = (
    "사업 개요", "사업분야", "공고일", "사업기간", "예산",
    "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건",
    "평가 배점", "제출 방식", "필수 제출 서류", "과업 범위",
)

_ALLOWED_FIELDS = {"예산", "지역제한"}
_ALLOWED_OPERATORS = {
    ">=", "<=", ">", "<", "==",
    "no_restriction", "has_restriction",
}

FIELD_KEYWORDS: dict[str, list[str]] = {
    "사업 개요": ["사업개요", "사업 개요", "사업내용", "사업 내용"],
    "사업분야": ["사업분야", "사업 분야"],
    "공고일": ["공고일"],
    "사업기간": ["사업기간", "사업 기간", "계약기간", "수행기간"],
    "예산": ["예산", "사업금액", "계약금액", "사업 금액", "소요예산", "사업비"],
    "참가 자격(면허·실적)": ["참가자격", "참가 자격", "자격요건", "자격 요건", "면허", "실적"],
    "지역제한": ["지역제한", "지역 제한", "지역참가", "참가 지역"],
    "컨소시엄 요건": ["컨소시엄", "공동수급"],
    "평가 배점": ["평가배점", "평가 배점", "배점"],
    "제출 방식": ["제출방식", "제출 방식"],
    "필수 제출 서류": ["제출서류", "제출 서류", "필수서류", "필수 서류", "서류"],
    "과업 범위": ["과업범위", "과업 범위"],
}

_DOC_ID_RE = re.compile(r"RFP-\d{6}")

_NEEDS_EXPLANATION_KEYWORDS = ["설명해", "왜", "자세히", "근거", "이유"]

# 마감일은 12필드 밖 — identity_v2 전용 경로로 간다
_DEADLINE_KEYWORDS = [
    "마감일", "마감 일", "마감기한", "마감 기한", "입찰 마감", "제출 마감",
    "언제까지", "마감시한", "접수마감", "접수 마감",
]


def detect_field(question: str) -> str | None:
    for field, keywords in FIELD_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            return field
    return None


def detect_fields(question: str) -> list[str]:
    return [field for field, keywords in FIELD_KEYWORDS.items()
            if any(kw in question for kw in keywords)]


# 마감일 값을 직접 묻지는 않지만 마감일 자료가 있어야 답할 수 있는 질문
# ("지금 입찰 참여할 수 있어?"). ⚠️ 라우터의 extract 키워드 목록에는 넣지
# 않는다 — 평가셋이 qa로 라벨한 문항을 강제로 추출형으로 바꾸지 않기 위해서다.
_DEADLINE_ELIGIBILITY_KEYWORDS = [
    "입찰 참여", "입찰참여", "참여할 수 있", "참여 가능", "참여가능",
    "입찰할 수 있", "입찰 가능", "응찰", "참가할 수 있", "참가 가능",
    "지원할 수 있", "신청할 수 있", "아직 지원", "아직 신청", "아직 참여",
]


def detect_deadline_question(question: str) -> bool:
    """마감일 '값'을 직접 묻는 질문."""
    return any(kw in question for kw in _DEADLINE_KEYWORDS)


def detect_deadline_eligibility_question(question: str) -> bool:
    """마감일을 근거로 판단해야 하는 질문(참여 가능 여부 등)."""
    return any(kw in question for kw in _DEADLINE_ELIGIBILITY_KEYWORDS)


def needs_deadline_data(question: str) -> bool:
    """identity_v2의 마감일 값이 필요한 질문인지."""
    return detect_deadline_question(question) or detect_deadline_eligibility_question(question)


def detect_document_id(question: str) -> str | None:
    m = _DOC_ID_RE.search(question)
    return m.group(0) if m else None


def detect_document_ids(question: str) -> list[str]:
    seen: list[str] = []
    for m in _DOC_ID_RE.finditer(question):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def needs_explanation(question: str) -> bool:
    return any(kw in question for kw in _NEEDS_EXPLANATION_KEYWORDS)


# ---------------------------------------------------------------------------
# 조건 파서 — 숫자 안의 쉼표 보호 후 조건 분리
# ---------------------------------------------------------------------------

# 천 단위 쉼표가 들어간 숫자(소수점 포함). 조건 분리 전에 통째로 보호한다.
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_MASK_OPEN = "\x00"
_MASK_CLOSE = "\x01"

# 조건과 조건 사이를 잇는 연결어. 쉼표는 숫자를 보호한 **뒤에만** 구분자로 쓴다.
_CONNECTOR_RE = re.compile(r"(?:이고|이면서|이며|그리고|,\s*|、|및\s|｜|\|)")

_AMOUNT_UNIT_MULTIPLIER = {
    None: 1, "": 1,
    "조": 10**12, "억": 10**8,
    "천만": 10**7, "백만": 10**6, "십만": 10**5, "만": 10**4,
    "천": 10**3, "백": 10**2, "십": 10**1,
}
# 긴 복합 단위를 먼저 — "5천만"이 "5천"+"만"으로 쪼개지지 않게 순서가 중요
_UNIT_ALT = "조|억|천만|백만|십만|만|천|백|십"
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
# ⚠️ 2026-09-02: "1억 5천만원 이상"처럼 단위가 여러 번 붙는 복합 금액을 한 덩어리로
#    잡는다. 예전 패턴은 숫자 하나+단위 하나만 봐서 "1억 5천만원 이상"에서 "5천만"만
#    읽고 5천만으로 비교했다(실제 재현). 잡은 덩어리는 _parse_korean_amount로 환산해
#    표 값 파싱과 같은 규칙을 쓴다.
_AMOUNT_EXPR = rf"((?:\d[\d,]*(?:\.\d+)?\s*(?:{_UNIT_ALT})?\s*)+)"


def mask_grouped_numbers(text: str) -> tuple[str, list[str]]:
    """천 단위 쉼표 숫자를 자리표시자로 바꾼다. (마스킹된 문장, 원본 숫자 목록)

    >>> mask_grouped_numbers("예산 49,500만원 이상, 지역제한 없음")[1]
    ['49,500']
    """
    values: list[str] = []

    def _repl(m: re.Match) -> str:
        values.append(m.group(0))
        return f"{_MASK_OPEN}{len(values) - 1}{_MASK_CLOSE}"

    return _GROUPED_NUMBER_RE.sub(_repl, text), values


def unmask_grouped_numbers(text: str, values: list[str]) -> str:
    def _repl(m: re.Match) -> str:
        return values[int(m.group(1))]

    return re.sub(f"{_MASK_OPEN}(\\d+){_MASK_CLOSE}", _repl, text)


def split_conditions(question: str) -> list[str]:
    """조건 구분자로만 문장을 나눈다. 숫자 안 쉼표는 절대 구분자가 아니다.

    >>> split_conditions("예산 49,500만원 이상, 지역제한 없음")
    ['예산 49,500만원 이상', '지역제한 없음']
    """
    masked, values = mask_grouped_numbers(question)
    segments = [s.strip() for s in _CONNECTOR_RE.split(masked) if s.strip()]
    if not segments:
        segments = [masked]
    return [unmask_grouped_numbers(s, values) for s in segments]


# (정규식, 필드, 연산자) — 라우터가 인식하는 표현과 1:1로 맞춘다
_CONDITION_PATTERNS: list[tuple[str, str, str]] = [
    (rf"{_AMOUNT_EXPR}원?\s*이상", "예산", ">="),
    (rf"{_AMOUNT_EXPR}원?\s*이하", "예산", "<="),
    (rf"{_AMOUNT_EXPR}원?\s*이내", "예산", "<="),
    (rf"{_AMOUNT_EXPR}원?\s*초과", "예산", ">"),
    (rf"{_AMOUNT_EXPR}원?\s*미만", "예산", "<"),
    (rf"{_AMOUNT_EXPR}원?\s*(?:을|를)?\s*(?:넘는|넘은|넘어가는|넘어서는|넘게)",
     "예산", ">"),
    (rf"{_AMOUNT_EXPR}원?\s*(?:보다)?\s*(?:작은|적은|안\s*되는|안되는)",
     "예산", "<"),
    (r"지역\s*제한\s*(?:이|은|가)?\s*(?:없는|없음|없어|없다|없이|안\s*하는)", "지역제한", "no_restriction"),
    (r"지역\s*제한\s*(?:이|은|가)?\s*(?:있는|있음|있어|있다|걸린|걸려)", "지역제한", "has_restriction"),
    (r"전국\s*(?:에서|어디서나)?\s*(?:참여|응찰|입찰)\s*(?:가능|되는)", "지역제한", "no_restriction"),
]


@dataclass
class ConditionQuery:
    field: str
    operator: str
    value: Any = None
    raw_text: str = ""


class InvalidQueryError(ValueError):
    pass


def _amount_from_match(m: re.Match) -> float | None:
    """조건 표현에서 잡은 금액 덩어리를 실제 금액으로 환산.
    표 값 파싱(_parse_korean_amount)과 같은 규칙을 써서 두 경로가 어긋나지 않게 한다.
    모든 비교 연산자(>=, <=, >, <)에 똑같이 적용한다."""
    groups = m.groups()
    if not groups:
        return None
    expr = (groups[0] or "").strip()
    if not expr:
        return None
    return _parse_korean_amount(expr)


def _match_one(segment: str) -> ConditionQuery | None:
    for pattern, field, op in _CONDITION_PATTERNS:
        m = re.search(pattern, segment)
        if m:
            value = None
            if op in (">=", "<=", ">", "<", "=="):
                value = _amount_from_match(m)
                if value is None:
                    continue  # 숫자를 못 읽었으면 이 조건으로 확정하지 않는다
            return ConditionQuery(field=field, operator=op, value=value,
                                  raw_text=segment.strip())
    return None


def parse_condition(question: str) -> ConditionQuery | None:
    conditions, _ = parse_conditions(question)
    return conditions[0] if conditions else None


# 남은 문구에 "조건 같아 보이는 표현"이 있는지 판별하는 신호 (결함 1-3).
# 이 중 하나라도 남아 있으면 조건을 다 읽지 못한 것으로 본다.
_LEFTOVER_COMPARATOR_RE = re.compile(
    r"(?:이상|이하|초과|미만|넘는|넘은|없는|있는|제한|이내|한정|한함)"
)
# "발주기관이 서울인", "담당자가 김철수인" 같은 필터 표현
_LEFTOVER_ATTRIBUTIVE_RE = re.compile(r"[가-힣]{2,}(?:이|가|은|는)\s*[가-힣0-9]{1,}\s*인(?:\s|$|[,.])")

_ALL_FIELD_KEYWORDS = [kw for kws in FIELD_KEYWORDS.values() for kw in kws]


def _find_all_matches(segment: str) -> list[tuple[int, int, ConditionQuery]]:
    """조각 안에서 조건 표현을 **전부** 찾는다(첫 매칭만 보지 않는다).

    ⚠️ 결함 1-3: 예전엔 조각마다 첫 매칭 하나만 읽어서, 쉼표·접속사가 없는
    "예산 5억 이상 지역제한 없는 사업"에서 예산 조건만 읽고 fully_matched=True 로
    성공 처리했다. 이제 겹치지 않는 매칭을 모두 모은다.
    """
    found: list[tuple[int, int, ConditionQuery]] = []
    for pattern, field, op in _CONDITION_PATTERNS:
        for m in re.finditer(pattern, segment):
            value = None
            if op in (">=", "<=", ">", "<", "=="):
                value = _amount_from_match(m)
                if value is None:
                    continue
            found.append((m.start(), m.end(),
                          ConditionQuery(field=field, operator=op, value=value,
                                         raw_text=m.group(0).strip())))
    # 겹치는 매칭은 더 긴 쪽만 남긴다(앞선 것 우선)
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, ConditionQuery]] = []
    for start, end, cq in found:
        if any(start < k_end and end > k_start for k_start, k_end, _ in kept):
            continue
        kept.append((start, end, cq))
    return kept


def _leftover_has_unread_condition(segment: str,
                                   spans: list[tuple[int, int]],
                                   covered_fields: set[str]) -> bool:
    """매칭된 구간을 뺀 나머지에 아직 읽지 못한 조건 표현이 남아 있는가."""
    chars = list(segment)
    for start, end in spans:
        for i in range(start, min(end, len(chars))):
            chars[i] = " "
    leftover = "".join(chars)
    # 이미 조건으로 읽은 필드의 이름은 남아 있어도 정상("예산 5억 이상"의 "예산")
    for field in covered_fields:
        for kw in FIELD_KEYWORDS.get(field, []):
            leftover = leftover.replace(kw, " ")
    if _LEFTOVER_COMPARATOR_RE.search(leftover):
        return True
    if _LEFTOVER_ATTRIBUTIVE_RE.search(leftover):
        return True
    # 아직 조건으로 읽지 않은 12필드 키워드가 남아 있으면 못 읽은 조건
    uncovered = [kw for kw in _ALL_FIELD_KEYWORDS
                 if kw in leftover
                 and not any(kw in FIELD_KEYWORDS.get(f, []) for f in covered_fields)]
    return bool(uncovered)


def parse_conditions(question: str) -> tuple[list[ConditionQuery], bool]:
    """AND 복합조건 파서. (인식된 조건 목록, 전부 인식됐는지)

    두 번째 값이 False면 조건 같아 보이는 조각을 놓쳤다는 뜻 — 호출측은 조용히
    무시하지 말고 확인 질문을 하거나 보류해야 한다.

    두 층으로 본다(결함 1-3):
      ① 연결어(쉼표·이고·그리고…)로 나눈 조각마다 조건이 하나라도 잡히는가
      ② 구분자가 없어도 한 조각 안에서 서로 다른 필드 조건을 **전부** 찾는가
    ①은 "발주기관이 서울인" 같은 미지원 조건을, ②는 "예산 5억 이상 지역제한 없는"
    같은 무구분자 복합조건을 잡는다. 둘 중 하나라도 미해석이면 fully_matched=False.
    """
    masked, values = mask_grouped_numbers(question)
    segments_masked = [s for s in _CONNECTOR_RE.split(masked) if s.strip()]
    if not segments_masked:
        segments_masked = [masked]

    matched: list[ConditionQuery] = []
    unread = 0
    for seg_masked in segments_masked:
        seg = unmask_grouped_numbers(seg_masked, values)
        hits = _find_all_matches(seg)
        if not hits:
            unread += 1
            continue
        covered = {cq.field for _, _, cq in hits}
        if _leftover_has_unread_condition(seg, [(a, b) for a, b, _ in hits], covered):
            unread += 1
        matched.extend(cq for _, _, cq in hits)

    fully_matched = unread == 0 and len(matched) > 0
    return matched, fully_matched


def validate_condition(cq: ConditionQuery) -> None:
    if cq.field not in _ALLOWED_FIELDS:
        raise InvalidQueryError(f"허용 안 된 필드: {cq.field}")
    if cq.operator not in _ALLOWED_OPERATORS:
        raise InvalidQueryError(f"허용 안 된 연산자: {cq.operator}")


# ---------------------------------------------------------------------------
# 공식 추출표 로드·검증
# ---------------------------------------------------------------------------

class ExtractionTableFormatError(ValueError):
    pass


OFFICIAL_ROW_COUNT = 1200
OFFICIAL_DOCUMENT_COUNT = 100
OFFICIAL_FIELD_COUNT = 12


def _row_active(row: dict) -> bool:
    v = row.get("active", "true")
    return str(v).lower() != "false"


def validate_extraction_table(
    doc: dict,
    cfg: dict[str, Any] | None = None,
    metadata: dict | None = None,
    official: bool = False,
    source: str = "extraction_table",
) -> None:
    """추출표 정합성 검사.

    official=True면 공식 규모(1,200행 / 100문서 / 문서당 12필드 / 확정 필드명)와
    선언 버전(schema_version·extraction_version·corpus_version·registry_version)을
    강제한다. 파일명·폴더명이 v3인 것만으로는 인정하지 않는다.
    """
    if "rows" not in doc:
        raise ExtractionTableFormatError(
            f"{source}: 최상위 객체에 'rows' 키가 없습니다."
        )
    rows = doc["rows"]

    declared_rows = doc.get("row_count")
    declared_docs = doc.get("document_count")
    declared_fields = doc.get("field_count")
    declared_field_names = doc.get("fields")

    if declared_rows is not None and len(rows) != declared_rows:
        raise ExtractionTableFormatError(
            f"{source}: row_count 선언값({declared_rows})과 실제 행 수({len(rows)})가 다릅니다."
        )

    doc_ids = {r["document_id"] for r in rows}
    field_names = {r["field_name"] for r in rows}

    if declared_docs is not None and len(doc_ids) != declared_docs:
        raise ExtractionTableFormatError(
            f"{source}: document_count 선언값({declared_docs})과 실제 문서 수"
            f"({len(doc_ids)})가 다릅니다."
        )
    if declared_fields is not None and len(field_names) != declared_fields:
        raise ExtractionTableFormatError(
            f"{source}: field_count 선언값({declared_fields})과 실제 필드 종류"
            f"({len(field_names)})가 다릅니다."
        )
    if declared_field_names is not None and set(declared_field_names) != field_names:
        raise ExtractionTableFormatError(
            f"{source}: 선언된 fields와 실제 field_name 집합이 다릅니다. "
            f"선언에만 있음={sorted(set(declared_field_names) - field_names)}, "
            f"실제에만 있음={sorted(field_names - set(declared_field_names))}"
        )

    bad_status = {r.get("status") for r in rows} - ALL_ALLOWED_STATUS
    if bad_status:
        raise ExtractionTableFormatError(f"{source}: 허용 안 된 status 값: {bad_status}")

    # (문서, 필드) 조합 — 중복 0건 / 누락 0건 / 예상 밖 필드 0건
    active_rows = [r for r in rows if _row_active(r)]
    combo_counts = Counter((r["document_id"], r["field_name"]) for r in active_rows)
    dupes = [k for k, v in combo_counts.items() if v > 1]
    if dupes:
        raise ExtractionTableFormatError(
            f"{source}: 같은 문서·같은 필드에 활성 행이 2개 이상: {dupes[:5]}"
            + (f" 외 {len(dupes) - 5}건" if len(dupes) > 5 else "")
        )

    expected_fields = set(OFFICIAL_FIELDS) if official else field_names
    if official:
        unexpected = field_names - expected_fields
        missing_fields = expected_fields - field_names
        if unexpected or missing_fields:
            raise ExtractionTableFormatError(
                f"{source}: 공식 12필드와 필드명이 다릅니다. "
                f"예상 밖={sorted(unexpected)}, 빠짐={sorted(missing_fields)}"
            )

    for doc_id in sorted(doc_ids):
        present = {f for (d, f) in combo_counts if d == doc_id}
        missing = expected_fields - present
        if missing:
            raise ExtractionTableFormatError(
                f"{source}: {doc_id}에 누락된 필드: {sorted(missing)}"
            )
        extra = present - expected_fields
        if extra:
            raise ExtractionTableFormatError(
                f"{source}: {doc_id}에 예상 밖 필드: {sorted(extra)}"
            )

    if not official:
        return

    # --- 공식 규모 ---
    if len(rows) != OFFICIAL_ROW_COUNT:
        raise ExtractionTableFormatError(
            f"{source}: 공식 추출표는 {OFFICIAL_ROW_COUNT}행이어야 합니다(실제 {len(rows)})."
        )
    if len(doc_ids) != OFFICIAL_DOCUMENT_COUNT:
        raise ExtractionTableFormatError(
            f"{source}: 공식 추출표는 {OFFICIAL_DOCUMENT_COUNT}문서여야 합니다(실제 {len(doc_ids)})."
        )
    if len(field_names) != OFFICIAL_FIELD_COUNT:
        raise ExtractionTableFormatError(
            f"{source}: 공식 추출표는 {OFFICIAL_FIELD_COUNT}필드여야 합니다(실제 {len(field_names)})."
        )

    # --- 선언 버전 대조 (폴더·파일 이름이 아니라 파일 안의 값으로 판정) ---
    version_checks: list[tuple[str, Any, Any]] = []
    if cfg is not None:
        version_checks = [
            ("schema_version", doc.get("schema_version"), cfg.get("schema_version")),
            ("extraction_version", doc.get("extraction_version"), cfg.get("extraction_version")),
            ("corpus_version", doc.get("corpus_version"), cfg.get("corpus")),
            ("registry_version", doc.get("registry_version"),
             cfg.get("document_registry_version", cfg.get("corpus"))),
        ]
        mismatches = [
            (k, actual, expected) for k, actual, expected in version_checks
            if expected is not None and actual != expected
        ]
        if mismatches:
            detail = "\n".join(
                f"  - {k}: 파일={actual!r} / 설정 요구={expected!r}"
                for k, actual, expected in mismatches
            )
            raise ExtractionTableFormatError(
                f"{source}: 추출표 버전이 설정과 다릅니다.\n{detail}\n"
                f"폴더·파일 이름이 v3인 것만으로는 인정하지 않습니다."
            )

    # --- 행 단위 버전도 파일 헤더와 같은지 ---
    for key in ("schema_version", "extraction_version", "corpus_version", "registry_version"):
        header_val = doc.get(key)
        if header_val is None:
            raise ExtractionTableFormatError(f"{source}: 헤더에 {key}가 없습니다.")
        row_vals = {r.get(key) for r in rows if key in r}
        bad = {v for v in row_vals if v != header_val}
        if bad:
            raise ExtractionTableFormatError(
                f"{source}: 행의 {key}가 헤더({header_val!r})와 다릅니다: {sorted(bad)[:5]}"
            )

    # --- extraction_metadata.json 대조 ---
    if metadata is not None:
        meta_checks = [
            ("row_count", len(rows), metadata.get("row_count")),
            ("document_count", len(doc_ids), metadata.get("document_count")),
            ("field_count", len(field_names), metadata.get("field_count")),
            ("schema_version", doc.get("schema_version"), metadata.get("schema_version")),
            ("extraction_version", doc.get("extraction_version"), metadata.get("extraction_version")),
            ("corpus_version", doc.get("corpus_version"), metadata.get("corpus_version")),
            ("registry_version", doc.get("registry_version"), metadata.get("registry_version")),
        ]
        bad_meta = [
            (k, a, b) for k, a, b in meta_checks if b is not None and a != b
        ]
        if bad_meta:
            detail = "\n".join(f"  - {k}: 실제집계={a!r} / metadata={b!r}" for k, a, b in bad_meta)
            raise ExtractionTableFormatError(
                f"{source}: 실제 집계와 extraction_metadata.json이 다릅니다.\n{detail}"
            )


def load_extraction_table_document(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extraction_table(
    path: Path,
    cfg: dict[str, Any] | None = None,
    metadata_path: Path | None = None,
    official: bool = False,
) -> list[dict]:
    """공식 추출표를 읽고 검증한 뒤 rows를 반환한다.

    official=True면 공식 규모·버전·메타데이터 대조까지 강제한다."""
    path = Path(path)
    doc = load_extraction_table_document(path)
    metadata = None
    if official:
        if metadata_path is None:
            metadata_path = path.parent / "extraction_metadata.json"
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise ExtractionTableFormatError(
                f"{metadata_path}: 공식 이름표(extraction_metadata.json)가 없습니다 — "
                f"공식 추출표로 인정하지 않습니다."
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    validate_extraction_table(doc, cfg=cfg, metadata=metadata,
                              official=official, source=str(path))
    return doc["rows"]


# ---------------------------------------------------------------------------
# 지역제한 — 뜻으로 판정
# ---------------------------------------------------------------------------

REGION_NONE = "none"            # 실제로 제한 없음
REGION_RESTRICTED = "restricted"  # 실제 제한 조건이 존재
REGION_UNKNOWN = "unknown"      # 판정 불가(빈칸·상태 이상·문구 모호)

_REGION_NONE_PATTERNS = (
    r"제한\s*없", r"지역\s*제한\s*없", r"해당\s*없", r"전국\s*(?:대상|단위|입찰|참가|가능)",
    r"제한\s*이\s*없", r"지역\s*무관", r"제한\s*미적용", r"없음",
)
_REGION_RESTRICTED_PATTERNS = (
    r"제한경쟁", r"지역\s*제한\s*\(", r"소재지", r"소재", r"본점", r"주된\s*영업소",
    r"관내", r"에\s*한함", r"에\s*한한다", r"으로\s*제한", r"로\s*제한", r"제한하여",
    r"[가-힣]{2,10}(?:광역시|특별시|특별자치시|특별자치도|도)\s*(?:에|인|소재|기업|업체)",
)


def classify_region_restriction(row: dict | None) -> tuple[str, str]:
    """(판정, 사유). 빈칸을 '제한 없음'으로 보지 않고, value_present라는
    이유만으로 '제한 있음'으로 보지도 않는다 — 값의 뜻으로 판정한다."""
    if row is None:
        return REGION_UNKNOWN, "추출표에 '지역제한' 행이 없습니다"
    status = row.get("status")
    if status != "value_present":
        return REGION_UNKNOWN, f"'지역제한' 상태가 {status}라 제한 유무를 단정할 수 없습니다"
    text = " ".join(
        str(row.get(k) or "") for k in ("answer_raw", "answer_normalized", "matched_expression")
    ).strip()
    if not text:
        return REGION_UNKNOWN, "'지역제한' 값이 비어 있어 제한 유무를 단정할 수 없습니다"
    if any(re.search(p, text) for p in _REGION_RESTRICTED_PATTERNS):
        return REGION_RESTRICTED, "원문에 지역 제한 조건이 명시돼 있습니다"
    if any(re.search(p, text) for p in _REGION_NONE_PATTERNS):
        return REGION_NONE, "원문에 지역 제한이 없다고 명시돼 있습니다"
    return REGION_UNKNOWN, "'지역제한' 값의 뜻을 자동으로 판정하지 못했습니다"


# ---------------------------------------------------------------------------
# 조건 실행
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    document_id: str
    state: str
    value: Any
    warning: str | None = None
    row: dict | None = None


def run_condition_query(
    table: list[dict], cq: ConditionQuery, max_results: int = 20
) -> tuple[list[QueryResult], int]:
    validate_condition(cq)

    matched: list[QueryResult] = []
    for row in table:
        if row["field_name"] != cq.field or not _row_active(row):
            continue
        status = row["status"]
        value = row.get("answer_normalized")

        if cq.field == "지역제한" and cq.operator in ("no_restriction", "has_restriction"):
            verdict, reason = classify_region_restriction(row)
            want = REGION_NONE if cq.operator == "no_restriction" else REGION_RESTRICTED
            if verdict == want:
                matched.append(QueryResult(row["document_id"], status,
                                           row.get("answer_raw"), row=row))
            elif verdict == REGION_UNKNOWN:
                matched.append(QueryResult(
                    row["document_id"], status, None,
                    warning=f"{row['document_id']}: {reason} — 확인 필요", row=row,
                ))
            continue

        if status == "value_present":
            if _evaluate(row.get("answer_raw") or value, cq.operator, cq.value):
                matched.append(QueryResult(row["document_id"], status, value, row=row))
            elif _parse_korean_amount(row.get("answer_raw") or value) is None:
                matched.append(QueryResult(
                    row["document_id"], status, None,
                    warning=(f"{row['document_id']}: '{cq.field}' 값을 숫자로 읽지 못해 "
                             f"조건 판정에서 제외됨 — 직접 확인 필요"), row=row,
                ))
        elif status == "field_absent":
            matched.append(QueryResult(
                row["document_id"], status, None,
                warning=(f"{row['document_id']}: '{cq.field}'는 원문에 항목 자체가 없습니다"
                         f"(제한이 없다는 뜻으로 단정할 수 없음) — 확인 필요"), row=row,
            ))
        elif status in ("external_reference", "not_disclosed"):
            matched.append(QueryResult(
                row["document_id"], status, None,
                warning=f"{row['document_id']}: '{cq.field}'는 {status}로, 조건 판정에서 제외됨",
                row=row,
            ))
        elif status == "conflict":
            matched.append(QueryResult(
                row["document_id"], status, None,
                warning=(f"{row['document_id']}: '{cq.field}'는 원문 내 값이 서로 달라 "
                         f"자동 판정에서 제외됨 — 직접 확인 필요"), row=row,
            ))
        elif status in _FUTURE_ALLOWED_STATUS:
            matched.append(QueryResult(
                row["document_id"], status, None,
                warning=f"{row['document_id']}: '{cq.field}'는 값을 확인하지 못했습니다({status})",
                row=row,
            ))

    return matched[:max_results], len(matched)


def run_conditions_query(
    table: list[dict], conditions: list[ConditionQuery], max_results: int = 20
) -> tuple[list[QueryResult], int, list[str]]:
    """AND 복합조건 — '깨끗하게 매칭된'(경고 없는) 문서의 교집합만 결과로 삼고,
    경고는 모아서 함께 돌려준다."""
    if not conditions:
        return [], 0, []

    per_condition = [
        run_condition_query(table, cq, max_results=len(table) or 1) for cq in conditions
    ]

    clean_id_sets: list[set[str]] = []
    by_id: dict[str, QueryResult] = {}
    warnings: list[str] = []
    for results, _ in per_condition:
        clean_ids: set[str] = set()
        for r in results:
            if r.warning:
                warnings.append(r.warning)
            else:
                clean_ids.add(r.document_id)
                by_id.setdefault(r.document_id, r)
        clean_id_sets.append(clean_ids)

    intersected = set.intersection(*clean_id_sets) if clean_id_sets else set()
    matched = [by_id[doc_id] for doc_id in sorted(intersected)]
    seen: set[str] = set()
    deduped = [w for w in warnings if not (w in seen or seen.add(w))]
    return matched[:max_results], len(matched), deduped


# ---------------------------------------------------------------------------
# 금액 파서
# ---------------------------------------------------------------------------

_AMOUNT_UNITS: list[tuple[str, int]] = [
    ("조", 10**12), ("억", 10**8),
    ("천만", 10**7), ("백만", 10**6), ("십만", 10**5), ("만", 10**4),
    ("천", 10**3), ("백", 10**2), ("십", 10**1),
]


def _parse_korean_amount(value: Any) -> float | None:
    """원문 그대로인 예산 값을 숫자로 환산. 못 읽으면 None(조건 판정에서 제외)."""
    if value is None:
        return None
    original = str(value)

    comma_matches = list(_GROUPED_NUMBER_RE.finditer(original))
    has_trailing_unit = any(
        re.match(r"\s*[조억만천백십]", original[m.end():]) for m in comma_matches
    )
    if comma_matches and not has_trailing_unit:
        best = max((m.group(0) for m in comma_matches),
                   key=lambda x: len(x.replace(",", "").split(".")[0]))
        try:
            return float(best.replace(",", ""))
        except ValueError:
            return None

    s = re.sub(r"\([^)]*\)", "", original)
    s = s.replace(",", "").replace("￦", "").replace("₩", "").replace("$", "").replace("원", "")
    s = re.sub(r"^\s*금\s*", "", s)
    s = s.strip()
    if not s:
        return None

    total = 0.0
    found = False
    for unit, mult in _AMOUNT_UNITS:
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{unit}", s)
        if m:
            total += float(m.group(1)) * mult
            found = True
            s = s[: m.start()] + s[m.end():]

    remainder = re.sub(r"[^\d.]", "", s)
    if remainder and re.fullmatch(r"\d+(\.\d+)?", remainder):
        total += float(remainder)
        found = True

    return total if found else None


def lookup_field(table: list[dict], document_id: str, field_name: str) -> dict | None:
    for row in table:
        if (row["document_id"] == document_id and row["field_name"] == field_name
                and _row_active(row)):
            return row
    return None


def _evaluate(value: Any, operator: str, target: Any) -> bool:
    v = _parse_korean_amount(value)
    if v is None:
        return False
    try:
        t = float(target)
    except (TypeError, ValueError):
        return False
    return {
        ">=": v >= t, "<=": v <= t, ">": v > t, "<": v < t, "==": v == t,
    }.get(operator, False)
