"""
G-2 — 조건 질의 (4-9-8 확정).
자연어 조건을 허용된 컬럼·연산자로 규칙 기반 변환, 벡터 검색이 아니라
추출 테이블(rfp_extraction_table_v2)을 조회한다.

공식 추출 테이블 형식 (extraction_table_v2.json, JSONL 아님. 1-12-2 확정 기준
100문서×12필드=1200행, 최상위 'rows' 키 안에 행이 있음). 실제 열 이름:
{
  "document_id": "RFP-000001",
  "field_name": "사업 개요" | "사업분야" | "공고일" | "사업기간" | "예산" |
                 "참가 자격(면허·실적)" | "지역제한" | "컨소시엄 요건" |
                 "평가 배점" | "제출 방식" | "필수 제출 서류" | "과업 범위",
  "status": "value_present" | "field_absent" | "external_reference"
            | "not_disclosed" | "conflict",
  "answer_raw": "...",
  "answer_normalized": "500000000" | null,
  "active": "true" | "false"  (문자열!)
}
⚠️ 사업명·발주기관·마감일시는 최종 12필드에서 제거됨 — 조건 질의 대상 아님.
   문서 식별용 메타데이터가 필요하면 document_registry_v2.json을 따로 봐야 함.

허용 컬럼·연산자는 _ALLOWED_FIELDS에 정의 — 여기 없는 필드/연산자는
거부한다(4-9-8 확정: "규칙이든 LLM이든 출력을 그대로 실행하지 않고
반드시 검증 통과 후 실행").
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

State = Literal[
    "value_present", "field_absent", "external_reference", "not_disclosed",
    "conflict",
]
# extraction_failed·review_required는 향후 상태로 허용은 하되, 현재 공식표에는
# 0건이라 State 타입엔 넣지 않는다(4-9-8) — 실제로 나타나면 여기 추가.
_FUTURE_ALLOWED_STATUS = {"extraction_failed", "review_required"}

# 실제 공식 12필드 중 조건 질의 규칙(_CONDITION_PATTERNS)이 지금 다루는 필드만
# 허용한다 — "사업명"·"입찰_참여_마감일"은 4-9-8 v3에서 12필드 밖으로 빠졌으므로
# 이 표에서 조건 질의 대상이 아니다(4-9-8 확정 스키마와 대조 완료).
_ALLOWED_FIELDS = {"예산", "지역제한"}
_ALLOWED_OPERATORS = {">=", "<=", ">", "<", "==", "contains", "is_empty"}

# 12필드 추출형(extract) 질문에서 어느 필드를 묻는지 감지하는 키워드.
# _ALLOWED_FIELDS(조건 질의용)와는 별개 — 여기는 "얼마·언제·뭐야" 같은 단일값
# 조회에 쓰이고, 12필드 전체를 감지 대상으로 삼는다(조건 질의보다 범위가 넓음).
FIELD_KEYWORDS: dict[str, list[str]] = {
    "사업 개요": ["사업개요", "사업 개요", "사업내용", "사업 내용"],
    "사업분야": ["사업분야", "사업 분야"],
    "공고일": ["공고일"],
    "사업기간": ["사업기간", "사업 기간", "계약기간", "수행기간"],
    "예산": ["예산", "사업금액", "계약금액", "사업 금액"],
    "참가 자격(면허·실적)": ["참가자격", "참가 자격", "자격요건", "자격 요건", "면허", "실적"],
    "지역제한": ["지역제한", "지역 제한"],
    "컨소시엄 요건": ["컨소시엄"],
    "평가 배점": ["평가배점", "평가 배점", "배점"],
    "제출 방식": ["제출방식", "제출 방식"],
    "필수 제출 서류": ["제출서류", "제출 서류", "필수서류", "필수 서류", "서류"],
    "과업 범위": ["과업범위", "과업 범위"],
}

_DOC_ID_RE = re.compile(r"RFP-\d{6}")

# "값만 필요"가 아니라 "원문 근거·맥락 설명"까지 필요해 보이는 질문 —
# message.txt 8번의 세 번째 경로("추출형에서 원문 설명이 필요한 경우: F-0 →
# G → I → J → K-2")를 가르는 신호.
_NEEDS_EXPLANATION_KEYWORDS = ["설명해", "왜", "자세히", "근거", "어떻게", "이유"]


def detect_field(question: str) -> str | None:
    """질문에서 12필드 중 어느 필드를 묻는지 감지(첫 매칭만). 못 찾으면 None."""
    for field, keywords in FIELD_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            return field
    return None


def detect_fields(question: str) -> list[str]:
    """질문에서 12필드 중 언급된 필드를 전부 감지(비교형이 "예산이랑
    사업기간만" 처럼 여러 필드를 동시에 요구할 때 씀). FIELD_KEYWORDS
    순서를 그대로 따른다."""
    return [field for field, keywords in FIELD_KEYWORDS.items()
            if any(kw in question for kw in keywords)]


# 마감일은 4-9-8 v3 확정으로 12필드에서 빠졌다 — FIELD_KEYWORDS에 안 넣는다.
# 하지만 실제로 물어보는 질문은 많아서(선별형 필터용 CSV 데이터는 이미 있음),
# 12필드 표 대신 CSV(deadline_map)를 보는 별도 경로로 감지한다.
_DEADLINE_KEYWORDS = ["마감일", "마감 일", "마감기한", "입찰 마감", "제출 마감", "언제까지"]


def detect_deadline_question(question: str) -> bool:
    return any(kw in question for kw in _DEADLINE_KEYWORDS)


def detect_document_id(question: str) -> str | None:
    """질문에 명시된 문서 ID(RFP-000001 형식)를 감지. 없으면 None —
    호출측이 활성 문서 상태(4-14)나 확인 질문으로 보완해야 한다."""
    m = _DOC_ID_RE.search(question)
    return m.group(0) if m else None


def detect_document_ids(question: str) -> list[str]:
    """질문에 명시된 문서 ID를 전부 감지(비교형용). 순서·중복 제거해서 반환."""
    seen = []
    for m in _DOC_ID_RE.finditer(question):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def needs_explanation(question: str) -> bool:
    return any(kw in question for kw in _NEEDS_EXPLANATION_KEYWORDS)

# 자연어 → (필드, 연산자) 매핑 규칙 (baseline 시작값, 표현 다양성 실측 전)
_CONDITION_PATTERNS: list[tuple[str, str, str]] = [
    (r"(\d[\d,]*)\s*(억|만)?\s*원?\s*이상", "예산", ">="),
    (r"(\d[\d,]*)\s*(억|만)?\s*원?\s*이하", "예산", "<="),
    (r"지역제한\s*없는", "지역제한", "is_empty"),
]


@dataclass
class ConditionQuery:
    field: str
    operator: str
    value: Any = None


class InvalidQueryError(ValueError):
    pass


def parse_condition(question: str) -> ConditionQuery | None:
    """자연어 조건 하나만 규칙으로 파싱(하위 호환용). 매칭 안 되면 None.
    복합조건이 있는 질문에는 parse_conditions()를 쓴다 — 이 함수는 첫 매칭
    하나만 보고 나머지를 조용히 버리므로 단일조건 확실한 경우에만 쓴다."""
    conditions, _ = parse_conditions(question)
    return conditions[0] if conditions else None


# 조건을 잇는 연결어 — 이 중 하나로 문장을 나눠 조각마다 패턴 매칭을 시도한다
_CONNECTOR_RE = re.compile(r"(?:이고|그리고|이면서|,\s*|및\s)")


def _match_one(segment: str) -> ConditionQuery | None:
    for pattern, field, op in _CONDITION_PATTERNS:
        m = re.search(pattern, segment)
        if m:
            value = None
            if op in (">=", "<=") and m.groups():
                raw = m.group(1).replace(",", "")
                unit = m.group(2) if len(m.groups()) > 1 else None
                multiplier = {"억": 100_000_000, "만": 10_000}.get(unit, 1)
                value = int(raw) * multiplier
            return ConditionQuery(field=field, operator=op, value=value)
    return None


def parse_conditions(question: str) -> tuple[list[ConditionQuery], bool]:
    """AND 복합조건을 지원하는 파서. 문장을 연결어로 나눠 조각마다 매칭한다.

    반환값: (인식된 조건 목록, 전부_인식됐는지 여부)
    두 번째 값이 False면 문장 일부에서 조건 같아 보이는 조각을 놓쳤다는 뜻 —
    호출측은 조용히 무시하지 말고 확인 질문을 하거나 명시적으로 보류해야
    한다(4-9-8/message.txt 7번 확정: "조건을 하나만 읽고 나머지를 조용히
    무시하지 않는다").
    """
    segments = [s.strip() for s in _CONNECTOR_RE.split(question) if s.strip()]
    if not segments:
        segments = [question]

    matched: list[ConditionQuery] = []
    unmatched = 0
    for seg in segments:
        cq = _match_one(seg)
        if cq is not None:
            matched.append(cq)
        else:
            unmatched += 1

    fully_matched = unmatched == 0 and len(matched) > 0
    return matched, fully_matched


def validate_condition(cq: ConditionQuery) -> None:
    """허용된 컬럼·연산자인지 검증(4-9-8 확정 — 필수 층)."""
    if cq.field not in _ALLOWED_FIELDS:
        raise InvalidQueryError(f"허용 안 된 필드: {cq.field}")
    if cq.operator not in _ALLOWED_OPERATORS:
        raise InvalidQueryError(f"허용 안 된 연산자: {cq.operator}")


class ExtractionTableFormatError(ValueError):
    pass


def load_extraction_table(path: Path) -> list[dict]:
    """공식 추출표(extraction_table_v2.json)를 읽는다.
    ⚠️ JSONL이 아니라 단일 JSON 객체이고, 행은 최상위 'rows' 키 안에 있다.
    행의 실제 열 이름은 status / answer_raw / answer_normalized (state/value 아님).
    로드 직후 100문서×12필드=1200행 정합성을 검증한다."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    if "rows" not in doc:
        raise ExtractionTableFormatError(
            f"{path}: 최상위 객체에 'rows' 키가 없습니다. "
            "존재하지 않는 table.jsonl 형식을 가정하지 마세요."
        )
    rows = doc["rows"]

    expected_docs = doc.get("document_count")
    expected_fields = doc.get("field_count")
    expected_rows = doc.get("row_count")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ExtractionTableFormatError(
            f"{path}: row_count 선언값({expected_rows})과 실제 행 수({len(rows)})가 다릅니다."
        )
    if expected_docs and expected_fields:
        doc_ids = {r["document_id"] for r in rows}
        if len(doc_ids) != expected_docs:
            raise ExtractionTableFormatError(
                f"{path}: document_count 선언값({expected_docs})과 실제 문서 수"
                f"({len(doc_ids)})가 다릅니다."
            )
        if len(rows) != expected_docs * expected_fields:
            raise ExtractionTableFormatError(
                f"{path}: 문서수×필드수({expected_docs}×{expected_fields})가 "
                f"실제 행 수({len(rows)})와 다릅니다."
            )

    allowed_status = set(State.__args__) | _FUTURE_ALLOWED_STATUS
    bad_status = {r.get("status") for r in rows} - allowed_status
    if bad_status:
        raise ExtractionTableFormatError(f"{path}: 허용 안 된 status 값 발견: {bad_status}")

    return rows


@dataclass
class QueryResult:
    document_id: str
    state: State
    value: Any
    warning: str | None = None  # field_absent·extraction_failed 등 경고 문구


def _row_active(row: dict) -> bool:
    # 공식 파일에서 active는 문자열 "true"/"false"로 들어온다(불리언 아님) — 주의
    v = row.get("active", "true")
    return str(v).lower() != "false"


def run_condition_query(
    table: list[dict], cq: ConditionQuery, max_results: int = 20
) -> tuple[list[QueryResult], int]:
    """단일조건으로 문서를 찾는다. (결과 목록, 전체 건수) 반환 —
    잘렸다는 사실과 전체 건수를 항상 함께 알린다(4-9-8 확정)."""
    validate_condition(cq)

    matched: list[QueryResult] = []
    for row in table:
        if row["field_name"] != cq.field or not _row_active(row):
            continue
        status: State = row["status"]
        value = row.get("answer_normalized")

        if status == "value_present":
            if _evaluate(value, cq.operator, cq.value):
                matched.append(QueryResult(row["document_id"], status, value))
        elif status == "field_absent":
            # ⚠️ 정정(리뷰 반영): 기존엔 "항목 없음=is_empty 조건 충족"으로 안전하게
            # 포함시켰으나(4-9-8 원 결정), 이는 4-12-1에서 확정한 "field_absent를
            # '제한 없음'으로 추정하지 않는다" 원칙과 정면으로 충돌한다. 원문에
            # 지역제한 항목 자체가 없는 것과 실제로 제한이 없는 것은 다르다 —
            # 참가 불가능한 사업을 가능하다고 잘못 보여줄 위험이 있어 경고로
            # 내리고 교집합(깨끗한 매칭)에서 제외한다. 사람 확인이 필요한 항목.
            matched.append(
                QueryResult(
                    row["document_id"], status, None,
                    warning=(
                        f"{row['document_id']}: '{cq.field}'는 원문에 항목 자체가 없습니다"
                        f"(제한이 없다는 뜻으로 단정할 수 없음) — 확인 필요"
                    ),
                )
            )
        elif status in ("external_reference", "not_disclosed"):
            matched.append(
                QueryResult(
                    row["document_id"], status, None,
                    warning=f"{row['document_id']}: '{cq.field}'는 {status}로, 조건 판정에서 제외됨",
                )
            )
        elif status in _FUTURE_ALLOWED_STATUS:
            matched.append(
                QueryResult(
                    row["document_id"], status, None,
                    warning=f"{row['document_id']}: '{cq.field}'는 값을 확인하지 못했습니다({status})",
                )
            )
        # conflict는 baseline에서 조건 판정 보류 (원문 내 상충, 별도 처리 필요)

    total = len(matched)
    return matched[:max_results], total


def run_conditions_query(
    table: list[dict], conditions: list[ConditionQuery], max_results: int = 20
) -> tuple[list[QueryResult], int, list[str]]:
    """AND 복합조건. 각 조건을 독립적으로 실행한 뒤 '깨끗하게 매칭된'
    (경고 없는 value_present) 문서 ID의 교집합만 최종 결과로
    삼는다. field_absent를 포함한 경고 있는 상태는 최종 결과에서 제외하되
    경고 문구는 모아서 함께 보여준다."""
    if not conditions:
        return [], 0, []

    per_condition: list[tuple[list[QueryResult], int]] = [
        run_condition_query(table, cq, max_results=len(table)) for cq in conditions
    ]

    clean_id_sets = []
    by_id: dict[str, QueryResult] = {}
    warnings: list[str] = []
    for results, _ in per_condition:
        clean_ids = set()
        for r in results:
            if r.warning:
                warnings.append(r.warning)
            else:
                clean_ids.add(r.document_id)
                by_id[r.document_id] = r
        clean_id_sets.append(clean_ids)

    intersected = set.intersection(*clean_id_sets) if clean_id_sets else set()
    matched = [by_id[doc_id] for doc_id in intersected]
    total = len(matched)
    # 중복 경고 문구는 순서를 지키며 한 번씩만
    seen = set()
    deduped_warnings = [w for w in warnings if not (w in seen or seen.add(w))]
    return matched[:max_results], total, deduped_warnings


# 억/만/천/백/십 — 긴 복합 단위(천만·백만·십만)를 먼저 검사해야 "5천만"이
# "5천"+"만"으로 잘못 쪼개지지 않는다(순서 중요).
_AMOUNT_UNITS: list[tuple[str, int]] = [
    ("조", 10**12), ("억", 10**8),
    ("천만", 10**7), ("백만", 10**6), ("십만", 10**5), ("만", 10**4),
    ("천", 10**3), ("백", 10**2), ("십", 10**1),
]


def _parse_korean_amount(value: Any) -> float | None:
    """실제 공식표의 예산 값은 정규화돼 있지 않고 원문 그대로다
    (예: "352,000,000원(부가가치세 포함)", "1억 5천만 원(부가가치세 포함)",
    "49,500천원"). 괄호 부연·화폐기호·쉼표를 제거하고 억/만/천 단위
    혼합 표현을 숫자로 환산한다. 순수 한글 숫자(예: "삼억이천만원", 아라비아
    숫자 없이 한글로만 쓰인 경우)는 변환하지 못하고 None을 반환한다 —
    이 경우 호출측이 조건 판정에서 제외하고 경고로 노출해야 한다."""
    if value is None:
        return None
    original = str(value)

    # 우선 원문 어디든(괄호 안 포함) 쉼표로 3자리씩 묶인 정확한 숫자가 있으면
    # 그걸 최우선으로 쓴다 — "일금 일억구천오백삼만원정(￦195,030,000, VAT포함)"
    # 처럼 순한글 숫자보다 괄호 안 아라비아 숫자가 더 정확한 원본인 경우가 있다.
    # ⚠️ 버그 수정(리뷰 반영): 단, 그 숫자 바로 뒤에 억/만/천/백/십 같은 단위가
    # 붙어 있으면(예: "49,500천원") 숏컷을 쓰면 안 된다 — 이 경우 "49,500"이
    # 아니라 "49,500 × 1,000"이 진짜 값이다. 단위가 바로 붙은 매치가 하나라도
    # 있으면 숏컷 전체를 포기하고 아래 단위 환산 경로로 넘긴다.
    comma_matches = list(re.finditer(r"\d{1,3}(?:,\d{3})+", original))
    has_trailing_unit = any(
        re.match(r"\s*[억만천백십]", original[m.end():]) for m in comma_matches
    )
    if comma_matches and not has_trailing_unit:
        best = max((m.group(0) for m in comma_matches), key=lambda x: len(x.replace(",", "")))
        return float(best.replace(",", ""))

    s = re.sub(r"\([^)]*\)", "", original)   # 괄호 부연설명 제거
    s = s.replace(",", "").replace("￦", "").replace("$", "").replace("원", "")
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
    """특정 문서의 특정 필드 행 하나를 찾는다. 활성(active) 행만 본다."""
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
