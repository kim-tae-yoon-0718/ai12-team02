"""
G-2 — 조건 질의 및 설정된 공식 추출표 조회.

공식 추출 테이블 형식 (extraction_table_vN.json, JSONL 아님. 최상위 'rows' 키):
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
from dataclasses import dataclass, field
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

# 공식 12필드 — 설정된 extraction_table_vN.json의 "fields"와 정확히 같아야 한다
OFFICIAL_FIELDS: tuple[str, ...] = (
    "사업 개요", "사업분야", "공고일", "사업기간", "예산",
    "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건",
    "평가 배점", "제출 방식", "필수 제출 서류", "과업 범위",
)

# ---------------------------------------------------------------------------
# 조건의 공통 구조 — "필드 + 판정 종류 + 값 + 조건 연결(AND)"
# ---------------------------------------------------------------------------
# 규칙을 끝없이 나열하는 대신 이 네 조각으로만 조건을 표현한다.
KIND_NUMERIC = "numeric"    # 예산 금액 비교
KIND_STATUS = "status"      # 12필드의 기재 상태 조회
KIND_SEMANTIC = "semantic"  # 값의 '뜻' 판정(지역제한 유무, 공동수급 요구 여부)

# --- 상태 조건 연산자 ---
# ⚠️ 서로 다른 상태를 한데 묶지 않는다. value_present가 아니라고 해서 전부
#    field_absent가 아니고, external_reference(외부 문서 참조 안내)·not_disclosed
#    (비공개 명시)·conflict(원문 충돌)는 각각 다른 사실이다.
OP_STATUS_PRESENT = "status_value_present"
OP_STATUS_ABSENT = "status_field_absent"
OP_STATUS_EXTERNAL = "status_external_reference"
OP_STATUS_NOT_DISCLOSED = "status_not_disclosed"
OP_STATUS_CONFLICT = "status_conflict"
OP_STATUS_UNRESOLVED = "status_unresolved"   # extraction_failed / review_required

STATUS_OPERATOR_TARGETS: dict[str, set[str]] = {
    OP_STATUS_PRESENT: {"value_present"},
    OP_STATUS_ABSENT: {"field_absent"},
    OP_STATUS_EXTERNAL: {"external_reference"},
    OP_STATUS_NOT_DISCLOSED: {"not_disclosed"},
    OP_STATUS_CONFLICT: {"conflict"},
    OP_STATUS_UNRESOLVED: set(_FUTURE_ALLOWED_STATUS),
}

# --- 뜻으로 판정하는 연산자 ---
OP_REGION_NONE = "no_restriction"        # 실제로 지역 제한이 없음
OP_REGION_SOME = "has_restriction"       # 실제 지역 제한 조건이 있음
OP_CONSORTIUM_REQUIRED = "consortium_required"          # 공동수급 필수
OP_CONSORTIUM_NOT_REQUIRED = "consortium_not_required"  # 필수는 아님(단독 가능)
OP_CONSORTIUM_ALLOWED = "consortium_allowed"            # 허용(조건부 포함)
OP_CONSORTIUM_FORBIDDEN = "consortium_forbidden"        # 금지

NUMERIC_OPERATORS = {">=", "<=", ">", "<", "=="}
_CONSORTIUM_OPERATORS = {
    OP_CONSORTIUM_REQUIRED, OP_CONSORTIUM_NOT_REQUIRED,
    OP_CONSORTIUM_ALLOWED, OP_CONSORTIUM_FORBIDDEN,
}

# 연산자별로 쓸 수 있는 필드 — 금액 비교를 사업기간에 붙이는 식의 혼선을 막는다
_OPERATOR_FIELDS: dict[str, set[str]] = {}
for _op in NUMERIC_OPERATORS:
    _OPERATOR_FIELDS[_op] = {"예산"}
for _op in (OP_REGION_NONE, OP_REGION_SOME):
    _OPERATOR_FIELDS[_op] = {"지역제한"}
for _op in _CONSORTIUM_OPERATORS:
    _OPERATOR_FIELDS[_op] = {"컨소시엄 요건"}
for _op in STATUS_OPERATOR_TARGETS:
    _OPERATOR_FIELDS[_op] = set(OFFICIAL_FIELDS)

_ALLOWED_FIELDS = set(OFFICIAL_FIELDS)
_ALLOWED_OPERATORS = set(_OPERATOR_FIELDS)


def operator_kind(operator: str) -> str:
    if operator in NUMERIC_OPERATORS:
        return KIND_NUMERIC
    if operator in STATUS_OPERATOR_TARGETS:
        return KIND_STATUS
    return KIND_SEMANTIC

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


# ---------------------------------------------------------------------------
# "제출 방식"의 자연스러운 바꿔 말하기 (2026-09-03)
# ---------------------------------------------------------------------------
# 문제: "제안서를 어떤 방식으로 제출해야 하는지"처럼 필드명을 풀어 쓰면 글자
#       그대로의 "제출 방식"이 없어 추출형으로 분기하지 못했다(실제 재현).
#
# ⚠️ 일반적인 "어떤 방식"을 전부 제출 방식으로 잡으면 "제안서 평가 방식",
#    "사업을 어떤 방식으로 수행"까지 끌려온다. 그래서 아래 **세 가지가 모두**
#    있을 때만 인정하는 좁은 규칙을 쓴다.
#      ① 제출 대상  — 제안서·입찰서류·신청서 …
#      ② 방법을 묻는 말 — 어떤 방식/방법/절차·어떻게
#      ③ 제출 동작  — 제출·접수·내다
#
# ⚠️ 2026-09-03 적대적 점검에서 확인한 결함으로 규칙을 좁혔다. 예전엔 ②와 ③
#    사이에 "아무 글자나 20자"를 허용해서 **절 경계를 넘어** 붙었다.
#      "제안서 평가 방식이 어떻게 되고 제출 기한은 언제인가요?" → 제출 방식(오탐)
#      "제안서 발표 자료는 어떻게 만들어 내나요?"               → 제출 방식(오탐)
#    지금은 ②와 ③ 사이에 **목적어 한 덩이(…을/를/은/는/이/가)만** 허용한다.
#    다른 서술어("되고", "진행되고", "평가해서", "만들어")가 끼면 인정하지 않는다.
#    트레이드오프: "어떻게 제안서와 관련 서류를 제출하나요?"처럼 목적어가 두 덩이인
#    문장은 못 잡는다. 이건 수정 전과 같은 동작(QA 경로)이라 나빠지지 않는다.
_SUBMISSION_TARGET_RE = re.compile(
    r"제안서|제안\s*서류|입찰\s*서류|입찰서|신청서|응찰서|견적서|제출\s*서류|서류|제안요청서"
)
# 제출 동작. ⚠️ "내다"는 어미를 붙여서만 인정한다.
_SUBMIT_VERB = r"제출|접수|내야|내면|내나|내는|낼\s|낼지|냅니"
# 방법을 뜻하는 명사. ⚠️ "식"은 "식당·식자재·식별번호"의 첫 글자로도 걸리므로
#    바로 뒤에 "으로/로"가 올 때만 인정한다("어떤 식으로 제출하나요?").
_METHOD_NOUN = r"방식|방법|절차|경로|형태|수단|식(?=\s*으로)"
_METHOD_ASK = (rf"어떻게|(?:어떤|어떠한|무슨|어느)\s*(?:{_METHOD_NOUN})"
               rf"(?:으로|으론|로|은|는|이|가|을|를)?")
# ②와 ③ 사이에 끼워도 되는 것 — 목적어 한 덩이뿐이다(다른 서술어는 불가).
_OBJECT_NP = r"(?:\s*\S{1,14}(?:을|를|은|는|이|가))?"
_SUBMISSION_METHOD_PATTERNS = (
    # ① "어떤 방식으로 (제안서를) 제출" / "어떻게 (제안서를) 내야"
    rf"(?:{_METHOD_ASK}){_OBJECT_NP}\s*(?:{_SUBMIT_VERB})",
    # ② 어순이 뒤집힌 형태 — "제출은 어떻게", "접수는 어떤 절차로"
    rf"(?:제출|접수)(?:은|는|을|를|이|가)?\s*(?:{_METHOD_ASK})",
)
# 필드명을 그대로 바꿔 부른 표현 — 제출 대상 없이도 이 필드를 가리킨다
_SUBMISSION_METHOD_NAME_RE = re.compile(r"제출\s*방법")


def detect_submission_method_question(question: str) -> bool:
    """질문이 최종 12필드의 "제출 방식"을 묻는지(필드명이 없어도 인정)."""
    if _SUBMISSION_METHOD_NAME_RE.search(question):
        return True
    if not _SUBMISSION_TARGET_RE.search(question):
        return False
    return any(re.search(p, question) for p in _SUBMISSION_METHOD_PATTERNS)


# "서류"라는 낱말이 제출 **대상**으로만 쓰였는지 가른다.
# "입찰서류를 어떻게 제출하나요?"의 서류는 제출 동사의 목적어이고,
# "어떤 서류가 필요하고 어떻게 제출하나요?"·"필요한 서류와 제출 방법을 알려주세요"의
# 서류는 따로 묻는 항목이다(두 항목을 함께 물으면 둘 다 남긴다).
#
# ⚠️ 2026-09-03 적대적 점검 반영: 예전엔 "서류 뒤에 필요/알려/무엇이 오는가"라는
#    어순 목록으로 판정해서 "필요한 서류와 …"처럼 수식어가 앞에 오는 흔한 어순에서
#    사용자가 물은 '필수 제출 서류'를 통째로 지웠다. 이제는 반대로,
#    **질문에 나온 모든 "서류"가 제출 동사의 목적어일 때만** 지운다.
_SUBMIT_OBJECT_DOCUMENT_RE = re.compile(
    rf"서류(?:를|을)\s*(?:{_METHOD_ASK})?\s*(?:{_SUBMIT_VERB})")
# 사용자가 서류 목록을 명시적으로 부른 표현 — 있으면 절대 지우지 않는다
_DOCUMENT_LIST_KEYWORDS = ("제출서류", "제출 서류", "필수서류", "필수 서류")


def _document_word_is_only_submit_object(question: str) -> bool:
    """질문 속 "서류"가 전부 제출 동사의 목적어로만 쓰였는가."""
    if any(kw in question for kw in _DOCUMENT_LIST_KEYWORDS):
        return False
    total = len(re.findall("서류", question))
    if not total:
        return False
    as_object = len(_SUBMIT_OBJECT_DOCUMENT_RE.findall(question))
    return as_object >= total


def _apply_submission_method(question: str, fields: list[str]) -> list[str]:
    """제출 방식 바꿔 말하기를 반영하고, 제출 대상뿐인 '서류'를 걷어낸다.

    필드 순서는 FIELD_KEYWORDS(공식 12필드) 순서를 그대로 지킨다.
    """
    if not detect_submission_method_question(question):
        return fields
    result = list(fields)
    if "제출 방식" not in result:
        result.append("제출 방식")
    if ("필수 제출 서류" in result
            and _document_word_is_only_submit_object(question)):
        result.remove("필수 제출 서류")
    order = list(FIELD_KEYWORDS)
    return sorted(result, key=order.index)


def _fields_from_keywords(question: str) -> list[str]:
    return [field for field, keywords in FIELD_KEYWORDS.items()
            if any(kw in question for kw in keywords)]


def detect_field(question: str) -> str | None:
    fields = detect_fields(question)
    return fields[0] if fields else None


def detect_fields(question: str) -> list[str]:
    """질문에 들어 있는 최종 12필드 이름. 라우터와 실행부가 함께 쓰는 유일한 규칙."""
    return _apply_submission_method(question, _fields_from_keywords(question))


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


# ---------------------------------------------------------------------------
# 조건 표현 사전 — 필드 키워드 뒤의 짧은 창(window)에서 판정 표현을 찾는다
# ---------------------------------------------------------------------------
# 왜 창(window)인가: 조사("이/가/은/는/도/만")·부사("아예/따로/명확히")·
# 수식 명사("조건/요건/안내/정보")는 끝없이 변주된다. 표현마다 규칙을 새로
# 쓰면 목록이 끝나지 않는다. 대신 "필드 키워드 바로 뒤 N글자 안에서 판정
# 표현을 찾는다"는 한 가지 규칙만 두고, 부정 표현이 **어느 필드에 걸리는지**를
# 위치로 결정한다. 창은 다음 필드 키워드를 만나면 거기서 끊는다.
_PREDICATE_WINDOW = 30

# 값이 문서에 '적혀 있다'는 뜻의 어간
_STATED = r"(?:기재|명시|표기|기술|안내|공개|제시|제공|포함|분류|적시|고지|수록|게재|나열)"
_STATED_TAIL = r"(?:되어|돼|되|하)?\s*(?:있는|있음|있고|있어|있다|있으|된|돼|됨|됐|한|해)"

# (연산자, 정규식) — 창 안에서 **가장 먼저 시작하는** 매칭을 쓴다.
# 목록 순서는 같은 위치에서 겹칠 때의 우선순위다(부정이 긍정보다 앞).
_STATUS_PREDICATES: list[tuple[str, str]] = [
    # 외부 문서를 보라는 안내 — '아무 설명이 없음'과 다르다
    (OP_STATUS_EXTERNAL,
     r"(?:별도|다른|외부|타)\s*(?:의\s*)?(?:문서|공고문|공고|자료|서류|파일)"
     r"|(?:입찰\s*)?공고문?\s*(?:을|를|에|엔)?\s*(?:참조|참고|따름|따라|봐야|보아야)"
     r"|붙임\s*(?:문서|자료)?\s*(?:참조|참고)"
     r"|따로\s*(?:봐야|보아야|확인해야|찾아봐야|참고해야)"),
    # 비공개로 '명시'된 경우 — 미기재와 다르다
    (OP_STATUS_NOT_DISCLOSED,
     r"비공개|미공개|공개하지\s*않|공개\s*안\s*(?:하|되|된|함)"),
    # 원문 내용 충돌
    (OP_STATUS_CONFLICT, r"충돌|상충|서로\s*다른|엇갈리|모순"),
    # 추출 실패·추가 판단 필요
    (OP_STATUS_UNRESOLVED, r"추출\s*(?:에\s*)?실패|확인하지\s*못한|판단\s*보류|검토\s*필요"),
    # 미기재(부정)
    (OP_STATUS_ABSENT,
     rf"미기재|누락|빠진|빠져\s*있"
     rf"|안\s*{_STATED}"
     rf"|{_STATED}\s*(?:가|이|은|는|를|을|도)?\s*안\s*(?:되|된|돼|함|해)"
     rf"|{_STATED}(?:되어|돼|되|하)?\s*(?:있지|하지|지)?\s*않"
     rf"|안\s*(?:나와|나온|적혀|적힌|보이|쓰여|써)"
     rf"|(?:나와|적혀|보이|쓰여)(?:\s*있)?지\s*않"
     rf"|없는|없음|없고|없어|없다|없으|없을|없이"
     rf"|찾을\s*수\s*없|확인할\s*수\s*없"),
    # 값이 실제로 적혀 있음
    (OP_STATUS_PRESENT,
     rf"{_STATED}{_STATED_TAIL}"
     rf"|(?:나와|적혀|쓰여|표시(?:되어|돼)?)\s*있"
     rf"|나온|적힌|나와있"
     rf"|확인(?:할\s*수\s*있|되는|됨|된)"
     rf"|있는|있음|있고|있어|있다|있으"),
]

# 필드별 '뜻' 판정 표현 — 기재 여부가 아니라 값의 의미를 묻는 표현이다.
# ⚠️ "지역제한 미기재"와 "지역제한 없음"은 다르고, "공동수급 허용"과
#    "공동수급 필수"도 다르다. 그래서 상태 조건과 분리해 둔다.
_SEMANTIC_PREDICATES: dict[str, list[tuple[str, str]]] = {
    "지역제한": [
        (OP_REGION_NONE,
         r"없는|없음|없고|없어|없다|없으|안\s*걸리|걸리지\s*않|무관|자유"),
        (OP_REGION_SOME, r"있는|있음|있고|있어|있다|있으|걸린|걸려|걸리는|붙은|적용되"),
    ],
    "컨소시엄 요건": [
        # 필수 — "허용"과 절대 같지 않다
        (OP_CONSORTIUM_REQUIRED,
         r"필요한|필요하|필요해|필수|의무|반드시|요구하는|요구되는|구성해야|구성하여야"),
        (OP_CONSORTIUM_NOT_REQUIRED,
         r"필요\s*없|불필요|요구\s*(?:하지\s*)?않|요구\s*안|없이|안\s*해도|않아도|"
         r"단독으로도|단독으로만"),
        (OP_CONSORTIUM_ALLOWED, r"허용|가능한|가능하"),
        (OP_CONSORTIUM_FORBIDDEN, r"불가|불허|금지|안\s*되는"),
    ],
}

# 이 명사가 필드 키워드와 판정 표현 사이에 있으면 "값의 뜻"이 아니라
# "그 항목이 문서에 적혀 있는가"를 묻는 것이다.
#   "지역제한 조건이 명시된 공고"  → 상태(value_present)
#   "지역 제한이 아예 없는 공고"   → 뜻(실제 제한 없음)
_STATUS_NOUN_RE = re.compile(
    r"(?:요건|조건|안내|명시|기재|정보|내용|규정|항목|사항|기준|배점표|표|목록|규모|범위|비중|란)"
)

# 조건이 아니라 배경 설명인 조각과, 못 읽은 조건을 가르는 신호
_SELECTION_TARGET_RE = re.compile(r"(?:공고|사업|문서|프로젝트|용역|과업)")
# "발주기관이 서울인", "담당자가 김철수인" 같은 미지원 필터 표현
_LEFTOVER_ATTRIBUTIVE_RE = re.compile(
    r"[가-힣]{2,}(?:이|가|은|는)\s*[가-힣0-9]{1,}\s*인(?:\s|$|[,.])")
# 이번 범위에서 지원하지 않는 연결(OR) — AND로 바꾸지 않고 명시적으로 알린다
# 그 자체로 OR인 표식 — 앞뒤 문맥을 따질 필요가 없다
_OR_MARKER_UNAMBIGUOUS_RE = re.compile(r"또는|혹은|아니면|이거나|중\s*(?:에서\s*)?하나")
# 다른 뜻으로도 쓰이는 표식 — 바로 앞이 조건을 맺는 표현일 때만 연결어로 본다
#   "서류를 빠뜨리거나 하면 안 되니까 …" 의 '거나'는 조건 연결이 아니다(적대적 점검)
#   "…공고나 …공고" 의 '나'는 명사를 잇는 OR 조사다
_OR_MARKER_AMBIGUOUS_RE = re.compile(
    r"거나|(?:공고|사업|문서|용역|과업|공모|것)\s*(?:이)?나(?=\s)")
_OR_MARKER_RE = re.compile(
    _OR_MARKER_UNAMBIGUOUS_RE.pattern + "|" + _OR_MARKER_AMBIGUOUS_RE.pattern)
_OR_LEFT_ANCHOR_RE = re.compile(
    r"(?:이상|이하|초과|미만|이내|넘는|넘은)\s*$"
    r"|(?:기재|명시|표기|기술|안내|공개|제시|제공|포함|분류)(?:되어|돼|되|하)?"
    r"(?:있|음|은|는|된|됐|한)?\s*$"
    r"|(?:있는|있음|있고|없는|없음|없고|허용|불허|불가|가능)\s*$"
)

# ---------------------------------------------------------------------------
# [2026-09-03 라운드2] "못 읽은 조건"의 기본값을 뒤집는다
# ---------------------------------------------------------------------------
# 예전 규칙: 조건이 안 잡힌 조각은 **기본이 배경 설명(무시)** 이고, 12필드 키워드와
# 선별 대상 명사가 **둘 다** 있을 때만 미해석 조건으로 남겼다. 그래서
#   "예산 5억 이상이고 서울 소재 업체만 참여 가능한 공고를 찾아줘"  → 지역 조건이 사라지고
#   "평가 배점은 80점 이상이고 예산 5억 이상인 공고를 찾아줘"        → 배점 조건이 사라지고
# 둘 다 "조건 전체 해석 성공"으로 답이 나갔다(실제 재현).
#
# 이제는 **조건처럼 생긴 신호가 하나라도 있으면 미해석 조건**으로 남긴다. 배경 설명은
# "조건 신호가 아예 없는 조각"으로 좁게 정의한다. 아래 세 신호는 12필드 밖이라 이번
# 범위에서 지원하지 않지만, 지원하지 않는다는 사실을 반드시 알려야 하는 것들이다.

# ① 수량 + 비교 — 금액(예산) 외 단위도 잡는다. 지원 대상은 예산뿐이므로,
#    잡히기만 하고 실행되지 않으면 "읽지 못한 조건"으로 보고된다.
_QUANT_COMPARATOR_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*[가-힣%]{0,3}\s*(?:이상|이하|초과|미만|넘는|넘은|이내)"
)
# ② 참여 자격·지역을 거는 표현(12필드 밖). "지역 제한 걸리면 골치 아픈데" 같은 배경
#    문장에는 걸리지 않도록 소재지·한정 어구·"~만 참여 가능"만 본다.
# ② 참여 자격·지역을 거는 표현(12필드 밖).
#   ⚠️ 두 갈래로 나눈다. "…만 참여 가능"은 **공고가 거는 자격 조건**이라 그 조각에 대상
#      명사가 없어도(쉼표·연결어로 잘려도) 조건이다. 반면 소재지·본점 같은 낱말만 있는
#      조각은 "본사 소재지가 부산이라" 처럼 말하는 사람 사정일 수 있어, 그 조각이 문서를
#      가리킬 때만 조건으로 본다.
_ELIGIBILITY_SHAPE_RE = re.compile(
    r"만\s*(?:참여|참가|입찰|응찰)\s*(?:가능|할\s*수\s*있)"
    r"|에\s*한(?:함|정|해|한다)|으로\s*제한|로\s*제한"
)
_RESTRICTION_SHAPE_RE = re.compile(
    r"소재(?:지|한)?|본점|주된\s*영업소|관내"
)
# ③ 제외(빼기) 조건 — 반대로 실행하면 정확히 틀린 답이 나온다
# ⚠️ '말고는'은 "그것 말고는 필요 없어"(= 그것만)라는 뜻이 되므로 제외로 보지 않는다.
_EXCLUSION_RE = re.compile(
    r"제외(?!\s*(?:사유|요건|기준|대상|업체|규정|조항|목록))(?:하고|한|하는|하며|하고서|시키고)?"
    r"|빼고|뺀\s|말고(?!는)"
    r"|(?:공고|사업|문서)(?:가|는|은|이)?\s*아닌"
)
# "빼지 말고", "빼먹지 말고", "놓치지 말고" 는 **빼지 말라**는 뜻이라 제외가 아니다
_ANTI_EXCLUSION_RE = re.compile(r"(?:빼|빼먹|놓치|빠뜨리|누락하)지\s*$")
# "X 말고 다른 건 필요 없어" = "X만 달라" — 제외가 아니라 **한정**이다.
# 표식 뒤가 이런 형태면 조건을 뒤집으면 정확히 반대 답이 나간다(적대적 점검에서 확인).
_ONLY_REQUEST_RE = re.compile(
    r"^\s*(?:다른|나머지|딴)\s*(?:건|것|거|공고|사업)?\s*(?:은|는|이|가|도)?\s*"
    r"(?:필요\s*없|안\s*(?:봐도|줘도|해도)|괜찮|됐)"
    r"|^\s*(?:은|는)?\s*(?:필요\s*없|관심\s*없)"
)


def condition_signal(text: str, scope_text: str | None = None) -> str | None:
    """이 조각에 '조건처럼 생긴' 표현이 있는가. 있으면 사람이 읽을 사유를 돌려준다.

    ⚠️ 신호는 그 조각이 **고를 문서에 대한 이야기**일 때만 센다. 선별 대상 명사(공고·사업…)도
       12필드 키워드도 없는 조각은 말하는 사람 사정이지 검색 조건이 아니다.
         "3개월 이내에 끝내야 해서"        → 배경(신호 아님)
         "본사 소재지가 부산이라"          → 배경(신호 아님)
         "서울 소재 업체만 참여 가능한 공고" → 조건(대상 명사 있음)
         "평가 배점은 80점 이상"           → 조건(12필드 키워드 있음)
       이 구분이 없으면 정상 질문이 전부 되묻기로 바뀐다(적대적 점검에서 확인)."""
    # 공고가 거는 자격 조건은 조각이 잘려도 조건이다("…업체만 참여 가능하고, …")
    if _ELIGIBILITY_SHAPE_RE.search(text):
        return "참여 자격 조건"
    scope = scope_text if scope_text is not None else text
    about_documents = bool(_SELECTION_TARGET_RE.search(scope) or find_field_mentions(scope))
    if not about_documents:
        return None
    if _QUANT_COMPARATOR_RE.search(text):
        return "수량 비교 조건"
    if _RESTRICTION_SHAPE_RE.search(text):
        return "지역 제한 조건"
    if _EXCLUSION_RE.search(text):
        return "제외(빼기) 조건"
    if _LEFTOVER_ATTRIBUTIVE_RE.search(text):
        return "지원하지 않는 필터 표현"
    return None
# 결과를 요청하는 표현 — 배경 설명 조각과 실제 요청 조각을 가른다
_REQUEST_MARKERS = (
    "알려줘", "알려 줘", "알려주", "찾아줘", "찾아 줘", "찾아주", "뽑아", "추려",
    "정리해", "보여줘", "보여 줘", "보여주", "골라", "모아", "줄 수 있", "주세요",
    "주실", "달라", "주라", "리스트", "목록 줘", "목록 좀", "걸러", "필터",
)
# 여러 문서를 원한다는 신호(조건이 하나도 안 읽힌 질문의 선별 의도 판단용)
_PLURAL_MARKERS = (
    "공고들", "사업들", "문서들", "목록", "리스트", "전부", "모두", "다 보여",
    "전체", "추천", "골라", "추려", "모아", "뽑아", "것만", "만 보여",
)
# "여러 공고를 골라 달라"가 문장 자체로 분명한 경우 — 조건을 하나도 못 읽었을 때만 쓴다.
# 넓은 _PLURAL_MARKERS 를 그대로 쓰면 "제안요청서 전체 내용 요약해줘" 같은 QA 질문까지
# 선별형으로 끌려온다. 그래서 대상 명사가 직접 복수/목록으로 불린 경우만 본다.
_MULTI_TARGET_RE = re.compile(
    r"(?:공고|사업|문서|공모|용역)\s*(?:들|목록|리스트|전부|전체|모두|다\s)")
# 명시적으로 "전체 공고"를 대상으로 삼는 표현 — 지시 표현이나 기관·사업명이 함께 있어도
# 한 문서에 가두면 안 된다("이 사업과 별개로 전체 공고에서 …").
# ⚠️ 범위를 뜻하는 조사가 붙은 형태만 본다. "사업 전체 개요"처럼 대상이 아니라 내용을
#    가리키는 표현까지 잡으면 특정 문서 질문이 선별형으로 새어나간다.
# ⚠️ 뒤에 오는 조사를 **반드시** 확인한다. 예전엔 "이 사업의 **전체 사업**비가 기재되어
#    있어?" 의 '전체 사업'에 걸려서 특정 문서 질문이 전체 선별로 새어나갔다(적대적 점검).
#    '비/기간/명/자/장' 처럼 낱말이 이어지는 경우는 대상이 아니라 내용을 가리킨다.
_SCOPE_PARTICLE = r"(?:에서|중에서|중|가운데|내에서|을|를|의|는|은|도|에|들|[\s,.]|$)"
_GLOBAL_SCOPE_RE = re.compile(
    rf"전체\s*(?:공고|사업|문서|공모)(?:들)?\s*{_SCOPE_PARTICLE}"
    rf"|(?:공고|사업|문서|공모)\s*(?:전체|전부)\s*(?:에서|중|가운데|를|을)"
    rf"|모든\s*(?:공고|사업|문서|공모)(?:들)?\s*{_SCOPE_PARTICLE}"
    rf"|전\s*(?:공고|사업)\s*(?:에서|중|을|를)"
)

# 금액 비교 — 필드 키워드가 앞에 없으면 예산으로 본다(기존 동작 유지).
# 앞에 **다른 필드** 키워드가 붙어 있으면 그 필드의 숫자 조건이므로
# 예산으로 읽지 않고 '미지원'으로 남긴다(사업기간 6개월 → 예산 6 금지).
_AMOUNT_CONDITION_PATTERNS: list[tuple[str, str]] = [
    (rf"{_AMOUNT_EXPR}원?\s*이상", ">="),
    (rf"{_AMOUNT_EXPR}원?\s*이하", "<="),
    (rf"{_AMOUNT_EXPR}원?\s*이내", "<="),
    (rf"{_AMOUNT_EXPR}원?\s*초과", ">"),
    (rf"{_AMOUNT_EXPR}원?\s*미만", "<"),
    (rf"{_AMOUNT_EXPR}원?\s*(?:을|를)?\s*(?:넘는|넘은|넘어가는|넘어서는|넘게)", ">"),
    (rf"{_AMOUNT_EXPR}원?\s*(?:보다)?\s*(?:작은|적은|안\s*되는|안되는)", "<"),
]
_NUMERIC_BINDING_WINDOW = 15   # 금액 앞 이 거리 안의 필드 키워드에 묶는다
# 금액 비교 뒤에 붙는 부정("5억 이상이 아닌"). 대상 명사(_TAIL_STOP_RE)를 넘지 않는다.
_NUMERIC_NEGATION_SPAN = 8
_NUMERIC_NEGATION_RE = re.compile(r"(?:이|가)?\s*(?:아닌|아니|않)")
# 비교 연산자의 정확한 여집합 — 알려진 금액에 대해서만 성립한다
_NUMERIC_FLIP = {">=": "<", "<": ">=", "<=": ">", ">": "<="}
# 금액 바로 앞의 명사 — 12필드가 아니면 그 명사의 수치이므로 예산으로 읽지 않는다
_AMOUNT_OWNER_RE = re.compile(r"([가-힣]{2,6})\s*(?:이|가|은|는|의)?\s*$")

# 필드 키워드가 없어도 그 자체로 조건인 표현
_STANDALONE_PATTERNS: list[tuple[str, str, str]] = [
    (r"전국\s*(?:에서|어디서나|어디든|어디에서나)?\s*(?:참여|응찰|입찰|참가)\s*"
     r"(?:가능|되는|할\s*수\s*있)", "지역제한", OP_REGION_NONE),
]

# (호환 유지) 예전 이름 — 금액 패턴만 (필드, 연산자) 형태로 노출한다
_CONDITION_PATTERNS: list[tuple[str, str, str]] = [
    (pat, "예산", op) for pat, op in _AMOUNT_CONDITION_PATTERNS
]


@dataclass
class ConditionQuery:
    field: str
    operator: str
    value: Any = None
    raw_text: str = ""
    kind: str = ""            # numeric / status / semantic (비면 operator로 유추)
    segment_index: int = 0    # 몇 번째 조각에서 읽었는지(배경 설명 판별용)
    from_request: bool = False  # 결과를 요청하는 조각에서 읽었는지
    negated: bool = False     # "…인 공고는 제외하고" — 조건을 뒤집어 적용한다

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = operator_kind(self.operator)

    def signature(self) -> tuple:
        return (self.field, self.operator, self.value, self.negated)

    def describe(self) -> str:
        return describe_condition(self)


class InvalidQueryError(ValueError):
    pass


_OPERATOR_LABELS = {
    ">=": "이상", "<=": "이하", ">": "초과", "<": "미만", "==": "같음",
    OP_STATUS_PRESENT: "값이 기재돼 있음",
    OP_STATUS_ABSENT: "원문에 기재 없음",
    OP_STATUS_EXTERNAL: "외부 문서 참조 안내",
    OP_STATUS_NOT_DISCLOSED: "비공개로 명시",
    OP_STATUS_CONFLICT: "원문 내용 충돌",
    OP_STATUS_UNRESOLVED: "추출 실패·추가 판단 필요",
    OP_REGION_NONE: "실제 지역 제한 없음",
    OP_REGION_SOME: "실제 지역 제한 있음",
    OP_CONSORTIUM_REQUIRED: "공동수급 필수",
    OP_CONSORTIUM_NOT_REQUIRED: "공동수급 필수 아님(단독 가능)",
    OP_CONSORTIUM_ALLOWED: "공동수급 허용",
    OP_CONSORTIUM_FORBIDDEN: "공동수급 금지",
}


def describe_condition(cq: ConditionQuery) -> str:
    label = _OPERATOR_LABELS.get(cq.operator, cq.operator)
    if cq.kind == KIND_NUMERIC:
        base = f"{cq.field} {format_amount(cq.value)} {label}"
    else:
        base = f"{cq.field}: {label}"
    return f"{base} — 제외(해당하지 않는 문서만)" if cq.negated else base


def format_amount(value: Any) -> str:
    if value is None:
        return "(금액 미상)"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(v):,}원" if v == int(v) else f"{v:,}원"


def _amount_from_match(m: re.Match) -> float | None:
    """조건 표현에서 잡은 금액 덩어리를 실제 금액으로 환산.
    표 값 파싱(_parse_korean_amount)과 같은 규칙을 써서 두 경로가 어긋나지 않게 한다."""
    groups = m.groups()
    if not groups:
        return None
    expr = (groups[0] or "").strip()
    if not expr:
        return None
    return _parse_korean_amount(expr)


# --- 필드 키워드 위치 스캔 -------------------------------------------------

_FIELD_KEYWORD_ITEMS = sorted(
    ((kw, field) for field, kws in FIELD_KEYWORDS.items() for kw in kws),
    key=lambda t: -len(t[0]),
)


def find_field_mentions(text: str) -> list[tuple[int, int, str]]:
    """문장 안 12필드 키워드 위치 (시작, 끝, 필드명). 최장 일치·겹침 제거."""
    spans: list[tuple[int, int, str]] = []
    for kw, field in _FIELD_KEYWORD_ITEMS:
        start = 0
        while (i := text.find(kw, start)) != -1:
            end = i + len(kw)
            if not any(i < e and end > s for s, e, _ in spans):
                spans.append((i, end, field))
            start = i + 1
    return sorted(spans)


def _window_after(text: str, mentions: list[tuple[int, int, str]], idx: int) -> tuple[int, str]:
    """idx번째 필드 언급 뒤의 판정 표현 탐색 창. (창 시작 위치, 창 문자열)"""
    _, end, _ = mentions[idx]
    stop = min(end + _PREDICATE_WINDOW, len(text))
    for s, _e, _f in mentions[idx + 1:]:
        if s > end:
            stop = min(stop, s)
            break
    return end, text[end:stop]


# 판정 표현 **뒤**에 붙어 뜻을 뒤집는 꼬리.
#   "공동수급이 허용되지 않는"  → 허용(X) → 금지(O)
#   "공동수급을 필수로 요구하지 않는" → 필수(X) → 필수 아님(O)
# 예전에는 창 안에서 가장 먼저 시작하는 매칭만 골라서 뒤의 부정을 보지 못했다.
_NEGATION_TAIL_RE = re.compile(
    r"지\s*않|안\s*(?:되|됨|함|해|하)|못\s*(?:하|해|되)|불가|불허|금지|아닌|아니"
)
_NEGATION_TAIL_SPAN = 10   # 판정 표현 뒤 이만큼 안의 부정만 그 판정에 걸린 것으로 본다
# 꼬리 탐색을 여기서 끊는다 — 대상 명사가 나오면 그 뒤는 다른 절이다
_TAIL_STOP_RE = re.compile(r"공고|사업|문서|용역|과업|공모|것|건")

# 뜻이 정확히 뒤집히는 짝. 여기에 없는 연산자는 부정을 만나면 '미해석'으로 남긴다
# (외부 참조·비공개·충돌의 부정은 뜻이 하나로 정해지지 않는다).
_NEGATED_OPERATOR = {
    OP_CONSORTIUM_ALLOWED: OP_CONSORTIUM_FORBIDDEN,
    OP_CONSORTIUM_FORBIDDEN: OP_CONSORTIUM_ALLOWED,
    OP_CONSORTIUM_REQUIRED: OP_CONSORTIUM_NOT_REQUIRED,
    OP_CONSORTIUM_NOT_REQUIRED: OP_CONSORTIUM_REQUIRED,
    OP_REGION_NONE: OP_REGION_SOME,
    OP_REGION_SOME: OP_REGION_NONE,
    # ⚠️ 상태는 6가지라 여집합이 하나로 정해지지 않는다.
    #    NOT(미기재) = {값 있음, 외부 참조, 비공개, 충돌, 추출 실패}이지 '값 있음'이 아니다.
    #    그래서 상태 쌍은 여기 넣지 않고 미해석으로 되묻는다(적대적 점검에서 확인).
    #    "기재되지 않은"처럼 부정이 표현 자체에 들어간 경우는 애초에 미기재 패턴이 잡는다.
}
NEGATION_UNSUPPORTED = "__negation_unsupported__"


def _predicate_in_window(field: str, window: str) -> tuple[str, int, str] | None:
    """창 안에서 가장 먼저 시작하는 판정 표현. (연산자, 위치, 매칭 문구)

    매칭 **뒤**에 부정 꼬리가 붙어 있으면 뜻을 뒤집는다. 뒤집을 짝이 없는 연산자면
    NEGATION_UNSUPPORTED 를 돌려줘서 호출측이 '읽지 못한 조건'으로 남기게 한다."""
    best: tuple[str, int, str] | None = None

    def _consider(op: str, m: re.Match) -> None:
        nonlocal best
        if best is None or m.start() < best[1]:
            best = (op, m.start(), m.group(0))

    semantic = _SEMANTIC_PREDICATES.get(field)
    if semantic:
        # 상태 명사가 판정 표현보다 앞에 있으면 "기재 여부" 질문이다
        for op, pat in semantic:
            m = re.search(pat, window)
            if m:
                noun = _STATUS_NOUN_RE.search(window[:m.start()])
                if noun is None:
                    _consider(op, m)
    for op, pat in _STATUS_PREDICATES:
        m = re.search(pat, window)
        if m:
            _consider(op, m)
    if best is None:
        return None

    op, pos, text = best
    tail = window[pos + len(text): pos + len(text) + _NEGATION_TAIL_SPAN]
    # ⚠️ 꼬리가 대상 명사를 넘어가면 다른 절의 부정이다.
    #    "공동수급이 허용되는 공고**인지 아닌지**" 의 '아닌'은 '허용'에 걸리지 않는다.
    cut = _TAIL_STOP_RE.search(tail)
    if cut:
        tail = tail[:cut.start()]
    if _NEGATION_TAIL_RE.search(tail):
        flipped = _NEGATED_OPERATOR.get(op)
        if flipped is None:
            return (NEGATION_UNSUPPORTED, pos, text)
        return (flipped, pos, text + tail[:_NEGATION_TAIL_SPAN].rstrip())
    return best


def _numeric_conditions(segment: str, mentions: list[tuple[int, int, str]]
                        ) -> tuple[list[tuple[int, int, ConditionQuery]], set[str], list[str]]:
    """금액 비교 조건. (구간·조건, 소비한 필드, 미지원 사유)"""
    found: list[tuple[int, int, ConditionQuery]] = []
    unsupported: list[str] = []
    covered: set[str] = set()
    for pattern, base_op in _AMOUNT_CONDITION_PATTERNS:
        for m in re.finditer(pattern, segment):
            # 매칭마다 원래 연산자에서 시작한다. 앞 조건의 부정이 다음 조건에 전파되지 않는다.
            op = base_op
            value = _amount_from_match(m)
            if value is None:
                continue
            owner = None
            for s, e, f in mentions:
                if e <= m.start() and m.start() - e <= _NUMERIC_BINDING_WINDOW:
                    owner = (e, f)
            if owner is None:
                # 12필드 키워드가 없을 때만 예산으로 본다("5억 이상인 사업 알려줘").
                # ⚠️ 바로 앞에 다른 명사가 있으면 그 명사의 수치다 —
                #    "우리 회사는 **매출** 100억 이상이라" 를 예산으로 읽으면 안 된다.
                lead = _AMOUNT_OWNER_RE.search(segment[:m.start()])
                if lead:
                    # 조각이 '고를 문서'에 대한 이야기가 아니면 말하는 사람 사정이다.
                    # "우리 회사는 매출 100억 이상이라" → 배경(예산 조건도, 미지원 안내도 아님)
                    about_documents = bool(_SELECTION_TARGET_RE.search(segment) or mentions)
                    if about_documents:
                        unsupported.append(
                            f"'{lead.group(1)}'의 숫자 조건({m.group(0).strip()})은 이번 "
                            f"범위에서 지원하지 않습니다 — 예산 금액 비교만 지원합니다")
                    continue
            field = owner[1] if owner else "예산"
            if field != "예산":
                unsupported.append(
                    f"'{field}'의 숫자 조건({m.group(0).strip()})은 이번 범위에서 "
                    f"지원하지 않습니다 — 예산 금액 비교만 지원합니다")
                continue
            if owner:
                covered.add("예산")
            # 금액 비교에 붙은 명확한 부정을 반대 연산자로 바꾼다.
            #   "5억 이상이 아닌" → 5억 미만 / "1억 이하가 아닌" → 1억 초과
            # ⚠️ 미기재·충돌처럼 금액을 못 읽는 문서는 반대 조건에 자동으로 들어가지
            #    않는다 — evaluate_condition 이 그대로 '판단 불가'로 남긴다.
            end = m.end()
            tail = segment[end:end + _NUMERIC_NEGATION_SPAN]
            cut = _TAIL_STOP_RE.search(tail)
            if cut:
                tail = tail[:cut.start()]
            if _NUMERIC_NEGATION_RE.search(tail):
                flipped = _NUMERIC_FLIP.get(op)
                if flipped is None:
                    unsupported.append(
                        f"'{m.group(0).strip()}'에 붙은 부정 표현을 정확히 해석하지 "
                        f"못했습니다")
                    continue
                op = flipped
                end = end + len(tail)
            found.append((m.start(), end,
                          ConditionQuery(field="예산", operator=op, value=value,
                                         raw_text=segment[m.start():end].strip(),
                                         kind=KIND_NUMERIC)))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, ConditionQuery]] = []
    for start, end, cq in found:
        if any(start < k_end and end > k_start for k_start, k_end, _ in kept):
            continue
        kept.append((start, end, cq))
    return kept, covered, unsupported


# 조건 끝과 제외 표식 사이에 올 수 있는 것 — 대상 명사와 조사뿐
_EXCLUSION_GAP_RE = re.compile(r"^[\s]*(?:공고|사업|문서|용역|과업|것|건)?\s*"
                               r"(?:는|은|를|을|만|도|의)?\s*$")


def _exclusion_attaches(segment: str, hits: list[tuple[int, int, "ConditionQuery"]],
                        marker_pos: int) -> bool:
    """제외 표식이 읽은 조건 바로 뒤에 붙어 있는가."""
    ends = [e for _s, e, _c in hits if e <= marker_pos]
    if not ends:
        return False
    return bool(_EXCLUSION_GAP_RE.match(segment[max(ends):marker_pos]))


# 필드에 구체적인 '값'을 지정한 조건 — "사업분야가 인공지능", "제출 방식은 방문접수".
# 이번 범위는 상태(기재/미기재/…)와 예산 금액 비교만 지원하므로, 이런 값 조건은
# 읽지 못한 조건으로 남겨 되물어야 한다. 조용히 버리면 다른 값을 가진 문서가 답이 된다.
_FIELD_VALUE_RE = re.compile(
    r"^\s*(?:이|가|은|는|을|를)\s*"
    r"(?P<value>[가-힣A-Za-z0-9][가-힣A-Za-z0-9·\-]{0,11}"
    r"(?:\s+[가-힣A-Za-z0-9][가-힣A-Za-z0-9·\-]{0,11})?)\s*"
    r"(?:이고|이며|이다|입니다|임|인(?![가-힣])|$)"
)
# 값처럼 보이지만 실제로는 배경 서술("사업분야가 궁금해서 그런데")
_NOT_A_VALUE_RE = re.compile(
    r"해서|라서|어서|아서|니까|는데|런데|한데|려고|면서|지만|거든|어야|아야"
    r"|싶|봐야|챙겨야|걸려|하고\s*있"
)
# 값 자체를 요구하는 문장과 준비·확인 같은 배경 행동을 구분한다.
# "인공지능이어야"/"방문접수여야"는 값 조건이며 "서류를 준비해야"는 여기에 해당하지 않는다.
_FIELD_VALUE_REQUIREMENT_RE = re.compile(
    r"^\s*(?:이|가|은|는|을|를)\s*(?P<value>.+?)"
    r"(?:이어야|이여야|여야|이\s*아니어야|가\s*아니어야)(?:만)?(?:\s|하|되|$)"
)
# 기재 여부를 읽었다고 그 앞의 구체적인 값을 버리지 않는다("방문접수로 기재된").
# ⚠️ 조사 '로/으로'를 낱말 안에서 끊지 않도록 조사까지 함께 잡는다. 예전 정규식은
#    부사 "따로"를 "따" + "로"로 잘라 값으로 읽어서, "사업분야가 **따로** 분류·명시된
#    공고"(SEL-003)가 미지원 값 조건이 돼 정상 답변이 되묻기로 바뀌었다(서버 검증에서 확인).
_LITERAL_STATED_VALUE_RE = re.compile(
    rf"^\s*(?:이|가|은|는|을|를)?\s*(?P<value>.+?)\s*(?P<particle>으로|로)\s*{_STATED}"
)
# 아래는 값이 아니라 '어떻게 기재됐는가'를 꾸미는 일반적인 부사다.
# 조사를 붙인 형태(따로·별도로…)까지 함께 걸러야 낱말이 잘리지 않는다 —
# "따"+"로" 처럼 조각으로 잘린 경우는 이 목록에 붙여서 가려낸다.
# ⚠️ 글자 수로 거르지 않는다. 한 글자여도 "웹"·"앱"·"0" 은 진짜 값이며,
#    이를 버리면 값 조건을 조용히 잃고 다른 값의 문서까지 답으로 확정된다.
#    부사는 여기 확인된 낱말로만 예외 처리한다.
_STATED_VALUE_ADVERBS = frozenset({
    "명확히", "명확", "구체적", "상세", "정확", "별도", "정식", "공식",
    "직접", "간접", "실제", "일반적", "명시적", "독립적",
    "따로", "새로", "서로", "제대로", "그대로", "별도로", "구체적으로",
    "명확하게", "상세히", "정확히",
})


def _states_field_value(segment: str, mentions: list[tuple[int, int, str]],
                        field_name: str, *, strong_only: bool = False) -> str | None:
    """못 읽은 필드에 구체적인 값이 지정돼 있으면 그 값을 돌려준다."""
    for _s, end, f in mentions:
        if f != field_name:
            continue
        # 다른 필드의 요구 표현을 현재 필드에 잘못 붙이지 않도록 그 앞에서 끊는다.
        stop = next((s for s, _e, _f in mentions if s >= end), len(segment))
        tail = segment[end:stop]
        requirement = _FIELD_VALUE_REQUIREMENT_RE.match(tail)
        if requirement:
            return requirement.group("value").strip()
        literal = _LITERAL_STATED_VALUE_RE.match(tail)
        if literal:
            literal_value = literal.group("value").strip()
            # 조사를 붙인 형태도 부사인지 확인한다("따" + "로" → "따로")
            with_particle = literal_value + literal.group("particle")
            if (literal_value not in _STATED_VALUE_ADVERBS
                    and with_particle not in _STATED_VALUE_ADVERBS):
                return literal_value
        if strong_only:  # 이미 읽은 상태·뜻 조건 자체를 값 조건으로 중복 취급하지 않는다.
            continue
        m = _FIELD_VALUE_RE.match(tail)
        if m is None:
            continue
        value = m.group("value").strip()
        if not value or _NOT_A_VALUE_RE.search(value):   # 배경 서술은 값이 아니다
            continue
        return value
    return None


_UNCOVERED_TARGET_SPAN = 16   # 필드 키워드가 이만큼 안의 대상 명사를 수식하면 조건으로 본다


def _modifies_target(segment: str, mentions: list[tuple[int, int, str]],
                     field_name: str) -> bool:
    """못 읽은 필드 키워드가 가까운 대상 명사(공고·사업…)를 수식하는가."""
    for _s, end, f in mentions:
        if f != field_name:
            continue
        if _SELECTION_TARGET_RE.search(segment[end:end + _UNCOVERED_TARGET_SPAN]):
            return True
    return False


def _points_at_document(segment: str, marker_pos: int) -> bool:
    """제외 표식 바로 앞이 '그 사업 / 이 공고' 처럼 문서를 가리키는 표현인가."""
    from doc_resolver import has_anaphora
    prefix = segment[max(0, marker_pos - 14):marker_pos]
    return has_anaphora(prefix) or bool(_DOC_ID_RE.search(prefix))


def _leftover_text(segment: str, hits: list[tuple[int, int, ConditionQuery]],
                   mentions: list[tuple[int, int, str]]) -> str:
    """읽은 조건 구간과 필드 키워드를 지운 나머지 — 아직 못 읽은 조건 탐지용."""
    chars = list(segment)
    spans = [(s, e) for s, e, _ in hits] + [(s, e) for s, e, _ in mentions]
    for start, end in spans:
        for i in range(max(start, 0), min(end, len(chars))):
            chars[i] = " "
    return "".join(chars)


@dataclass
class SelectionParse:
    """질문 하나를 읽은 결과 — 조건·미해석 조각·배경 설명을 모두 남긴다."""

    conditions: list[ConditionQuery] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    background: list[str] = field(default_factory=list)
    unsupported_or: bool = False
    notes: list[str] = field(default_factory=list)
    targets_specific_document: bool = False
    has_request_marker: bool = False
    has_plural_marker: bool = False
    has_global_scope: bool = False

    @property
    def fully_matched(self) -> bool:
        return (not self.unresolved and not self.unsupported_or
                and len(self.conditions) > 0)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(m in text for m in markers)


def _resolve_same_field_conflicts(
    conditions: list[ConditionQuery],
) -> tuple[list[ConditionQuery], list[str], list[str]]:
    """같은 필드에 서로 다른 상태 조건이 잡혔을 때 정리한다.

    "예산을 비공개로 두는 공고도 있다고 해서, 예산 정보가 안 나와 있는 공고 알려줘"
    처럼 앞은 배경 설명, 뒤가 실제 요청인 경우가 있다. 요청 조각의 조건을 쓰고
    배경 쪽은 버린 사실을 기록한다. 어느 쪽이 요청인지 가릴 수 없으면 모호한
    질문이므로 되묻는다(조용히 한쪽을 고르지 않는다)."""
    kept: list[ConditionQuery] = []
    dropped: list[str] = []
    ambiguous: list[str] = []
    by_field: dict[str, list[ConditionQuery]] = {}
    for cq in conditions:
        if cq.kind == KIND_STATUS:
            by_field.setdefault(cq.field, []).append(cq)
        else:
            kept.append(cq)

    for field_name, group in by_field.items():
        operators = {c.operator for c in group}
        if len(operators) == 1:
            kept.append(group[0])
            continue
        preferred = [c for c in group if c.from_request]
        if len({c.operator for c in preferred}) == 1 and preferred:
            kept.append(preferred[0])
            for c in group:
                if c.operator != preferred[0].operator:
                    dropped.append(
                        f"'{field_name}'의 '{c.raw_text}'는 배경 설명으로 보고 "
                        f"조건에서 제외했습니다(실제 요청은 '{preferred[0].raw_text}')")
            continue
        ambiguous.append(
            f"'{field_name}'에 서로 다른 조건이 함께 있습니다: "
            + ", ".join(sorted(_OPERATOR_LABELS.get(o, o) for o in operators)))
    kept.sort(key=lambda c: (c.segment_index, c.field, c.operator))
    return kept, dropped, ambiguous


def parse_selection(question: str) -> SelectionParse:
    """질문을 '필드 + 판정 종류 + 값 + AND 연결'로 읽는다.

    ① 연결어(쉼표·이고·그리고…)로 조각을 나눈다(숫자 안 쉼표는 보호).
    ② 조각마다 필드 키워드를 찾고, 그 **뒤 짧은 창**에서 판정 표현을 찾는다.
    ③ 조건이 하나도 없는 조각은
       - 미지원 조건 신호(다른 필드 키워드 + 선별 대상 명사, "X가 Y인", 금액 표현)가
         있으면 **미해석 조건**으로 남기고(조용히 버리지 않는다),
       - 아니면 배경 설명으로 보고 무시한다.
    """
    parse = SelectionParse()
    parse.targets_specific_document = bool(_DOC_ID_RE.search(question))
    parse.has_request_marker = _has_any(question, _REQUEST_MARKERS)
    parse.has_plural_marker = _has_any(question, _PLURAL_MARKERS)
    parse.has_global_scope = bool(_GLOBAL_SCOPE_RE.search(question))

    masked, values = mask_grouped_numbers(question)
    segments_masked = [s for s in _CONNECTOR_RE.split(masked) if s.strip()]
    if not segments_masked:
        segments_masked = [masked]

    # --- OR 연결 감지 ---
    # ⚠️ 예전에는 **서로 다른 필드 언급 사이**만 봐서, "예산 5억 이상 또는 1억 이하"처럼
    #    같은 필드 안의 OR을 놓치고 두 조건을 AND로 실행해 0건이라고 답했다(실제 재현).
    #    이제 OR 표현의 좌우에 조건 닻(필드 언급 또는 수량 비교)이 있으면 필드가 같든
    #    다르든 미지원으로 알린다.
    _OR_ANCHOR_SPAN = 28
    for m in _OR_MARKER_RE.finditer(question):
        left = question[max(0, m.start() - _OR_ANCHOR_SPAN):m.start()]
        right = question[m.end():m.end() + _OR_ANCHOR_SPAN]
        def _anchored(text: str) -> bool:
            return bool(find_field_mentions(text) or _QUANT_COMPARATOR_RE.search(text))
        # 애매한 표식은 바로 앞이 조건을 맺는 표현이어야 연결어다
        if not _OR_MARKER_UNAMBIGUOUS_RE.match(m.group(0)):
            if not _OR_LEFT_ANCHOR_RE.search(question[max(0, m.start() - 10):m.start()]):
                continue
        if _anchored(left) and _anchored(right):
            parse.unsupported_or = True
            parse.notes.append(
                f"'{question[max(0, m.start() - 12):m.end() + 12].strip()}' — "
                f"OR(또는·이거나) 연결은 이번 범위에서 지원하지 않습니다. "
                f"AND로 바꾸거나 한쪽만 적용하지 않습니다.")
            break

    raw_conditions: list[ConditionQuery] = []
    for seg_index, seg_masked in enumerate(segments_masked):
        seg = unmask_grouped_numbers(seg_masked, values).strip()
        if not seg:
            continue
        from_request = _has_any(seg, _REQUEST_MARKERS)
        mentions = find_field_mentions(seg)
        hits: list[tuple[int, int, ConditionQuery]] = []
        covered_fields: set[str] = set()

        num_hits, num_covered, num_unsupported = _numeric_conditions(seg, mentions)
        hits.extend(num_hits)
        covered_fields |= num_covered
        for reason in num_unsupported:
            if reason not in parse.unresolved:
                parse.unresolved.append(reason)

        for i in range(len(mentions)):
            start, end, field_name = mentions[i]
            win_start, window = _window_after(seg, mentions, i)
            hit = _predicate_in_window(field_name, window)
            if hit is None:
                continue
            op, offset, text = hit
            if op == NEGATION_UNSUPPORTED:
                # "외부 참조가 아닌" 처럼 뒤집을 짝이 없는 부정 — 조용히 버리지 않는다
                msg = (f"'{field_name}' 조건의 부정 표현을 정확히 해석하지 못했습니다: "
                       f"\"{seg}\"")
                if msg not in parse.unresolved:
                    parse.unresolved.append(msg)
                covered_fields.add(field_name)
                continue
            covered_fields.add(field_name)
            # 사람이 읽을 근거 문구 — 필드 키워드부터 판정 표현 끝까지 원문 그대로
            raw = seg[start:win_start + offset + len(text)].strip()
            hits.append((start, win_start + offset + len(text),
                         ConditionQuery(field=field_name, operator=op,
                                        raw_text=raw, kind=operator_kind(op))))

        for pattern, field_name, op in _STANDALONE_PATTERNS:
            m = re.search(pattern, seg)
            if m:
                covered_fields.add(field_name)
                hits.append((m.start(), m.end(),
                             ConditionQuery(field=field_name, operator=op,
                                            raw_text=m.group(0).strip(),
                                            kind=operator_kind(op))))

        hits.sort(key=lambda t: t[0])
        for _s, _e, cq in hits:
            cq.segment_index = seg_index
            cq.from_request = from_request
            raw_conditions.append(cq)

        # --- 못 읽은 조건이 남았는지 — 배경 설명과 구분한다 ---
        # ⚠️ 기본값을 뒤집었다. 조건 신호가 하나라도 남아 있으면 미해석 조건이고,
        #    신호가 아예 없을 때만 배경 설명으로 무시한다.
        uncovered = [f for _s, _e, f in mentions if f not in covered_fields]
        target = bool(_SELECTION_TARGET_RE.search(seg))
        leftover = _leftover_text(seg, hits, mentions)

        def _unread(msg: str) -> None:
            if msg not in parse.unresolved:
                parse.unresolved.append(msg)

        # ① 제외(빼기) 조건 — 상태 조건 하나에만 정확히 적용할 수 있다
        # ⚠️ "그 사업 말고 전체 공고에서 …" 처럼 지시 표현에 붙은 '말고/제외'는 필드
        #    조건이 아니라 **범위**를 말한 것이다. 조건으로 읽으면 정상 질문이 되묻기로
        #    바뀐다. 표식 바로 앞이 문서를 가리키는 표현이면 조건으로 보지 않는다.
        exclusion = _EXCLUSION_RE.search(leftover)
        if exclusion and (_ANTI_EXCLUSION_RE.search(seg[:exclusion.start()])
                          or _ONLY_REQUEST_RE.match(seg[exclusion.end():])):
            # "하나도 빼지 말고 전부" — 빼라는 게 아니라 빼지 말라는 뜻이다
            # "X 말고 다른 건 필요 없어" — X를 빼라는 게 아니라 X만 달라는 뜻이다
            leftover = (leftover[:exclusion.start()]
                        + " " * (exclusion.end() - exclusion.start())
                        + leftover[exclusion.end():])
            exclusion = None
        if exclusion and _points_at_document(seg, exclusion.start()):
            # 범위를 말한 것이므로 조건 신호에서도 지운다(아래 ③에서 다시 잡히지 않게)
            leftover = (leftover[:exclusion.start()]
                        + " " * (exclusion.end() - exclusion.start())
                        + leftover[exclusion.end():])
            exclusion = None
        # ⚠️ 표식이 읽은 조건 바로 뒤(대상 명사만 사이에 두고)에 붙어야 그 조건에 걸린다.
        #    "평가 배점이 기재된 공고 중 **마감 지난 건** 빼고" 의 '빼고'는 마감 얘기지
        #    평가 배점 조건이 아니다 — 뒤집으면 정확히 반대 답이 나간다(적대적 점검).
        if exclusion and hits and not _exclusion_attaches(seg, hits, exclusion.start()):
            _unread(f"'제외' 조건이 어느 조건에 걸리는지 확정할 수 없습니다: \"{seg}\"")
            leftover = (leftover[:exclusion.start()]
                        + " " * (exclusion.end() - exclusion.start())
                        + leftover[exclusion.end():])
            exclusion = None
        if exclusion:
            status_hits = [cq for _s, _e, cq in hits if cq.kind == KIND_STATUS]
            if len(hits) == 1 and len(status_hits) == 1:
                status_hits[0].negated = True
            elif hits:
                _unread(f"'제외' 조건이 어느 조건에 걸리는지 확정할 수 없습니다: \"{seg}\"")
            else:
                _unread(f"'제외' 조건을 읽지 못했습니다: \"{seg}\"")

        # ② 읽지 못한 12필드 조건 — 그 키워드가 **가까이 있는 대상 명사를 수식**할 때만.
        #    "제출 서류 준비에 시간이 꽤 걸려서 지역제한이 없는 공고 알려줘" 처럼 배경에서
        #    스쳐 지나간 필드어까지 조건으로 세면 정상 질문이 되묻기가 된다(적대적 점검).
        # 조각이 사실상 필드 키워드뿐이면("예산," ) 나열된 조건이 잘린 것이다
        bare_field_segment = len(leftover.strip()) <= 2 and bool(mentions)
        for f in dict.fromkeys(f for _s, _e, f in mentions):
            # '값 + 명시'는 상태를 읽은 경우에도 값 제약이 남는다. 문구 전체를 소비해
            # 버리기 전에 검사하여 'AI로 명시'를 그냥 '기재됨'으로 축소하지 않는다.
            value = _states_field_value(seg, mentions, f, strong_only=f not in uncovered)
            # 이미 지원되는 뜻 판정(지역 제한 없음, 공동수급 허용 등)은 유지한다.
            if value and any(re.fullmatch(pat, value)
                             for _op, pat in _SEMANTIC_PREDICATES.get(f, [])):
                value = None
            if value:
                _unread(f"'{f}' 값 조건('{value}')은 이번 범위에서 지원하지 않습니다 "
                        f"— 상태(기재·미기재 등)와 예산 금액 비교만 지원합니다")
                continue
            if f not in uncovered:
                continue
            if not (_modifies_target(seg, mentions, f) or bare_field_segment):
                continue
            _unread(f"'{f}' 조건을 읽지 못했습니다: \"{seg}\"")

        # ③ 12필드 밖이지만 분명히 조건인 표현(수량 비교·지역/자격 제한·필터 표현)
        # 문서 이야기인지는 조각 **원문**으로 본다(남은 텍스트에는 필드어가 지워져 있다)
        signal = condition_signal(leftover, seg)
        if signal and not (exclusion and signal == "제외(빼기) 조건"):
            _unread(f"{signal}은 이번 범위에서 지원하지 않습니다 — 읽지 못했습니다: \"{seg}\"")

        # ④ 아무 신호도 없고 조건도 없으면 배경 설명
        if not hits and not uncovered and not signal and not exclusion:
            parse.background.append(seg)

    # 같은 (필드, 연산자, 값) 중복 제거 — 배경 문장이 같은 조건을 반복할 수 있다
    deduped: list[ConditionQuery] = []
    seen: set[tuple] = set()
    for cq in raw_conditions:
        if cq.signature() in seen:
            # 요청 조각에서 온 조건이면 그쪽 정보를 남긴다
            if cq.from_request:
                for i, kept in enumerate(deduped):
                    if kept.signature() == cq.signature():
                        deduped[i] = cq
            continue
        seen.add(cq.signature())
        deduped.append(cq)

    conditions, dropped, ambiguous = _resolve_same_field_conflicts(deduped)
    # 제외가 조건 하나에만 붙어야 범위가 분명하다. 조건이 둘 이상이면
    # "예산 5억 이상이고 사업기간이 기재된 공고는 제외하고" 가 어느 쪽(또는 둘 다)에
    # 걸리는지 확정할 수 없다 — 조용히 한쪽을 고르지 않고 되묻는다.
    if any(c.negated for c in conditions) and len(conditions) > 1:
        for c in conditions:
            c.negated = False
        ambiguous.append("'제외' 조건이 어느 조건에 걸리는지 확정할 수 없습니다 "
                         "— 조건이 둘 이상입니다")
    parse.conditions = conditions
    parse.notes.extend(dropped)
    for msg in ambiguous:
        if msg not in parse.unresolved:
            parse.unresolved.append(msg)
    return parse


def parse_conditions(question: str) -> tuple[list[ConditionQuery], bool]:
    """(인식된 조건 목록, 전부 인식됐는지) — 기존 호출 계약 유지."""
    parse = parse_selection(question)
    return parse.conditions, parse.fully_matched


def parse_condition(question: str) -> ConditionQuery | None:
    conditions, _ = parse_conditions(question)
    return conditions[0] if conditions else None


def names_a_specific_project(question: str) -> bool:
    """문서 검색과 같은 규칙으로 제목을 인식한다. 쉼표 유무로 범위를 바꾸지 않는다."""
    from doc_resolver import looks_like_specific_document
    return looks_like_specific_document(question)


def names_specific_document(question: str, identity: Any | None) -> str | None:
    """질문이 기관명·사업명으로 한 문서(또는 후보들)를 부르고 있는가. 방법 이름을 돌려준다.

    문서 특정은 doc_resolver 하나로 판단한다 — 라우터에 이름 목록을 복제하지 않는다.
    ⚠️ active_document_id 는 일부러 넘기지 않는다. 활성 문서가 있다는 이유만으로
       새 전역 질문이 그 문서에 갇히면 안 된다(지시 표현은 별도로 본다)."""
    if identity is None:
        return None
    from doc_resolver import resolve_document, RESOLVE_NONE, RESOLVE_ACTIVE
    res = resolve_document(question, identity)
    if res.method in (RESOLVE_NONE, RESOLVE_ACTIVE):
        return None
    return res.method if (res.document_id or res.candidates) else None


def is_selection_question(question: str, parse: SelectionParse | None = None,
                          identity: Any | None = None) -> bool:
    """"여러 문서를 골라 달라"는 질문인가 — 라우터와 실행부가 **같은** 함수를 쓴다.

    ⚠️ 단어 하나로 정하지 않는다.
      - "제출서류 목록이 기재된 사업들을 찾아줘"            → 선별(조건이 읽힘)
      - "이 사업의 제출서류 목록을 알려줘"                  → 특정 문서 값 조회(지시 표현)
      - "RFP-000038의 예산 알려줘"                          → 특정 문서 값 조회(문서 ID)
      - "한빛문화재단 ○○사업의 필수 제출 서류 목록을 알려줘" → 특정 문서 값 조회(사업명)
      - "이 사업과 별개로 전체 공고에서 예산 5억 이상"       → 선별(전체 범위가 지시 표현을 이김)

    [2026-09-03 라운드2] 예전에는 '목록' 같은 복수 표현 하나로 경로가 뒤집혔다.
    문서 특정 검사(기관명·사업명)를 복수 표현보다 **먼저** 하고, 명시적인 전체 범위
    표현이 있을 때만 지시 표현·이름 특정을 무시한다.

    identity 를 주면 기관명·사업명 특정까지 여기서 판단한다. 라우터는 identity 없이
    부르므로(잠정 판단), 최종 판단은 answer() 가 identity 를 넣어 다시 한 번 한다."""
    from doc_resolver import has_anaphora   # 지시 표현 목록은 doc_resolver가 정본

    parse = parse or parse_selection(question)
    if parse.targets_specific_document:
        return False                       # 문서 ID는 언제나 그 문서를 가리킨다
    # 조건을 읽었고 "공고들/목록/리스트"처럼 여러 문서를 원한다는 신호가 분명하면,
    # 조건 **값**에 기관명이 들어 있어도 선별이다("지역제한이 서울특별시로 걸린 공고들").
    explicit_multi = bool(parse.conditions and _MULTI_TARGET_RE.search(question))
    # 조건을 읽었는데 일부를 못 읽었다면 선별형 질문이다 — 한 문서 값 조회로 넘기면
    # 파서가 만든 되묻기가 통째로 사라지고 엉뚱한 문서를 확신에 차서 답하게 된다
    # ("대검찰청 말고 예산 5억 이상인 것만 추려줘" → 대검찰청 문서를 답함, 적대적 점검).
    partly_unread = bool(parse.conditions and parse.unresolved)
    if not parse.has_global_scope and not explicit_multi and not partly_unread:
        # 명시적인 전체 범위 표현이 없을 때만 "한 문서를 가리키는가"를 본다
        if has_anaphora(question):
            return False
        if names_specific_document(question, identity):
            return False
        # 이름을 등록부에서 **못 찾았더라도** 특정 사업을 부르는 모양이면 문서 질문이다.
        # 못 찾았다는 이유로 전체 공고 목록을 답하면 안 된다 — 추출 경로가 어느
        # 문서인지 되묻게 한다.
        if names_a_specific_project(question):
            return False
    if parse.conditions:
        return True
    # 조건을 못 읽었더라도 "여러 문서를 원한다"는 신호가 분명하면 선별로 보내
    # 무엇을 못 읽었는지 되묻는다(문서 특정 질문으로 잘못 보내지 않는다).
    if parse.unresolved and parse.has_plural_marker:
        return True
    # 조건도 미해석 조각도 없지만 "공고들을 골라 달라"는 요청이면, 어느 문서냐고
    # 묻는 대신 어떤 기준으로 고를지 되묻는 것이 맞다.
    return bool(parse.has_request_marker and _MULTI_TARGET_RE.search(question))


def validate_condition(cq: ConditionQuery) -> None:
    if cq.field not in _ALLOWED_FIELDS:
        raise InvalidQueryError(f"허용 안 된 필드: {cq.field}")
    if cq.operator not in _ALLOWED_OPERATORS:
        raise InvalidQueryError(f"허용 안 된 연산자: {cq.operator}")
    allowed_fields = _OPERATOR_FIELDS[cq.operator]
    if cq.field not in allowed_fields:
        raise InvalidQueryError(
            f"'{cq.field}' 필드에는 '{cq.operator}' 연산자를 쓸 수 없습니다 "
            f"(가능한 필드: {sorted(allowed_fields)})")
    if cq.negated and cq.kind != KIND_STATUS:
        # 상태는 문서마다 정확히 하나라서 여집합이 명확하다. 금액 비교나 뜻 판정은
        # '아닌 것'이 판단 불가와 섞여서 여집합이 정확하지 않다 — 지원하지 않는다.
        raise InvalidQueryError(
            f"'{cq.field}'의 '{cq.operator}' 조건에는 제외(부정)를 적용할 수 없습니다 "
            f"— 상태 조건에서만 지원합니다")


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
# 컨소시엄(공동수급) 요건 — 허용·금지·필수는 서로 다른 판정이다
# ---------------------------------------------------------------------------
# [2026-09-03 라운드2 재작성] 예전에는 _REQUIRE_RE/_ALLOW_RE/_FORBID_RE 를 값 **전체**에
# re.search 로 걸어서, 문장 어디엔가 금지 표현이 하나만 있어도 공동수급 전체가 불허로
# 뒤집히고 '반드시'가 어디에 있든 필수로 읽혔다. 실제 재현된 오독:
#   "공동수급은 허용하며 하도급은 불가합니다."            → 불허(X)  … 하도급이 옮겨붙음
#   "공동수급은 공동이행방식으로 허용하며 분담이행방식은 불허합니다." → 전체 불허(X)
#   "공동수급은 필수가 아닙니다. 단독 참여가 가능합니다."   → 필수(X)  … 부정을 못 봄
#   "공동수급체 구성원은 반드시 관련 면허를 보유해야 합니다." → 필수(X) … 구성원 의무일 뿐
#
# 이제 값을 **절(clause)** 로 나누고, 절마다 (무엇에 대한 말인지 = 주체) × (허용/금지/필수/
# 필수아님 = 양태)를 따로 읽어 합친다.
#
#   주체 CONSORTIUM  공동수급 그 자체            → 판정에 반영
#   주체 METHOD      공동이행/분담이행 등 방식만  → 허용이면 "가능"의 근거, 금지는 근거 아님
#   주체 SUBCONTRACT 하도급만                    → 공동수급 판정에 쓰지 않음
#   주체 MEMBER      구성원·지분율·협정서 조건    → 구성 자체의 허용/필수 근거가 아님
#
# ⚠️ 비대칭이 의도된 부분: "공동이행방식으로 허용"은 공동수급이 **가능하다**는 근거가 되지만,
#    "공동이행방식은 불허"는 공동수급 **전체가 금지**라는 근거가 되지 못한다(분담이행은 언급
#    없음). 마찬가지로 금지 문구는 결코 '필수'를 만들지 못한다.
CONSORTIUM_YES = True
CONSORTIUM_NO = False

# 공동수급 자체를 가리키는 표현(방식 한정어가 아닌 것)
_CONSORTIUM_TERM_RE = re.compile(r"공동수급|공동\s*수급|컨소시엄|공동도급|공동계약")
# 이행 '방식' 한정어
_CONSORTIUM_METHOD_RE = re.compile(r"공동이행|분담이행|주계약자관리")
_SUBCONTRACT_ONLY_RE = re.compile(r"하도급|재하도급")
# 공동수급 용어 바로 뒤(괄호 포함)에 방식이 붙어 범위를 좁히는 형태
_METHOD_SCOPED_RE = re.compile(
    r"(?:공동수급|공동\s*수급|컨소시엄|공동도급|공동계약)\s*(?:체|업체)?\s*"
    r"[（(\[「【]?\s*(?:공동이행|분담이행|주계약자관리)"
)
# 구성원·지분율·협정서 등 '구성 조건'을 말하는 절
_MEMBER_SCOPE_RE = re.compile(
    r"구성원|참여자|참여사|참가사|참가업체|수급체\s*참여|지분율|협정서|각\s*사(?:는|가|의)?"
)
# 그럼에도 공동수급 자체의 허용/금지를 분명히 말하는 형태(구성원 절이어도 이건 읽는다)
_ENTITY_MODAL_RE = re.compile(
    r"(?:공동수급|공동\s*수급|컨소시엄|공동도급|공동계약)\s*(?:체|업체)?\s*"
    r"(?:[（(\[「【][^）)\]」】]{0,20}[）)\]」】])?\s*"
    r"(?:을|를|은|는|이|가|:|：)?\s*(?:허용|불허|불가|금지|가능)"
)

# --- 양태 ---
# 부정된 필수("필수가 아니다")를 필수보다 **먼저** 본다
# ⚠️ "아닙니다"에는 '아니'가 들어 있지 않다(아+닙+니+다). 어미 변화를 모두 담는다.
# ⚠️ "아닙니다"에는 '아니'가 없고(아+닙), "아닌"에도 없다(아+닌). 어미 변화를 모두 담는다.
#    "의무 사항이 없습니다" 처럼 '없-'으로 부정하는 형태도 흔하다(적대적 점검에서 확인).
_NOT_SUFFIX = r"(?:아니|아닙|아님|아닌|아녀|아녜|않|없)"
_NEG_REQUIRE_RE = re.compile(
    rf"필수\s*(?:사항)?\s*(?:가|이|는|은|도)?\s*{_NOT_SUFFIX}"
    rf"|의무\s*(?:사항)?\s*(?:가|이|는|은|도)?\s*{_NOT_SUFFIX}"
    rf"|반드시[^.]{{0,20}}?(?:것은\s*)?{_NOT_SUFFIX}"
    rf"|요구\s*(?:하지\s*)?않|요구\s*안|강제\s*(?:하지\s*)?않"
    rf"|(?:강제|의무)\s*사항\s*(?:은|는|이|가)?\s*{_NOT_SUFFIX}"
)
# ⚠️ 필수 판정은 **아주 좁게** 잡는다. 공식 값에는 "공동수급체는 5개 이하로 구성하여야 하며"
#    처럼 구성 '조건'을 말하는 문장이 많아서, 맨 '구성하여야'만 보면 전부 필수가 돼 버린다.
#    단독 참여를 명시적으로 막았거나, 공동수급 자체에 필수·의무를 붙인 경우만 필수로 본다.
# 공동수급 자체에 필수·의무를 붙인 형태(절 단위로 본다)
_ENTITY_REQUIRE_RE = re.compile(
    r"(?:공동수급|공동\s*수급|컨소시엄|공동도급|공동계약)\s*(?:체|업체)?\s*"
    r"(?:을|를|은|는|이|가|으로|로)?\s*[^.]{0,10}?(?:필수|의무(?:적)?)"
    r"|반드시\s*(?:공동수급|공동\s*수급|컨소시엄|공동도급|공동계약)"
    r"|(?:공동수급|컨소시엄)\s*(?:체)?\s*(?:로만|으로만)"
)
# 단독 참여 가능/불가는 절이 아니라 값 전체의 성질이다 —
# "공동수급체를 구성하여야 하며 단독 입찰은 불가함"에서 두 번째 절에는 공동수급 용어가
# 없지만, 그 문장이야말로 공동수급을 필수로 만드는 근거다.
_SOLO_FORBIDDEN_RE = re.compile(
    r"단독\s*(?:입찰|참여|참가|응찰|수행)?\s*(?:은|는|이|가|으로|로)?\s*"
    r"(?:불가|불허|금지|안\s*됨|안\s*되|할\s*수\s*없|허용하지\s*않)"
)
_SOLO_OK_RE = re.compile(
    r"단독\s*(?:또는|이나|혹은|,|/)\s*(?:공동|컨소시엄)"
    r"|단독\s*(?:으로|이나|도|만)*\s*(?:입찰|참여|참가|응찰|수행)?\s*"
    r"(?:이|가|은|는|도|만)?\s*(?:가능|할\s*수\s*있|허용)"
)
# ⚠️ '구성하여 입찰'을 허용 표식으로 쓰면 "컨소시엄을 구성하여 입찰하는 경우 참여를 제한함"
#    이 허용으로 뒤집힌다(공식 RFP-000030, 적대적 점검에서 확인). 허용을 뜻하는 낱말만 본다.
_ALLOW_RE = re.compile(r"허용|가능|할\s*수\s*있|참가할\s*수\s*있")
# ⚠️ '제외'는 목적어를 반드시 확인한다. 맨 '제외한'을 금지로 읽으면
#    "일부 항목을 제외한 나머지 서류는 공동수급 허용" 이 금지로 뒤집힌다(적대적 점검에서 확인).
# ⚠️ '불가'는 '불가피한'을 잘라내야 한다 — 공식 RFP-000027/000088의
#    "공동수급체를 구성하지 못하는 **불가피한** 사정이 있는 경우"가 금지로 읽혔다.
# ⚠️ '제외'는 목적어가 공동수급일 때만 금지다.
# ⚠️ '참여/참가를 제한'은 금지다(RFP-000030). 다만 '구성원 5개 이하로 제한' 같은 구성 조건은 아니다.
# ⚠️ 긍정 낱말(허용·가능)에 붙은 **부정 어미**를 먼저 잡아야 한다. 예전에는
#    '허용하지 않'만 있어서 "허용되지 않습니다"·"가능하지 않습니다"·"허용하지 아니합니다"·
#    "허용할 수 없습니다" 를 놓쳤고, 그 안의 '허용/가능'이 허용으로 읽혔다(실제 재현).
#    어간(허용·가능·승인·인정) + 어미(하/되/할/될) + (지|수) + (않|아니|없) 을 한 덩어리로 본다.
_NEGATED_PERMISSION = (
    r"(?:허용|가능|승인|인정)\s*(?:하|되|할|될)?\s*(?:지|수)?\s*(?:않|아니|없)"
)
_FORBID_RE = re.compile(
    r"불가(?!피)|불허|금지|배제"
    rf"|{_NEGATED_PERMISSION}"
    r"|(?:참여|참가)\s*(?:를|을)?\s*제한"
    r"|(?:공동수급|공동\s*수급|컨소시엄|공동도급|공동계약)\s*(?:체|업체)?\s*(?:을|를)?\s*제외"
)
# 공동수급 '구성' 자체를 의무로 지우는 문구. 구성원 수·지분율 같은 구성 조건과 구분해야 한다.
_FORMATION_DUTY_RE = re.compile(
    r"(?:공동수급|공동\s*수급|컨소시엄|공동도급|공동계약)\s*(?:체|업체)?[^.]{0,16}?"
    r"(?:구성|참가|참여)\s*(?:하여|해)?\s*(?:하여야|여야|어야|해야|하여야만)"
)
# 구성 '조건'(개수·지분율·한도)을 말하는 절 — 이런 절의 '하여야'는 의무가 아니다
_COMPOSITION_LIMIT_RE = re.compile(r"\d+\s*(?:개|인|사|%)|지분율|이하로|이내로|이하\b|이내\b")
# 조건부 의무("…하려는 경우 …구성하여 참여해야") — 무조건 필수가 아니다
_CONDITIONAL_RE = re.compile(r"경우|하려는|할\s*때|하는\s*때|시\s*에는|단,")

# 절 구분 — 문장부호와 한국어 연결어미. 항목 번호("가.", "6)")는 따로 떼어낸다.
# ⚠️ 연결어미 뒤에는 쉼표가 붙는 경우가 훨씬 많다("…허용하며, 공동수급업체 구성원은 …").
#    쉼표를 흘리면 허용을 말한 앞절과 구성원 조건을 말한 뒷절이 한 덩어리가 되어,
#    구성원 조건 때문에 허용 문구까지 통째로 버려진다(공식 값 10건에서 실제로 발생).
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.;])\s+|\n+"
    r"|(?<=하며),?\s+|(?<=으며),?\s+|(?<=이며),?\s+"
    r"|(?<=하고),?\s+|(?<=하되),?\s+|(?<=지만),?\s+"
    r"|(?<=이나),?\s+|(?<=하나),?\s+"
)
_ENUM_PREFIX_ONLY_RE = re.compile(r"^\s*(?:[가-힣]\s*[.)]|\(?\d{1,2}\s*[.)]|[①-⑳])\s*$")

CONSORTIUM_UNKNOWN_REASON_NO_ROW = "추출표에 '컨소시엄 요건' 행이 없습니다"

# 절 판정 결과
_SUBJ_CONSORTIUM = "consortium"
_SUBJ_METHOD = "method"
_SUBCONTRACT = "subcontract"
_SUBJ_MEMBER = "member"
_SUBJ_OTHER = "other"


@dataclass
class ConsortiumVerdict:
    """공동수급에 대한 두 가지 독립 판단. None은 '이 문서로는 확정 불가'."""

    required: bool | None = None   # 공동수급을 반드시 구성해야 하는가
    allowed: bool | None = None    # 공동수급을 구성할 수 있는가
    reason: str = ""
    evidence: str = ""
    clauses: list[dict] = field(default_factory=list)   # 절별 판정(진단용)


def split_clauses(text: str) -> list[str]:
    """값을 절 단위로 나눈다. 항목 번호만 남는 조각은 버린다."""
    out: list[str] = []
    for piece in _CLAUSE_SPLIT_RE.split(text or ""):
        piece = (piece or "").strip()
        if not piece or _ENUM_PREFIX_ONLY_RE.match(piece):
            continue
        out.append(piece)
    return out or ([text.strip()] if (text or "").strip() else [])


# 절 안에서 주체가 될 수 있는 낱말들(위치까지 알아야 한다)
_TERM_SCAN = [
    (_SUBJ_CONSORTIUM, re.compile(r"공동수급|공동\s*수급|컨소시엄|공동도급|공동계약")),
    (_SUBJ_METHOD, re.compile(r"공동이행|분담이행|주계약자관리")),
    (_SUBCONTRACT, re.compile(r"재하도급|하도급|하수급인")),
]
# 두 낱말 사이가 '접속'뿐이면 둘 다 같은 서술어의 주체다("공동수급 및 하도급 불가").
# 반대로 " 여부 : " 처럼 다른 말이 끼면 가까운 쪽만 주체다("공동수급 여부 : 하도급 불허").
_COORDINATOR_ONLY_RE = re.compile(r"^[\s,·、/및와과()（）\[\]「」]*$")
_PARENTHETICAL_RE = re.compile(r"[（(\[「【][^）)\]」】]*[）)\]」】]")


def _terms_in(clause: str) -> list[tuple[int, int, str]]:
    """절 안 주체 낱말의 (시작, 끝, 종류). 겹치면 긴 쪽을 남긴다."""
    found: list[tuple[int, int, str]] = []
    for kind, rx in _TERM_SCAN:
        for m in rx.finditer(clause):
            if not any(m.start() < e and m.end() > s_ for s_, e, _ in found):
                found.append((m.start(), m.end(), kind))
    return sorted(found)


def _subjects_for(clause: str, terms: list[tuple[int, int, str]],
                  modality_start: int) -> set[str]:
    """양태 표현 하나의 주체 집합.

    ⚠️ 예전에는 "절 안에 공동수급이라는 낱말이 있으면 그 절의 모든 양태가 공동수급에 대한
       것"이라고 봤다. 그래서 공식 RFP-000057 "하도급/공동수급 여부 : 하도급 불허 /
       공동수급 허용" 이 통째로 '불허'가 됐다(적대적 점검에서 확인).
       이제 양태 표현에서 왼쪽으로 가장 가까운 낱말을 주체로 잡고, 그 앞이 접속뿐이면
       나란히 놓인 낱말들까지 같은 주체로 묶는다."""
    before = [t for t in terms if t[1] <= modality_start]
    if not before:
        after = [t for t in terms if t[0] >= modality_start]
        return {after[0][2]} if after else set()
    group = [before[-1]]
    for cand in reversed(before[:-1]):
        # 괄호 안 설명은 접속 판단에서 빼고 본다
        # ("공동계약(공동 및 분담 이행방식) 및 하도급 불가" → 둘 다 금지 대상)
        gap = _PARENTHETICAL_RE.sub("", clause[cand[1]:group[-1][0]])
        if _COORDINATOR_ONLY_RE.match(gap):
            group.append(cand)
        else:
            break
    return {t[2] for t in group}


def _allow_is_about_solo(clause: str, allow_pos: int) -> bool:
    """이 위치의 '가능/허용'이 공동수급이 아니라 **단독 참여**를 가리키는가.

    "공동수급은 의무가 아니며 단독으로도 참여 가능합니다" 에서 '가능'의 주어는 단독
    참여다. 이를 공동수급 허용의 근거로 삼으면 안 된다.
    ⚠️ 표현 하나만 보고 판단한다 — 절 안에 단독 문구가 있다는 이유로 같은 절의 다른
       '공동수급 허용'까지 무시하면 안 된다(적대적 점검에서 확인)."""
    for solo in _SOLO_OK_RE.finditer(clause):
        if solo.start() <= allow_pos < solo.end():
            return True
    return False


def _clause_modality(clause: str) -> set[str]:
    """이 절이 말하는 양태(절 전체 기준). 부정된 필수를 필수보다 먼저 본다.

    허용/금지는 주체를 따로 풀어야 하므로 _modality_hits() 를 쓴다. 이 함수는
    절 전체에 걸리는 양태(필수·필수아님·단독 가능/불가)만 돌려준다."""
    mods: set[str] = set()
    if _NEG_REQUIRE_RE.search(clause):
        mods.add("neg_require")
    elif _ENTITY_REQUIRE_RE.search(clause):
        mods.add("require")
    elif (_FORMATION_DUTY_RE.search(clause)
          and not _COMPOSITION_LIMIT_RE.search(clause)
          and not _CONDITIONAL_RE.search(clause)
          and not _MEMBER_SCOPE_RE.search(clause)):
        # "공동수급체를 구성하여 입찰에 참가하여야 한다" — 구성 자체가 의무다.
        # 단, 구성원 수·지분율 같은 구성 조건이거나 조건부("…하려는 경우")면 아니다.
        mods.add("require")
    if _SOLO_OK_RE.search(clause):
        mods.add("solo_ok")
    if _SOLO_FORBIDDEN_RE.search(clause):
        mods.add("solo_forbidden")
    return mods


def _modality_hits(clause: str) -> list[tuple[int, str]]:
    """절 안의 허용/금지 표현 위치. "허용하지 않음"은 금지 한 번으로만 센다."""
    hits: list[tuple[int, str]] = []
    forbid_spans: list[tuple[int, int]] = []
    for m in _FORBID_RE.finditer(clause):
        hits.append((m.start(), "forbid"))
        forbid_spans.append((m.start(), m.end()))
    for m in _ALLOW_RE.finditer(clause):
        # '허용하지 않음'의 '허용'은 이미 금지로 셌다 — 겹치면 버린다
        if any(fs - 6 <= m.start() <= fe for fs, fe in forbid_spans):
            continue
        hits.append((m.start(), "allow"))
    return sorted(hits)


def classify_consortium(row: dict | None) -> ConsortiumVerdict:
    """'컨소시엄 요건' 값의 뜻을 절 단위로 판정한다. 확정 못 하면 이유를 남긴다."""
    if row is None:
        return ConsortiumVerdict(reason=CONSORTIUM_UNKNOWN_REASON_NO_ROW)
    status = row.get("status")
    if status == "conflict":
        return ConsortiumVerdict(
            reason="'컨소시엄 요건' 원문 내용이 서로 충돌합니다 — 한쪽을 고르지 않습니다",
            evidence=str(row.get("answer_raw") or "")[:200])
    if status != "value_present":
        return ConsortiumVerdict(
            reason=f"'컨소시엄 요건' 상태가 {status}라 공동수급 조건을 단정할 수 없습니다 "
                   f"(미기재를 '단독 참여 가능'으로 읽지 않습니다)")
    text = " ".join(
        str(row.get(k) or "") for k in ("answer_raw", "answer_normalized")
    ).strip()
    if not text:
        return ConsortiumVerdict(reason="'컨소시엄 요건' 값이 비어 있어 판정할 수 없습니다")
    # answer_raw 와 answer_normalized 가 같은 문장을 두 번 담고 있으면 한 번만 본다
    raw = str(row.get("answer_raw") or "").strip()
    norm = str(row.get("answer_normalized") or "").strip()
    text = raw or norm
    if raw and norm and norm not in raw and raw not in norm:
        text = raw + "\n" + norm

    # --- 절별 판정 ---
    entity_allow = entity_forbid = entity_require = entity_neg_require = False
    method_allow = method_forbid_only = False
    solo_ok = solo_forbidden = False
    saw_any_consortium_clause = False
    detail: list[dict] = []

    for clause in split_clauses(text):
        terms = _terms_in(clause)
        mods = _clause_modality(clause)
        kinds = {k for _s, _e, k in terms}
        if kinds & {_SUBJ_CONSORTIUM, _SUBJ_METHOD}:
            saw_any_consortium_clause = True
        # 단독 참여 가능/불가는 어느 절에 있든 값 전체의 성질로 읽는다
        if "solo_ok" in mods:
            solo_ok = True
        if "solo_forbidden" in mods:
            solo_forbidden = True
        member_clause = bool(_MEMBER_SCOPE_RE.search(clause))
        if _SUBJ_CONSORTIUM in kinds and "require" in mods and not member_clause:
            entity_require = True
        if _SUBJ_CONSORTIUM in kinds and "neg_require" in mods:
            entity_neg_require = True

        # --- 허용/금지는 표현마다 주체를 푼다 ---
        mod_detail = []
        allow_spans = [pos for pos, kind in _modality_hits(clause) if kind == "allow"]
        for pos, kind in _modality_hits(clause):
            subjects = _subjects_for(clause, terms, pos)
            mod_detail.append({"at": clause[max(0, pos - 12):pos + 8], "kind": kind,
                               "subjects": sorted(subjects)})
            if _SUBJ_CONSORTIUM in subjects:
                scoped = bool(_METHOD_SCOPED_RE.search(clause))
                if kind == "allow":
                    if not _allow_is_about_solo(clause, pos):
                        entity_allow = True
                elif scoped:
                    # "공동수급(분담이행방식)을 허용하지 않음" — 전체 금지의 근거가 아니다
                    method_forbid_only = True
                else:
                    entity_forbid = True
            elif _SUBJ_METHOD in subjects:
                if kind == "allow":
                    method_allow = True   # 방식이 허용되면 공동수급 자체는 가능하다
                else:
                    # 방식만의 금지는 공동수급 전체의 근거가 되지 않는다(의도된 비대칭)
                    method_forbid_only = True
            # 하도급·그 밖의 주체는 공동수급 판정에 쓰지 않는다
        detail.append({"clause": clause[:160], "terms": sorted(kinds),
                       "modality": sorted(mods), "allow_forbid": mod_detail})

    reasons: list[str] = []

    # 단독 참여를 명시적으로 막았다면(공동수급을 말하는 값 안에서) 그것이 필수의 근거다
    if solo_forbidden and _CONSORTIUM_TERM_RE.search(text):
        entity_require = True

    # --- 필수 여부 ---
    if entity_require and (entity_neg_require or solo_ok):
        required = None
        reasons.append("공동수급이 필수라는 문구와 필수가 아니라는 문구가 함께 있어 단정하지 않습니다")
    elif entity_require:
        required = True
        reasons.append("공동수급을 반드시 구성해야 한다고 명시돼 있습니다(단독 참여 불가 포함)")
    elif entity_neg_require:
        required = False
        reasons.append("공동수급이 필수가 아니라고 명시돼 있습니다")
    elif solo_ok:
        required = False
        reasons.append("단독 참여가 가능하다고 명시돼 있어 공동수급이 필수는 아닙니다")
    elif entity_allow or entity_forbid or method_allow:
        # 허용·금지만 말하고 필수를 요구하는 문구가 없다 → 필수 아님
        # (특정 이행방식의 허용도 "구성할 수 있다"는 뜻이므로 필수는 아니다)
        required = False
        reasons.append("공동수급을 필수로 요구하는 문구가 없습니다")
    else:
        required = None

    # --- 허용 여부 ---
    if entity_allow and entity_forbid:
        allowed = None
        reasons.append("같은 값 안에 허용과 금지가 함께 있어 허용 여부를 단정하지 않습니다")
    elif entity_allow or method_allow:
        allowed = True
        if method_allow and not entity_allow:
            reasons.append("특정 이행방식의 공동수급이 허용된다고 명시돼 있습니다")
        else:
            reasons.append("공동수급을 허용한다고 명시돼 있습니다")
    elif entity_require:
        # 반드시 구성해야 한다면 구성 자체는 당연히 가능하다
        allowed = True
        reasons.append("공동수급 구성이 요구되므로 허용됩니다")
    elif entity_forbid:
        allowed = False
        reasons.append("공동수급을 불허한다고 명시돼 있습니다")
    else:
        allowed = None
        if method_forbid_only:
            reasons.append("특정 이행방식의 금지만 명시돼 있어 공동수급 전체의 허용 여부는 "
                           "단정할 수 없습니다")
        elif not saw_any_consortium_clause:
            if _SUBCONTRACT_ONLY_RE.search(text):
                reasons.append("하도급에 대한 문구만 있습니다 — 공동수급 조건으로 읽지 않습니다")
            else:
                reasons.append("공동수급을 가리키는 문구를 찾지 못했습니다")
        elif _CONSORTIUM_METHOD_RE.search(text) and not _CONSORTIUM_TERM_RE.search(text):
            reasons.append("특정 이행방식에 대한 문구만 있어 공동수급 전체의 허용 여부는 "
                           "단정할 수 없습니다")
        else:
            reasons.append("공동수급의 허용·금지를 문구로 확정하지 못했습니다")

    # 방식만 금지된 경우(허용 문구 없음)에는 '필수 아님'도 단정하지 않는다.
    # 금지 문구가 요구를 만들지는 못하지만, 그 문서가 단독 참여를 허락한다는 뜻도 아니다.
    if (allowed is None and required is False and not entity_neg_require and not solo_ok
            and not entity_allow and not entity_forbid and not entity_require):
        required = None
        reasons.append("특정 이행방식의 금지만으로는 단독 참여 가능 여부를 단정하지 않습니다")

    return ConsortiumVerdict(required=required, allowed=allowed,
                             reason=" / ".join(reasons), evidence=text[:200],
                             clauses=detail)


# ---------------------------------------------------------------------------
# 조건 실행
# ---------------------------------------------------------------------------

MATCH = "match"
NO_MATCH = "no_match"
UNDETERMINED = "undetermined"


@dataclass
class QueryResult:
    document_id: str
    state: str
    value: Any
    warning: str | None = None
    row: dict | None = None


def evaluate_condition(row: dict | None, cq: ConditionQuery) -> tuple[str, str]:
    """문서 한 건이 조건 하나를 만족하는지. (판정, 사유)

    판정은 세 갈래다 — 만족 / 불만족 / **판단 불가**.
    판단 불가를 불만족으로 뭉개면 "조건에 맞는 사업이 하나도 없다"는 잘못된
    단정이 나온다."""
    if row is None:
        return UNDETERMINED, f"추출표에 '{cq.field}' 행이 없습니다"
    status = row.get("status")

    if cq.kind == KIND_STATUS:
        want = STATUS_OPERATOR_TARGETS[cq.operator]
        hit = status in want
        # 제외 조건 — 상태는 문서마다 정확히 하나라서 여집합이 명확하다
        if cq.negated:
            hit = not hit
        return (MATCH, "") if hit else (NO_MATCH, "")

    if cq.field == "지역제한" and cq.operator in (OP_REGION_NONE, OP_REGION_SOME):
        verdict, reason = classify_region_restriction(row)
        want = REGION_NONE if cq.operator == OP_REGION_NONE else REGION_RESTRICTED
        if verdict == want:
            return MATCH, ""
        if verdict == REGION_UNKNOWN:
            return UNDETERMINED, reason
        return NO_MATCH, ""

    if cq.field == "컨소시엄 요건" and cq.operator in _CONSORTIUM_OPERATORS:
        v = classify_consortium(row)
        if cq.operator == OP_CONSORTIUM_REQUIRED:
            got = v.required
            want = True
        elif cq.operator == OP_CONSORTIUM_NOT_REQUIRED:
            got = v.required
            want = False
        elif cq.operator == OP_CONSORTIUM_ALLOWED:
            got = v.allowed
            want = True
        else:
            got = v.allowed
            want = False
        if got is None:
            return UNDETERMINED, v.reason
        return (MATCH, "") if got == want else (NO_MATCH, "")

    # 금액 비교
    if status != "value_present":
        return UNDETERMINED, (
            f"'{cq.field}' 상태가 {status}라 금액을 비교할 수 없습니다")
    raw = row.get("answer_raw") or row.get("answer_normalized")
    if _parse_korean_amount(raw) is None:
        return UNDETERMINED, (
            f"'{cq.field}' 값을 숫자로 읽지 못해 조건 판정에서 제외됨 — 직접 확인 필요")
    return (MATCH, "") if _evaluate(raw, cq.operator, cq.value) else (NO_MATCH, "")


@dataclass
class SelectionResult:
    """선별 실행 결과 — 확정 결과와 판단 불가를 절대 섞지 않는다."""

    document_ids: list[str] = field(default_factory=list)
    undetermined: list[str] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    rows: dict[str, dict[str, dict]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    scanned_document_ids: list[str] = field(default_factory=list)
    excluded_by_scope: list[str] = field(default_factory=list)


def _index_table(table: list[dict]) -> dict[str, dict[str, dict]]:
    """(문서 → 필드 → 행). 추출표의 행 단위 active=false는 여기서 제외한다."""
    index: dict[str, dict[str, dict]] = {}
    for row in table:
        if not _row_active(row):
            continue
        index.setdefault(row["document_id"], {})[row["field_name"]] = row
    return index


def run_selection_query(
    table: list[dict],
    conditions: list[ConditionQuery],
    allowed_document_ids: set[str] | None = None,
) -> SelectionResult:
    """조건 전체를 **모든 후보 문서**에 적용한다(중간에 자르지 않는다).

    처리 순서: 전체 후보 → 공식 등록부 범위(allowed_document_ids) → 질문의 전체 조건.
    마감일 정책은 호출측(answer_pipeline)이 이 결과 뒤에 적용한다."""
    for cq in conditions:
        validate_condition(cq)

    index = _index_table(table)
    result = SelectionResult()
    if not conditions:
        return result

    for doc_id in sorted(index):
        if allowed_document_ids is not None and doc_id not in allowed_document_ids:
            result.excluded_by_scope.append(doc_id)
            continue
        result.scanned_document_ids.append(doc_id)
        fields = index[doc_id]
        verdicts = []
        reasons: list[str] = []
        used_rows: dict[str, dict] = {}
        for cq in conditions:
            row = fields.get(cq.field)
            verdict, reason = evaluate_condition(row, cq)
            verdicts.append(verdict)
            if row is not None:
                used_rows[cq.field] = row
            if verdict == UNDETERMINED:
                reasons.append(f"{doc_id}: {reason} — 확인 필요" if reason
                               else f"{doc_id}: '{cq.field}' 판단 불가 — 확인 필요")
        if NO_MATCH in verdicts:
            continue
        result.rows[doc_id] = used_rows
        if UNDETERMINED in verdicts:
            result.undetermined.append(doc_id)
            result.reasons[doc_id] = reasons
            result.warnings.extend(reasons)
        else:
            result.document_ids.append(doc_id)

    # 중복 제거 + 재현 가능한 순서
    result.document_ids = sorted(dict.fromkeys(result.document_ids))
    result.undetermined = sorted(dict.fromkeys(result.undetermined))
    seen: set[str] = set()
    result.warnings = [w for w in result.warnings
                       if not (w in seen or seen.add(w))]
    return result


def run_condition_query(
    table: list[dict], cq: ConditionQuery, max_results: int | None = None
) -> tuple[list[QueryResult], int]:
    """조건 한 개 실행. max_results=None이면 **자르지 않는다**(기본값).

    ⚠️ 예전 기본값은 20이었다. 잘린 목록에 마감 필터가 걸려 결과가 조용히
    빠지던 원인이라 기본을 '전체'로 바꿨다."""
    sel = run_selection_query(table, [cq])
    out: list[QueryResult] = []
    for doc_id in sel.document_ids:
        row = sel.rows[doc_id].get(cq.field)
        out.append(QueryResult(doc_id, (row or {}).get("status", ""),
                               (row or {}).get("answer_normalized"), row=row))
    for doc_id in sel.undetermined:
        row = sel.rows[doc_id].get(cq.field)
        out.append(QueryResult(doc_id, (row or {}).get("status", ""), None,
                               warning="; ".join(sel.reasons.get(doc_id, [])) or None,
                               row=row))
    out.sort(key=lambda r: r.document_id)
    total = len(out)
    return (out if max_results is None else out[:max_results]), total


def run_conditions_query(
    table: list[dict], conditions: list[ConditionQuery],
    max_results: int | None = None,
    allowed_document_ids: set[str] | None = None,
) -> tuple[list[QueryResult], int, list[str]]:
    """AND 복합조건 — 확정 결과만 results로, 판단 불가 사유는 warnings로 돌려준다.

    max_results=None(기본)이면 자르지 않는다."""
    if not conditions:
        return [], 0, []
    sel = run_selection_query(table, conditions,
                              allowed_document_ids=allowed_document_ids)
    first = conditions[0]
    matched = [
        QueryResult(doc_id,
                    (sel.rows[doc_id].get(first.field) or {}).get("status", ""),
                    (sel.rows[doc_id].get(first.field) or {}).get("answer_normalized"),
                    row=sel.rows[doc_id].get(first.field))
        for doc_id in sel.document_ids
    ]
    total = len(matched)
    return (matched if max_results is None else matched[:max_results]), total, sel.warnings


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
