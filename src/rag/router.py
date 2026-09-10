"""
F-0 — 질문 유형 경로 분기 (4-10-1 확정).
규칙 기반으로 시작, 분기 실패 시 QA형으로 폴백(4-10-1).
검색 불필요 질문(인사·시스템 사용법) 필터링도 baseline 포함(4-10-0 ①).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

from table_query import (
    detect_fields, detect_deadline_question,
    is_selection_question, parse_selection, needs_explanation,
)
from doc_resolver import detect_document_ids, looks_like_specific_document

TaskType = Literal["select", "extract", "qa", "compare", "no_search_needed"]

# no_search_needed 안에서 다시 갈리는 두 갈래(4-10-0 ①)
NO_SEARCH_SYSTEM_HELP = "system_help"
NO_SEARCH_GREETING = "greeting"


@dataclass
class RouteResult:
    task_type: TaskType
    matched_rule: str | None  # 어느 규칙에 매칭됐는지 (분기 결과 기록용, 4-10-1)
    is_fallback: bool = False  # 규칙이 못 잡아 기본 경로(QA)로 갔는지
    no_search_kind: str | None = None  # no_search_needed일 때 system_help / greeting


# 규칙 기반 패턴 — 표현 다양성 실측(1-2·1-18) 도착 시 갱신 필요
# ⚠️ 버그 수정(리뷰 반영): 기존엔 "어떻게 (써|사용|쓰나요)"만 봐서
# "이 사업의 예산을 어떻게 사용하는 건가요?"까지 시스템 사용법 질문으로
# 오판했다(실제 재현 확인). "시스템/프로그램/이거"처럼 도구 자체를 가리키는
# 표현이 같이 있을 때만 사용법 질문으로 본다.
_SYSTEM_HELP_PATTERNS = [
    r"(?:이\s*)?(?:시스템|프로그램|서비스|플랫폼|챗봇|봇|툴|도구|이거|이걸|여기)"
    r"(?:은|는|를|을|이|가)?\s*(?:어떻게|어떤\s*식으로)\s*(?:써|쓰|사용|이용)",
    r"(?:시스템|프로그램|서비스|플랫폼|챗봇|봇|툴|도구)\s*사용법",
    r"^사용법",
    r"^도움말",
    r"(?:뭘|무엇을|어떤\s*(?:질문|걸))\s*(?:물어|물을|할\s*수)",
    r"(?:사용|이용)\s*방법(?:을|이|은)?\s*(?:알려|설명|안내)",
]
_GREETING_PATTERNS = [
    # 인사 뒤에 실제 질문이 이어지면 인사 전용 답으로 질문을 지우지 않는다.
    r"^\s*(?:안녕(?:하세요|하십니까)?|반가워(?:요)?|고마워(?:요)?|감사(?:합니다|해요)?|하이|헬로)[\s,.!?~^^]*$",
]
_NO_SEARCH_PATTERNS = _SYSTEM_HELP_PATTERNS + _GREETING_PATTERNS
# ⚠️ 2026-09-03 개정 — 선별형 판단을 table_query.is_selection_question 하나로 모았다.
# 예전엔 라우터가 자체 정규식("목록|추천|리스트"·금액 표현·지시 표현 제외 목록)으로
# 판단하고, 조건 파서는 또 다른 규칙으로 조건을 읽어서 둘이 어긋났다. 그래서
# "사업분야가 명시된 공고 알려줘"처럼 목록을 원하는 질문이 뒤의 넓은 필드 키워드
# 검사에 걸려 extract(특정 문서 값 조회)로 새어나갔다.
#
# 지금은 라우터와 조건 파서가 **같은 파싱 결과**를 본다.
#   - 질문이 특정 문서를 가리키면(문서 ID·지시 표현) 선별형이 아니다
#   - 조건을 하나라도 읽었으면 선별형
#   - 조건을 못 읽었어도 "여러 문서를 원한다"는 신호가 분명하면 선별형으로 보내
#     무엇을 못 읽었는지 되묻는다(문서 특정 질문으로 잘못 보내지 않는다)
# ⚠️ 2026-08-31 정정: 예전엔 "마감일"·"발주기관"이 여기 섞여 있었는데, 4-9-8 v3
# 확정으로 이 둘은 최종 12필드에서 빠졌다.
# ⚠️ 2026-09-03 정정: 예전엔 라우터가 FIELD_KEYWORDS를 직접 정규식으로 이어 붙여
#    "필드어가 글자 그대로 있는가"만 봤다. 그래서 table_query가 필드를 **뜻으로**
#    알아보게 되면(예: "제안서를 어떤 방식으로 제출" → 제출 방식) 라우터만 못 보고
#    QA로 새거나, 반대로 라우터만 추출형으로 보내고 실행부가 필드를 못 찾는
#    어긋남이 생긴다. 이제 라우터와 실행부가 **같은 함수**(table_query.detect_fields)를
#    쓴다 — 규칙이 두 벌로 갈라지지 않게 한다.
# ⚠️ 정정: 예전엔 필드 키워드 뒤에 정해진 질문 어미(얼마/언제/어디/뭐/무엇/
# 알려줘/확인/어떻게/어때/알고싶...)가 붙어야만 extract로 잡았다. 이 방식은
# "필요해?"/"있나요?"처럼 목록에 없는 어미가 나올 때마다 계속 단어를
# 추가해야 하는 안 끝나는 목록이었다(실제로 EXT-002/004에서 재현됨).
# 한국어로 "필요한지/가능한지/되는지"를 묻는 방식은 끝이 없어서, 질문
# 어미가 아니라 "12필드(+마감일) 키워드가 질문에 있는가"만 본다 — 이
# 목록은 이미 확정된 닫힌 집합이라 더 늘어날 일이 없다.
# 트레이드오프: 필드어가 있는 진짜 QA성 질문("예산이 부족해 취소됐다는데
# 왜 그런거야")도 일단 extract를 거치지만, extract 내부의 needs_explanation
# 감지가 검색+생성 경로로 넘겨서 최종 결과는 틀리지 않는다(에러도 없음).


def extract_signal(question: str) -> str | None:
    """추출형 신호 — 실행부와 **같은** 필드 감지 결과를 쓴다.

    최종 12필드(table_query.detect_fields) 또는 12필드 밖 마감일이 잡히면
    그 필드 이름을 규칙 이름으로 돌려준다. 아무것도 없으면 None.
    """
    fields = detect_fields(question)
    if detect_deadline_question(question):
        fields = fields + ["입찰 참여 마감일"]
    if not fields:
        return None
    return "field:" + ",".join(fields)


_COMPARE_PATTERNS = [
    r"(비교|차이|어느 쪽|둘 중)",
]

# ---------------------------------------------------------------------------
# 개념 차이 설명 vs 문서 비교 (2026-09-04 §10)
# ---------------------------------------------------------------------------
# "공동수급과 하도급의 차이가 뭐야?" 는 **제도 용어 두 개**의 뜻을 묻는 질문이고,
# "RFP-000001과 RFP-000002의 예산을 비교해줘" 는 **문서 두 건**의 같은 필드를 묻는
# 질문이다. 예전에는 "비교/차이/둘 중" 이 있으면 무조건 비교표 경로로 보냈고,
# 그래서 개념 질문이 "비교하려면 문서가 두 건 이상 필요합니다" 로 끝났다(실측).
#
# 판단 기준: **문서를 두 건 이상 가리켰는가**. 아니면 비교표를 만들 수 없다.
#   · 문서 ID 두 개              → 문서 비교
#   · "두 사업/양쪽/둘 중/문서 간" 같은 복수 문서 지시 → 문서 비교(후속 질문 포함)
#   · "A와 B의 차이" 의 A·B 가 문서를 부르는 말(…사업/공고/문서/용역/기관명)  → 문서 비교
#   · 그 외(용어 두 개)          → 일반 QA(개념 설명)
_MULTI_DOCUMENT_REF_RE = re.compile(
    r"(?:두|둘|양쪽|여러|각|서로\s*다른|다른|나머지)\s*(?:개\s*)?(?:문서|사업|공고|건|곳)"
    r"|문서\s*(?:간|끼리|들)|사업\s*(?:간|끼리|들)|공고\s*(?:간|끼리|들)"
    r"|둘\s*중|어느\s*쪽|양쪽|두\s*건")
# 비교 대상 한 짝을 이루는 두 표현을 잡는다("A와 B의 차이", "A과 B를 비교").
_PAIR_RE = re.compile(
    r"(?P<a>[^\s,，.。?!]{1,30})\s*(?:과|와|랑|이랑|vs\.?|대)\s+?(?P<b>[^\s,，.。?!]{1,30})"
    r"[^.]{0,20}?(?:차이|비교|어느\s*쪽|둘\s*중)")
# 표현이 '문서를 부르는 말'인지 — 사업/공고/문서 등 문서 지시어나 기관 접미사가 붙는다.
_DOCUMENT_TERM_RE = re.compile(
    r"(?:사업|공고|문서|용역|제안요청서|입찰|프로젝트|과업)"
    r"|(?:재단|공사|공단|진흥원|연구원|연구소|협회|협의회|위원회|교육청|시청|도청|청|처|"
    r"센터|대학교|공제회|조합|은행|병원|박물관)$")


def _is_document_term(term: str) -> bool:
    return bool(term) and bool(_DOCUMENT_TERM_RE.search(term))


def is_concept_comparison(question: str) -> bool:
    """문서 두 건이 아니라 **개념 두 개**의 차이를 묻는 질문인가.

    ★문서를 두 건 이상 가리키는 신호가 하나라도 있으면 개념 질문이 아니다 —
      후속 질문("둘 중 예산이 큰 건?")의 기존 비교 경로를 그대로 지킨다.
    """
    if len(detect_document_ids(question)) >= 2:
        return False
    if _MULTI_DOCUMENT_REF_RE.search(question):
        return False
    m = _PAIR_RE.search(question)
    if not m:
        return False
    a, b = m.group("a"), m.group("b")
    if _is_document_term(a) or _is_document_term(b):
        return False       # "서울시 사업과 부산시 사업" — 문서 두 건을 부른 것이다
    return True
_DOCUMENT_CONTENT_SIGNAL_RE = re.compile(
    r"RFP-|사업|문서|공고|입찰|제안요청서|참가\s*자격"
)


def _match_any(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def route(question: str, cfg: dict) -> RouteResult:
    """규칙 기반 분기. cfg['routing_fallback']로 폴백 태스크 결정(기본 qa)."""
    if cfg.get("routing_method") != "rule_based":
        raise NotImplementedError("LLM 기반 분기는 아직 구현 안 됨(baseline은 rule_based)")

    # 문서 속 시스템의 사용 방법과 이 챗봇의 사용법은 다른 질문이다.
    # 문서 번호·사업 문맥·특정 제목이 있으면 사용법 키워드만으로 검색을 생략하지 않는다.
    document_content = (bool(_DOCUMENT_CONTENT_SIGNAL_RE.search(question))
                        or looks_like_specific_document(question))
    if (m := _match_any(_SYSTEM_HELP_PATTERNS, question)) and not document_content:
        return RouteResult("no_search_needed", m, no_search_kind=NO_SEARCH_SYSTEM_HELP)
    if (m := _match_any(_GREETING_PATTERNS, question)):
        return RouteResult("no_search_needed", m, no_search_kind=NO_SEARCH_GREETING)
    if (m := _match_any(_COMPARE_PATTERNS, question)):
        # ★개념 두 개의 차이를 묻는 질문은 문서 비교표를 만들 수 없다 — 일반 QA 로 간다.
        #   "차이" 라는 낱말만으로 비교표 경로에 보내지 않는다(§10).
        if is_concept_comparison(question):
            return RouteResult("qa", "concept_difference_explanation")
        # 한 문서 안의 개념 차이 설명은 여러 문서의 고정 필드 비교표가 아니다.
        # 두 문서 이상을 명시하거나 복수 사업을 부른 질문의 기존 비교 경로는 유지한다.
        document_ids = detect_document_ids(question)
        single_reference = (len(document_ids) == 1 or (
            not document_ids and bool(re.search(
                r"(?:이|그|해당|위)\s*(?:문서|사업|공고)(?:의|에서|에|는|은|\s)", question))))
        multiple_reference = bool(re.search(
            r"(?:두|둘|여러|다른|양쪽)\s*(?:문서|사업|공고)|문서\s*간|사업\s*간", question))
        if single_reference and not multiple_reference and needs_explanation(question):
            return RouteResult("qa", "single_document_concept_explanation")
        return RouteResult("compare", m)

    # 선별형 — 조건 파서와 같은 판단을 쓴다(규칙이 두 벌로 갈라지지 않게).
    # 지시 표현("이 사업의 …")이 있으면 활성 문서 후속 질문이므로 선별형이 아니다.
    parse = parse_selection(question)
    if is_selection_question(question, parse):
        if parse.conditions:
            rule = "selection:" + ",".join(
                f"{c.field}/{c.operator}" for c in parse.conditions)
        else:
            rule = "selection:미해석조건+복수요청"
        return RouteResult("select", rule)

    if (m := extract_signal(question)):
        return RouteResult("extract", m)

    # 아무 규칙도 안 걸리면 폴백 (4-10-1 확정: 기본 경로 QA + 재시도)
    fallback_task = cfg.get("routing_fallback", "qa")
    return RouteResult(fallback_task, None, is_fallback=True)  # type: ignore
