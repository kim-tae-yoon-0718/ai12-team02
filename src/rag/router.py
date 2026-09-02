"""
F-0 — 질문 유형 경로 분기 (4-10-1 확정).
규칙 기반으로 시작, 분기 실패 시 QA형으로 폴백(4-10-1).
검색 불필요 질문(인사·시스템 사용법) 필터링도 baseline 포함(4-10-0 ①).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

from table_query import FIELD_KEYWORDS, _DEADLINE_KEYWORDS

TaskType = Literal["select", "extract", "qa", "compare", "no_search_needed"]


@dataclass
class RouteResult:
    task_type: TaskType
    matched_rule: str | None  # 어느 규칙에 매칭됐는지 (분기 결과 기록용, 4-10-1)
    is_fallback: bool = False  # 규칙이 못 잡아 기본 경로(QA)로 갔는지


# 규칙 기반 패턴 — 표현 다양성 실측(1-2·1-18) 도착 시 갱신 필요
_NO_SEARCH_PATTERNS = [
    r"^(안녕|반가워|고마워|감사)",
    # ⚠️ 버그 수정(리뷰 반영): 기존엔 "어떻게 (써|사용|쓰나요)"만 봐서
    # "이 사업의 예산을 어떻게 사용하는 건가요?"까지 시스템 사용법 질문으로
    # 오판했다(실제 재현 확인). "시스템/프로그램/이거"처럼 도구 자체를
    # 가리키는 표현이 같이 있을 때만 사용법 질문으로 본다.
    r"(?:시스템|프로그램|서비스|플랫폼|챗봇|이거|이걸|여기).*(?:사용법|어떻게\s*(?:써|사용|쓰나요))",
    r"^사용법",
]
_SELECT_PATTERNS = [
    r"(이상|이하|초과|미만|없는|있는).*(사업|공고)",
    r"(추천|리스트|목록|다 보여|전부 보여)",
]
# ⚠️ 2026-08-31 정정: 예전엔 "마감일"·"발주기관"이 여기 섞여 있었는데, 4-9-8 v3
# 확정으로 이 둘은 최종 12필드에서 빠졌다. table_query.FIELD_KEYWORDS(실제 공식
# 12필드 키워드)를 그대로 재사용해서 필드명이 어긋나지 않게 한다 — 목록이 둘로
# 나뉘어 있으면 한쪽만 고치고 잊어버리기 쉽다.
_FIELD_KEYWORD_ALTERNATION = "|".join(
    re.escape(kw) for kws in FIELD_KEYWORDS.values() for kw in kws
) + "|" + "|".join(re.escape(kw) for kw in _DEADLINE_KEYWORDS)
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
_EXTRACT_PATTERNS = [
    rf"(?:{_FIELD_KEYWORD_ALTERNATION})",
]
_COMPARE_PATTERNS = [
    r"(비교|차이|어느 쪽|둘 중)",
]


def _match_any(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def route(question: str, cfg: dict) -> RouteResult:
    """규칙 기반 분기. cfg['routing_fallback']로 폴백 태스크 결정(기본 qa)."""
    if cfg.get("routing_method") != "rule_based":
        raise NotImplementedError("LLM 기반 분기는 아직 구현 안 됨(baseline은 rule_based)")

    if (m := _match_any(_NO_SEARCH_PATTERNS, question)):
        return RouteResult("no_search_needed", m)
    if (m := _match_any(_COMPARE_PATTERNS, question)):
        return RouteResult("compare", m)
    if (m := _match_any(_SELECT_PATTERNS, question)):
        return RouteResult("select", m)
    if (m := _match_any(_EXTRACT_PATTERNS, question)):
        return RouteResult("extract", m)

    # 아무 규칙도 안 걸리면 폴백 (4-10-1 확정: 기본 경로 QA + 재시도)
    fallback_task = cfg.get("routing_fallback", "qa")
    return RouteResult(fallback_task, None, is_fallback=True)  # type: ignore
