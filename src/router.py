"""
F-0 — 질문 유형 경로 분기 (4-10-1 확정).
규칙 기반으로 시작, 분기 실패 시 QA형으로 폴백(4-10-1).
검색 불필요 질문(인사·시스템 사용법) 필터링도 baseline 포함(4-10-0 ①).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

from table_query import FIELD_KEYWORDS

TaskType = Literal["select", "extract", "qa", "compare", "no_search_needed"]


@dataclass
class RouteResult:
    task_type: TaskType
    matched_rule: str | None  # 어느 규칙에 매칭됐는지 (분기 결과 기록용, 4-10-1)
    is_fallback: bool = False  # 규칙이 못 잡아 기본 경로(QA)로 갔는지


# 규칙 기반 패턴 — 표현 다양성 실측(1-2·1-18) 도착 시 갱신 필요
_NO_SEARCH_PATTERNS = [
    r"^(안녕|반가워|고마워|감사)",
    r"(사용법|어떻게 (써|사용|쓰나요))",
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
)
_EXTRACT_PATTERNS = [
    rf"(?:{_FIELD_KEYWORD_ALTERNATION}).*(?:얼마|언제|어디|뭐|무엇|알려줘|확인)",
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
