"""
grader.task_scoring — 3-3 형식 계약 / 3-4 ~ 3-4-5 채점기 본체 / 3-3 기권 분리

확정 정책 (임현진 2-2 / 김하루 채점 정책 — 스키마 v0.2):
  task_type   : selection / extraction / qa            (3분류 유지)
  answer_type : document_set / value / list / summary / comparison / unanswerable
  document_set: 선별형 정답 — 문서 ID 배열은 answer_raw 에 담긴다
  list        : 독립 task 아님 → **extraction + answer_type=list** 로 처리
  list 최종 판정 : completeness_rule 은 스키마에 넣지 않는다 — exact-all 이 고정 정책이므로
                  **코드에서 고정**한다 (Coverage/Missing/Extra 는 진단용).
  field_tag   : 부분점수 기준이 아니라 오답 심각도 표시(critical/major/minor) +
                3-3-2 ①가중 평균 채점 입력(2026-08-28 재확인, diagnostics.
                severity_weighted_score 참고) — 이 모듈에서는 critical_abstain
                판정(grade_abstention)에만 직접 쓰고, 가중 평균 계산 자체는
                diagnostics 집계 단계에서 한다.
  format_status: 내용과 독립적인 축. 내용이 맞아도 output contract 를 어기면 최종 FAIL.

★ threshold(정답으로 인정할 점수 경계)는 selection/summary/comparison 처럼 부분점수가
  있는 태스크에서는 baseline 실측 전까지 확정하지 않는다(문서8 "아직 최종 확정하면
  안 되는 것"). 그래서 이런 태스크의 최종 PASS/FAIL 은 "PENDING_THRESHOLD" 로 남겨두고,
  임의의 숫자를 넣지 않는다. list/short_answer 처럼 정책 자체가 이진(binary)인 태스크만
  PASS/FAIL 을 즉시 낸다.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .models import (
    AbstentionResult,
    EvaluationItem,
    FormatStatus,
    ModelResponse,
    TaskScore,
    as_id_list,
)
from .normalize import match_short, normalize_text

ItemMatcher = Callable[[str, str], bool]


def _default_matcher(gold_item: str, answer_text: str) -> bool:
    return normalize_text(gold_item) in normalize_text(answer_text)


# ------------------------------------------------------------------ selection (Recall 중심)

def grade_selection(item: EvaluationItem, response: ModelResponse, cfg: dict) -> TaskScore:
    """선별형(answer_type=document_set). v0.2: 정답 문서 ID 배열은 answer_raw 에 담긴다
    (구 v0.1은 document_id). 0건 정답 허용(2-8-1).

    1-5: 누락 > 오검색 — 빠진 공고는 존재 자체를 모르지만, 잘못 낀 공고는 열어보면
    걸러진다. 그래서 재현율에 miss_weight, 정밀도에 (1-miss_weight)를 준다.
    ★ 이 점수는 항상 3-2-1(추출 정확도)과 함께 읽는다 — answer_source=table 이면
      정답 자체가 시스템과 같은 재료일 수 있다(circularity, extraction.py 참고).
    """
    gold = set(as_id_list(item.answer_raw) or as_id_list(item.document_id))
    got = set(response.selected_document_ids)
    partial_credit = cfg.get("docset_partial_credit", True)
    miss_weight = cfg.get("miss_weight", 0.7)

    detail: dict[str, Any] = {
        "gold_n": len(gold), "pred_n": len(got),
        "missing": sorted(gold - got), "extra": sorted(got - gold),
        "answer_source": item.answer_source,
        "circularity_warning": item.answer_source == "table",
    }

    if not gold:  # 0건이 정답인 문항
        score = 1.0 if not got else 0.0
        detail.update({"zero_hit_case": True})
        return TaskScore(kind="document_set", score=score, detail=detail)

    recall = len(gold & got) / len(gold)
    precision = len(gold & got) / len(got) if got else 0.0
    detail.update({"recall": recall, "precision": precision})

    if not partial_credit:
        score = 1.0 if gold == got else 0.0
    else:
        score = miss_weight * recall + (1 - miss_weight) * precision
    return TaskScore(kind="document_set", score=score, detail=detail)


# ------------------------------------------------------------------ answer_type=value (단일 값)

def grade_short_answer(item: EvaluationItem, response: ModelResponse, cfg: dict) -> TaskScore:
    """단일 값 채점(answer_type=value) — 이진(정확 일치/불일치). 태스크가 qa 든
    extraction 이든 답 형태가 value 면 이 함수로 채점한다."""
    accept = []
    if isinstance(item.answer_normalized, str) and item.answer_normalized != item.answer_raw:
        accept.append(item.answer_normalized)
    ok, why = match_short(
        item.answer_raw, response.answer, accept=accept,
        allow_partial=cfg.get("allow_partial", False),
    )
    return TaskScore(kind="value", score=1.0 if ok else 0.0,
                     detail={"why": why, "gold": item.answer_raw, "pred": response.answer})


# ------------------------------------------------------------------ extraction + answer_type=list

def grade_list(item: EvaluationItem, response: ModelResponse,
              matcher: ItemMatcher | None = None) -> TaskScore:
    """목록형 = extraction + answer_type=list.

    ★ 완결성 판정은 **exact-all 고정** — 스키마에 completeness_rule 필드를 두지 않는다.
    5개 중 4개는 80점이 아니라 오답이다. Coverage/Missing/Extra 는 진단용으로만 계산한다.
    빠뜨림과 덧붙임을 반드시 따로 센다 — 합치면 '다 넣고 환각도 덧붙인' 답변이
    '하나 빠뜨린' 답변보다 높게 나온다.
    """
    match = matcher or _default_matcher
    # ★정답이 문자열로 들어와도 list("문자열")로 글자 단위 분해되지 않게 한다.
    # (2층 check_evalset_integrity가 list 문항의 비배열 정답을 걸러주지만, 이 함수
    #  자체도 방어적이어야 한다 — score_item에서 직접 불릴 수 있으므로.)
    raw_gold = item.answer_normalized if item.answer_normalized is not None else item.answer_raw
    if raw_gold is None:
        gold_items = []
    elif isinstance(raw_gold, list):
        gold_items = list(raw_gold)
    else:
        gold_items = [raw_gold]
    text = response.answer
    got_items = list(response.structured_answer) if isinstance(response.structured_answer, list) else []

    hit, miss = [], []
    for g in gold_items:
        found = match(str(g), text) or any(match(str(g), str(x)) for x in got_items)
        (hit if found else miss).append(g)

    extra = [x for x in got_items if not any(match(str(g), str(x)) for g in gold_items)]
    coverage = len(hit) / len(gold_items) if gold_items else 1.0
    score = 1.0 if not miss else 0.0  # exact-all — 고정 정책, 설정으로 바꿀 수 없다

    return TaskScore(kind="list", score=score, detail={
        "completeness_rule": "exact_all",  # 코드 고정값. 스키마 필드 아님.
        "coverage": coverage, "hit": hit, "missing": miss, "extra": extra,
        "n_missing": len(miss), "n_extra": len(extra),
        "note": "빠뜨림과 덧붙임은 별도 집계 — 합산 금지(3-4-4)",
    })


# ------------------------------------------------------------------ qa: summary (checkpoint)

def grade_summary_checkpoint(item: EvaluationItem, response: ModelResponse,
                             matcher: ItemMatcher | None = None) -> TaskScore:
    """요약형(2-8-2) — 체크포인트 기반 부분점수. 통짜 LLM 채점보다 판정이 안정적이고
    이유 추적이 쉽다. 환각(근거 밖 내용)은 여기서 감점하지 않는다 — judge_faithfulness
    축(3-3-1)과 짝지어 읽는다. 여기서 이중으로 벌점을 주면 원인 분리가 안 된다."""
    match = matcher or _default_matcher
    # v0.2: 체크포인트 배열은 answer_raw 에 담긴다(별도 checkpoints 필드 없음).
    # item.checkpoints 는 answer_raw 리스트의 읽기 전용 뷰다.
    cps = item.checkpoints or []
    if not cps:
        return TaskScore(kind="summary", score=0.0, detail={
            "applicable": False,
            "reason": "answer_raw 에 체크포인트 배열 없음 — judge_checkpoint 항목 단위 채점 불가",
        })
    hit = [c for c in cps if match(c, response.answer)]
    miss = [c for c in cps if c not in hit]
    score = len(hit) / len(cps)
    return TaskScore(kind="summary", score=score,
                     detail={"applicable": True, "covered": hit, "missing": miss})


# ------------------------------------------------------------------ comparison

_CMP_DOC_RE = re.compile(r"RFP-\d{4,}")


def _parse_comparison_gold(value) -> list[tuple[str, str, str]]:
    """비교형 정답(2-8-4 확정: 문서 ID 명시형 + 지정 항목 + 값 정렬)을
    (document_id, field, value) 튜플 목록으로 편다. 두 표현을 받는다:

    ① 임현진 평가셋 실제 형식 — 행마다 {"항목": <필드>, "<RFP-ID>": <값>, ...}
         [{"항목": "예산", "RFP-000038": "230,000천원", "RFP-000043": "248,796천원"}, ...]
    ② 잠정 문자열 형식(구 테스트 호환) — "document_id|field|value"
    """
    if not isinstance(value, list):
        return []
    out: list[tuple[str, str, str]] = []
    for row in value:
        if isinstance(row, dict):
            field = str(row.get("항목") or row.get("field") or row.get("항목명") or "").strip()
            for k, v in row.items():
                if _CMP_DOC_RE.fullmatch(str(k).strip()):
                    out.append((str(k).strip(), field, str(v)))
        else:
            parts = str(row).split("|")
            if len(parts) == 3:
                out.append((parts[0], parts[1], parts[2]))
    return out


def grade_comparison(item: EvaluationItem, response: ModelResponse) -> TaskScore:
    """비교형(2-8-4) — 문서 ID 명시형 + 지정 항목 + 값 정렬까지만 채점한다.
    우열 판단(어느 쪽이 더 나은가)은 채점 대상이 아니다."""
    gold_cells = _parse_comparison_gold(item.answer_normalized or item.answer_raw)
    pred_tbl = response.structured_answer if isinstance(response.structured_answer, dict) else {}

    total = wrong = 0
    wrong_cells = []
    for doc_id, field, gv in gold_cells:
        total += 1
        pv = (pred_tbl.get(doc_id) or {}).get(field)
        ok, _ = match_short(gv, pv if pv is not None else "")
        if not ok:
            wrong += 1
            wrong_cells.append({"document_id": doc_id, "field": field, "gold": gv, "pred": pv})
    cell_acc = (total - wrong) / total if total else 0.0

    return TaskScore(kind="comparison", score=cell_acc, detail={
        "cells": total, "wrong_cells": wrong_cells, "cell_accuracy": cell_acc,
        "conclusion_at_risk": wrong > 0,
        "note": "우열 판단은 채점 대상 아님(2-8-4) — 지정 항목 값 일치까지만",
    })


# ------------------------------------------------------------------ 3-3 기권

def grade_abstention(item: EvaluationItem, response: ModelResponse) -> AbstentionResult:
    """기권율을 별도 지표로 분리(2-3 → 3-3).

    v0.2: 기권 대상 문항은 answer_type=unanswerable 로 표시된다(구 v0.1은
    unanswerable_reason 필드). 기권 사유 문자열은 answer_raw 에 담긴다.

    ★ field_tag=critical 문항의 기권은 감점이 아니라 정상 동작일 수 있다(1-5:
    추정 금지, 못 찾으면 기권) — 일반 문항의 과잉 거절과 반대 방향이라 같은 칸에
    합산하지 않는다. critical_abstain 축은 severity weighting(5:3:2)과 **별개**로
    유지한다 — 하나가 다른 하나를 대체하지 않는다.
    """
    should_abstain = item.answer_type == "unanswerable"
    did = response.abstained
    kind = "ok"
    extra: dict[str, Any] = {}
    if should_abstain and did:
        kind = "ok"
        if response.unanswerable_reason:
            extra["response_reason"] = response.unanswerable_reason
    elif should_abstain and not did:
        kind = "hallucination"
    elif not should_abstain and did:
        kind = "critical_abstain" if item.field_tag == "critical" else "over_refusal"

    return AbstentionResult(should_abstain=should_abstain, abstained=did,
                            abstention_kind=kind, **extra)


# ------------------------------------------------------------------ 3-3 형식 계약

def check_format(item: EvaluationItem, response: ModelResponse, cfg: dict) -> FormatStatus:
    """내용과 형식을 분리한다. 여기서 FAIL 이면 내용이 맞아도 최종 PASS가 아니다."""
    violations: list[str] = []
    at = item.answer_type

    if at == "list" and not isinstance(response.structured_answer, list):
        violations.append("list 답변은 structured_answer 가 배열이어야 함(output contract)")

    if at == "comparison":
        require_table = cfg.get("require_table_format", True)
        looks_table = ("|" in (response.answer or "")) or isinstance(response.structured_answer, dict)
        if require_table and not looks_table:
            violations.append("비교형 출력은 표 형태여야 함(4-11-1)")

    if at == "document_set" and not isinstance(response.selected_document_ids, list):
        violations.append("선별형(document_set) 응답은 selected_document_ids 가 배열이어야 함")

    # ★citation은 여기서 FAIL로 만들지 않는다 — "안 붙임"과 "틀리게 붙임"을 좌표
    # 단위로 실제 채점하는 로직은 retrieval.grade_citation()에 있다(3-4-3).
    # 형식 계약은 내용/출처 정확도와 독립된 축이라는 원래 설계를 유지한다.

    return FormatStatus(passed=not violations, violations=violations)


# ------------------------------------------------------------------ 디스패처 + 3-3 최종 판정

# 정책 자체가 이진(즉시 PASS/FAIL)인 태스크. document_set/summary/comparison 은
# 부분점수라 baseline 실측 전까지 PENDING_THRESHOLD 로 남긴다.
_BINARY_ANSWER_TYPES = {"list", "value", "abstention"}


def grade_content(item: EvaluationItem, response: ModelResponse, cfg: dict,
                  matcher: ItemMatcher | None = None) -> TaskScore:
    """answer_type(v0.2) 에 따라 채점기를 고른다. unanswerable 은 score_item 이
    grade_content 이전에 기권 채점으로 분기하므로 여기 오지 않는다."""
    at = item.answer_type
    if at == "document_set":
        return grade_selection(item, response, cfg)
    if at == "list":
        return grade_list(item, response, matcher)
    if at == "value":
        return grade_short_answer(item, response, cfg)
    if at == "summary":
        return grade_summary_checkpoint(item, response, matcher)
    if at == "comparison":
        return grade_comparison(item, response)
    return TaskScore(kind=str(at), score=0.0, detail={"error": f"알 수 없는 answer_type: {at}"})


def combine_status(format_status: FormatStatus, task_score: TaskScore) -> str:
    """3-3: format_pass + content_pass 조합.

    list/value/기권 는 정책 자체가 이진이라 즉시 PASS/FAIL 을 낸다.
    document_set/summary/comparison 은 부분점수 태스크라 baseline 실측 전까지
    "정답으로 인정할 점수 경계"가 없다 — 임의로 만들지 않고 PENDING_THRESHOLD 로 남긴다.
    """
    if not format_status.passed:
        content_known = task_score.kind in _BINARY_ANSWER_TYPES
        if content_known and task_score.score < 1.0:
            return "FAIL-format+content"
        return "FAIL-format"

    if task_score.kind in _BINARY_ANSWER_TYPES:
        return "PASS" if task_score.score >= 1.0 else "FAIL-content"
    return "PENDING_THRESHOLD"


def score_item(item: EvaluationItem, response: ModelResponse, cfg: dict,
              matcher: ItemMatcher | None = None) -> dict:
    abstention = grade_abstention(item, response)

    if abstention.should_abstain:
        # ★기권율은 일반 지표와 별도로 추적한다(2-3→3-3). answer_raw/answer_normalized가
        # 없는 문항을 answer_type 채점기(short_answer 등)에 그대로 태우면 "정답 없음"과
        # 비교하는 무의미한 판정이 나온다 — 여기서는 기권 정확도 자체가 content 점수다.
        task_score = TaskScore(kind="abstention", score=1.0 if abstention.abstention_kind == "ok" else 0.0,
                               detail={"abstention_kind": abstention.abstention_kind})
    else:
        task_score = grade_content(item, response, cfg, matcher)

    format_status = check_format(item, response, cfg)
    final_status = combine_status(format_status, task_score)
    return {
        "task_score": task_score,
        "format_status": format_status,
        "abstention": abstention,
        "final_status": final_status,
    }
