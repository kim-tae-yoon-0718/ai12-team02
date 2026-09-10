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
from .normalize import (
    ABSENCE_CONFIRMED, ABSENCE_NOT_VERIFIED, FIELD_ABSENT_INFERRED,
    FIELD_ABSENT_NOT_VERIFIED, FIELD_ABSENT_STATED, classify_absence,
    classify_field_absent_answer, classify_response_kind,
    match_short, normalize_text, strip_enum_prefix, strip_label_prefix,
)

ItemMatcher = Callable[[str, str], bool]


def _gold_value(item: EvaluationItem):
    """정규화형이 '의미 있게' 있으면 그것을, 아니면 answer_raw 를 정답으로 쓴다(팀장 2-2).
    answer_normalized 가 None / 빈 문자열 / 빈 목록·dict 면 answer_raw 로 폴백 —
    단 answer_raw 도 빈 목록이면 그대로 빈 정답으로 남긴다."""
    n = item.answer_normalized
    if n is None or n == "" or n == [] or n == {}:
        return item.answer_raw
    return n


_JOSA_AFTER = re.compile(r"^(은|는|이|가|을|를|과|와|의|에|에서|으로|로|도|만|까지|부터|및|와의|,|\.|:|\))?(\s|$)")


def _item_match_once(g: str, a: str) -> bool:
    if not g:
        return False
    if g == a:
        return True
    idx = a.find(g)
    while idx != -1:
        before = a[idx - 1] if idx > 0 else " "
        if before == " " and _JOSA_AFTER.match(a[idx + len(g):]):
            return True
        idx = a.find(g, idx + 1)
    return False


def _item_match(gold_item: str, answer_text: str) -> bool:
    """항목 하나가 '언급됐다' 판정 — 규칙 기반 기본 매처.
    ★순수 substring 은 짧은 항목에서 오탐이 크다(팀장 2-3). 정규화 후 완전 일치이거나,
      토큰 경계(앞은 공백/시작, 뒤는 공백/끝/한국어 조사)로 포함될 때만 인정한다.
      prefix 매치는 하지 않는다 — 'A' 가 'A등급' 에 매치되지 않는다.
    ★목록 항목 앞의 순번 표시(①, 1. 등)는 값의 일부가 아니다(2026-09-04, 실 데이터 확인 —
      모델이 항목 1개는 번호를 지우고 나머지는 안 지운 채 냈는데, 정답 쪽엔 전부 번호가
      있어서 그 항목만 안 맞았다). 원문/순번-제거본 둘 다로 대조한다."""
    g = normalize_text(gold_item)
    a = normalize_text(answer_text)
    if _item_match_once(g, a):
        return True
    g2 = normalize_text(strip_enum_prefix(gold_item))
    a2 = normalize_text(strip_enum_prefix(answer_text))
    if (g2, a2) != (g, a) and _item_match_once(g2, a2):
        return True
    return False


# 하위호환 별칭
_default_matcher = _item_match


_DEFAULT_CLARIFY_ROUTES = frozenset({"애매_되묻기"})  # ROUTE_CLARIFY, 이태민 answer_pipeline.py 확정


def _is_clarification(response: ModelResponse, cfg: dict | None = None) -> bool:
    """되묻기 응답인지 판정 — route/structured_answer(실제 모델이 채워 보내는 구조화 신호)를
    1차로 보고, 없을 때만 텍스트 패턴(classify_response_kind)으로 보조 판정한다.
    ★2026-09-04: 실제 모델 코드(answer_pipeline._clarify)를 확인해보니 되묻기 응답은
    route="애매_되묻기" 와 structured_answer.clarification_needed=True 를 항상 채워 보낸다 —
    정규식 추측보다 이게 훨씬 정확하므로 이걸 우선한다."""
    routes = (cfg or {}).get("clarify_routes") or _DEFAULT_CLARIFY_ROUTES
    route = (response.route or "").strip()
    if route and route in routes:
        return True
    sa = response.structured_answer
    if isinstance(sa, dict) and sa.get("clarification_needed") is True:
        return True
    return classify_response_kind(response.answer) == "CLARIFICATION"




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

def _gold_is_clarification(item: EvaluationItem) -> bool:
    """이 문항의 **정답이** 되묻는 문장인가.

    되묻기 수용 분기를 열지 말지를 가른다 — 문항에 unspecified_type 이 붙었다는
    사실만으로는 부족하다(해소 가능한 문항에서 되묻기만 해도 만점이 되어 버린다).
    """
    gold = item.answer_raw
    texts = [gold] if isinstance(gold, str) else [g for g in (gold or []) if isinstance(g, str)]
    return any(classify_response_kind(t) == "CLARIFICATION" for t in texts)


def field_absent_target(item: EvaluationItem) -> dict | None:
    """이 문항이 **추출표 미기재(field_absent) 문항**인가. 맞으면 그 근거를 돌려준다.

    정답 문자열에 '없음'이 들어갔는지로 판단하지 않는다 — 문항의 근거가 실제로
      kind=extraction_table / status=field_absent / document=정답 문서 / field=정답 필드
    인지 확인한다. 근거 중 값이 있는(value_present) 추출표 행이 섞여 있으면
    '미기재만으로 답하는 문항'이 아니므로 적용하지 않는다.
    """
    evidences = list(getattr(item, "evidence", None) or [])
    if not evidences:
        return None
    gold_docs = set(as_id_list(item.document_id))

    def _g(ev, key):
        return getattr(ev, key, None) if hasattr(ev, "kind") else ev.get(key)

    table_ev = [ev for ev in evidences if _g(ev, "kind") == "extraction_table"]
    if not table_ev:
        return None
    absent = [ev for ev in table_ev if _g(ev, "status") == "field_absent"]
    if len(absent) != len(table_ev):
        return None                       # 값이 있는 행이 섞였다 → 미기재 전용 문항 아님
    hit = None
    for ev in absent:
        doc, fld = _g(ev, "document"), _g(ev, "field")
        if not doc or not fld:
            return None                   # 문서·필드가 없는 근거로는 미기재를 확인할 수 없다
        if gold_docs and doc not in gold_docs:
            return None                   # 정답 문서와 다른 문서의 미기재 행이면 적용 안 함
        hit = {"document": doc, "field": fld, "status": "field_absent"}
    return hit


def grade_short_answer(item: EvaluationItem, response: ModelResponse, cfg: dict) -> TaskScore:
    """단일 값 채점(answer_type=value) — 이진(정확 일치/불일치). 태스크가 qa 든
    extraction 이든 답 형태가 value 면 이 함수로 채점한다."""
    accept = []
    # answer_normalized 는 단일 문자열 또는 **허용 답 목록**이다. 추출표 기반 문항은
    # 빌더가 [핵심 값, 연결된 추출표 원문 값] 을 여기에 넣는다 — 모델이 공식 추출표
    # 값을 그대로 돌려줘도 정답이 되도록. 목록에 없는 덧붙임은 여전히 오답이다.
    norm = item.answer_normalized
    for cand in (norm if isinstance(norm, list) else [norm]):
        if isinstance(cand, str) and cand and cand != item.answer_raw:
            accept.append(cand)
    if isinstance(item.answer_raw, str):
        # ★추출 원문 라벨("나. 사업기간 : ")을 생략하고 핵심 값만 답해도 정답으로 인정한다
        # (채점기 피드백) — 원문은 그대로 두고 대안 후보만 추가(치환 아님).
        core = strip_label_prefix(item.answer_raw)
        if core:
            accept.append(core)
    # ★자연어 '없음' 채점 — 확정 정책(2026-09-04). 기본 ON.
    #   정답이 "항목이 없다"는 뜻(없음/해당 없음/…)일 때:
    #     · 같은 뜻을 **단정한** 표현(해당 없음·명시되어 있지 않음·없다고 확인)은 정답
    #     · "확인할 수 없다/찾지 못했다/알 수 없다"는 모른다는 뜻 — 정답 아님
    #       (팀장 2-5: 부재와 확인 실패를 합치지 않는다)
    #     · 기권·빈 답·오류 메시지도 정답 아님
    #   ★이 게이트는 match_short **앞**에 있어야 한다. 뒤에 두면 어절 포함 경로가
    #     "확인할 수 없음"(어절에 '없음' 포함)을 먼저 정답으로 삼킨다 — 실측된 구멍.
    #   ★거부(확인 실패·기권·빈 답)는 **토글과 무관하게 항상** 적용한다 — 토글은
    #     "확정 부재의 자연어 표현을 인정할지"만 정한다. 거부까지 토글에 묶으면
    #     OFF 에서 어절 포함 경로가 '알 수 없음'을 다시 삼킨다(실측).
    absence_gate = None
    # ★① 추출표 미기재(field_absent) 문항 — 문서에 항목이 안 적혀 있다는 뜻만 인정한다.
    #   "없음"·"지역 제한 없음" 처럼 실체가 없다고 읽히는 답, 그리고 항목 부재에서
    #   실제 제도·조건·금액을 추론한 답은 오답이다(문서 사실이 아니라 새 주장이다).
    #   판단 근거는 정답 문자열이 아니라 문항의 **근거 상태**다(field_absent_target).
    fa = field_absent_target(item)
    if fa is not None and isinstance(item.answer_raw, str):
        exact_forms = {normalize_text(item.answer_raw).replace(" ", "")} | {
            normalize_text(a).replace(" ", "") for a in accept}
        pred_exact = normalize_text(response.answer_text).replace(" ", "")
        if response.abstained or not response.answer_text.strip():
            absence_gate = (False, "field_absent_rejected(기권·빈 답은 미기재 정답이 아니다)")
        else:
            kind = classify_field_absent_answer(response.answer_text)
            if kind == FIELD_ABSENT_NOT_VERIFIED:
                absence_gate = (False,
                                f"field_absent_not_verified(확인 실패는 '{fa['field']} 미기재'가 아니다)")
            elif kind == FIELD_ABSENT_INFERRED:
                absence_gate = (False,
                                "field_absent_overreach(항목 미기재에서 실제 조건·금액까지 단정했다)")
            elif pred_exact in exact_forms:
                absence_gate = (True, "field_absent_exact(정답 문구와 같음)")
            elif kind == FIELD_ABSENT_STATED and cfg.get("accept_natural_absence_phrasing", True):
                absence_gate = (True, f"field_absent_stated(문서에 '{fa['field']}' 항목이 없다고 말함)")
            else:
                absence_gate = (False,
                                "field_absent_unclear(문서에 항목이 기재되지 않았다는 진술이 아니다)")
    # ★②(기존) 정답이 '없음' 계열인데 위 미기재 근거가 없는 문항의 자연어 부재 채점.
    if absence_gate is None and isinstance(item.answer_raw, str) \
            and classify_absence(item.answer_raw) == ABSENCE_CONFIRMED:
        if response.abstained or not response.answer_text.strip():
            absence_gate = (False, "absence_rejected(기권·빈 답은 '없음' 정답이 아니다)")
        else:
            kind = classify_absence(response.answer_text)
            if kind == ABSENCE_NOT_VERIFIED:
                absence_gate = (False, "absence_not_verified(확인 실패는 '없음'이 아니다)")
            elif kind == ABSENCE_CONFIRMED and cfg.get("accept_natural_absence_phrasing", True):
                absence_gate = (True, "absence_confirmed(항목이 없다고 단정 — 정답과 같은 뜻)")
            else:
                # 토글 OFF 의 확정 부재, 또는 부재 단정도 확인 실패도 아닌 답:
                # ★'없음' 어절을 품고 있다는 이유로 어절 포함 경로에서 정답이 되면 안 된다
                #   — 실측: "없음(확인할 수 없음)". 부재 정답은 분류기 판정 또는
                #   정규화 완전 일치만 인정한다(토글 OFF 에서도 우회 통과 금지).
                exact = normalize_text(response.answer_text).replace(" ", "") in {
                    normalize_text(item.answer_raw).replace(" ", "")} | {
                    normalize_text(a).replace(" ", "") for a in accept}
                absence_gate = ((True, "normalized_exact") if exact
                                else (False, "absence_unclear(부재 단정도 확인 실패도 아닌 답)"))
    if absence_gate is not None:
        ok, why = absence_gate
    else:
        ok, why = match_short(
            item.answer_raw, response.answer, accept=accept,
            allow_partial=cfg.get("allow_partial", False),
            residual_limit=cfg.get("residual_limit", 20),  # 팀장 2-9: config 관리
        )
    # ★문항 자체가 모호해서(unspecified_type, 임현진 v0.2 확정 필드 — 예: 국민연금공단
    #   사업이 여러 건이라 특정 불가) gold 가 "되묻는 문장"인 경우, pred 가 gold 문장
    #   그대로가 아니어도 "되묻는 것" 자체가 맞으면 정답으로 인정한다(채점기 피드백:
    #   되묻기는 오답도 거절도 아니다). 새 필드 요구 없이 기존 unspecified_type +
    #   response.answer 텍스트 패턴만으로 판정 — response.abstained 값은 안 건드린다.
    # ★[2026-09-04 정정] 조건이 unspecified_type 존재만이면 오탐이 난다.
    #   EXT-14 는 약어(KUSF) 해소 능력을 재는 문항이라 unspecified_type 이 붙어 있지만
    #   **정답은 되묻기가 아니라 사실 답변**이다. 그런데 이 분기가 열려 있어서
    #   약어를 해소하지 못하고 "어떤 사업을 말씀하시는 건가요?" 라고만 답해도 만점이
    #   나왔다. 되묻기를 인정하려면 **정답 자체가 되묻기**여야 한다.
    if not ok and item.unspecified_type is not None and _gold_is_clarification(item):
        # 응답 쪽 판정은 route="애매_되묻기"/structured_answer.clarification_needed 를
        # 텍스트 패턴보다 우선한다(HR ad62478 계열 채택) — 모델이 실제로 채워 보낸다.
        if _is_clarification(response, cfg):
            ok, why = True, "clarification_accepted(정답 자체가 되묻기 — unspecified_type)"

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
    match = matcher or _item_match
    raw_gold = _gold_value(item)
    if raw_gold is None:
        gold_items = []
    elif isinstance(raw_gold, list):
        gold_items = list(raw_gold)
    else:
        gold_items = [raw_gold]
    text = response.answer_text
    got_items = list(response.structured_answer) if isinstance(response.structured_answer, list) else []

    # ★정규화 항목끼리 1:1 대응 — 한 답변 항목/구간을 여러 정답에 재사용하지 않는다(팀장 2-3).
    #   긴 정답부터 매칭해 'A' 가 'A 요약본' 을 먼저 소진하는 것을 막는다.
    used_got: set[int] = set()
    text_norm = normalize_text(text)          # 소진용 (매치된 구간을 지운다)
    hit, miss = [], []
    for g in sorted(gold_items, key=lambda x: -len(str(x))):
        gs = str(g)
        assigned = False
        for i, x in enumerate(got_items):
            if i not in used_got and match(gs, str(x)):
                used_got.add(i)
                assigned = True
                # ★structured_answer 로 이미 소진된 항목의 원문은 자유텍스트(text_norm) 폴백에서
                #   다시 못 쓰게 지운다 — 안 그러면 "제안서 요약본" 하나가 structured_answer로
                #   '제안서 요약본'을 만족시키고, response.answer 원문에 그 글자가 그대로 남아
                #   '제안서'까지 또 만족시켜 한 답을 두 개로 이중 계산한다(채점기 피드백 회귀).
                nx = normalize_text(str(x))
                idx = text_norm.find(nx)
                if idx != -1:
                    text_norm = text_norm[:idx] + " " * len(nx) + text_norm[idx + len(nx):]
                break
        if not assigned:
            ng = normalize_text(gs)
            if ng and _item_match(gs, text_norm):
                idx = text_norm.find(ng)
                if idx != -1:
                    text_norm = text_norm[:idx] + " " * len(ng) + text_norm[idx + len(ng):]
                assigned = True
        (hit if assigned else miss).append(g)

    extra = [got_items[i] for i in range(len(got_items)) if i not in used_got]
    coverage = len(hit) / len(gold_items) if gold_items else 1.0
    # exact-all: 빠뜨림도 덧붙임도 없어야 통과 (고정 정책). ★덧붙임(환각 항목)을 통과시키면
    # "다 넣고 환각도 덧붙인" 답이 "하나 빠뜨린" 답보다 높게 나온다 — 뒤집힌 순서(3-4-4).
    # 빠뜨림/덧붙임은 detail 에 따로 센다(합산 금지) — score 는 이진.
    score = 1.0 if (not miss and not extra) else 0.0

    return TaskScore(kind="list", score=score, detail={
        "completeness_rule": "exact_all",  # 코드 고정값. 스키마 필드 아님.
        "coverage": coverage, "hit": hit, "missing": miss, "extra": extra,
        "n_missing": len(miss), "n_extra": len(extra),
        "fail_reason": ("missing" if miss else "extra" if extra else None),
        "note": "빠뜨림과 덧붙임은 별도 집계 — 합산 금지(3-4-4). 둘 다 없어야 통과",
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
    # ★체크포인트끼리도 1:1 소진 대응(grade_list 와 동일 원칙, 팀장 2-3) — 응답 한 구절이
    #   여러 체크포인트를 동시에 만족한 것으로 이중 계산되지 않게 한다. 예: 체크포인트
    #   ["제안서","제안서 요약본"] 인데 응답이 "제안서 요약본"뿐이면, 예전엔 각 체크포인트를
    #   응답 원문 전체와 독립적으로 재대조해서 둘 다 hit 로 잡았다(채점기 피드백 회귀).
    #   긴 체크포인트부터 매칭해 소진하면 짧은 것이 남는 글자를 가로채 오탐하는 것도 막는다.
    text_norm = normalize_text(response.answer_text)
    satisfied = [False] * len(cps)
    for i in sorted(range(len(cps)), key=lambda i: -len(str(cps[i]))):
        cs = str(cps[i])
        if match(cs, text_norm):
            satisfied[i] = True
            nc = normalize_text(cs)
            idx = text_norm.find(nc)
            if idx != -1:
                text_norm = text_norm[:idx] + " " * len(nc) + text_norm[idx + len(nc):]
    hit = [c for c, ok in zip(cps, satisfied) if ok]
    miss = [c for c, ok in zip(cps, satisfied) if not ok]
    score = len(hit) / len(cps)
    return TaskScore(kind="summary", score=score,
                     detail={"applicable": True, "covered": hit, "missing": miss})


# ------------------------------------------------------------------ comparison

# 팀장 2-6: 공식 문서 ID 형식만 허용. RFP- 뒤 숫자 6자리.
_CMP_DOC_RE = re.compile(r"RFP-\d{6}$")


def _parse_comparison_gold(value) -> tuple[list[tuple[str, str, str]], list[str]]:
    """비교형 정답을 (document_id, field, value) 튜플 목록 + 오류 목록으로 편다.

    실제 형식 — 행마다 {"항목": <필드>, "<RFP-000000>": <값>, ...}
    ★잘못된/알 수 없는 키는 조용히 버리지 않고 errors 에 남긴다(팀장 2-6).
    """
    out: list[tuple[str, str, str]] = []
    errors: list[str] = []
    if not isinstance(value, list):
        return out, ["비교형 정답이 배열이 아님"]
    for r_i, row in enumerate(value):
        if not isinstance(row, dict):
            # 구 "doc|field|value" 문자열 (테스트 호환)
            parts = str(row).split("|")
            if len(parts) == 3:
                out.append((parts[0], parts[1], parts[2]))
            else:
                errors.append(f"행 {r_i}: dict 아님 ({row!r})")
            continue
        field = str(row.get("항목", "")).strip()
        if not field:
            errors.append(f"행 {r_i}: '항목' 키 없음")
        for k, v in row.items():
            if k == "항목":
                continue
            kk = str(k).strip()
            if _CMP_DOC_RE.match(kk):
                out.append((kk, field, str(v)))
            else:
                errors.append(f"행 {r_i}: 알 수 없는 키 '{kk}' (RFP-000000 형식 아님)")
    return out, errors


def grade_comparison(item: EvaluationItem, response: ModelResponse) -> TaskScore:
    """비교형(2-8-4) — 문서 ID · 필드 · 값 단위로 대조한다.
    우열 판단(어느 쪽이 더 나은가)은 채점 대상이 아니다.

    ★[2026-09-04 정정] 예전에는 **정답에 있는 칸만** 확인했다. 그래서 정답 칸을 다
      맞힌 뒤 관계없는 문서나 필드를 아무리 덧붙여도 1.0 이 나왔다. 비교표에서
      '묻지 않은 문서를 끌어와 붙이는 것'은 대표적인 환각인데 감점이 없었다.

      이제 추가 칸을 분모에 넣는다:
          맞은 정답 칸 수 / (전체 정답 칸 수 + 추가 칸 수)
      값이 틀린 정답 칸은 맞은 칸으로 세지 않는다.

      1.0 은 ①모든 정답 칸 존재 ②모든 값 일치 ③추가 문서 없음 ④추가 필드 없음
      ⑤문서 ID 형식 오류·구조 오류 없음 을 **전부** 만족할 때만 나온다.
    """
    gold_cells, gold_errors = _parse_comparison_gold(_gold_value(item))
    raw = response.structured_answer
    struct_errors: list[str] = []
    if isinstance(raw, dict):
        pred_tbl = raw
    else:
        pred_tbl = {}
        if raw is not None:
            # ★예외로 실행 전체를 멈추지 않는다 — 이 문항만 0점 + 사유 기록.
            struct_errors.append(
                f"비교형 응답 structured_answer 가 dict 아님({type(raw).__name__}) — "
                f"{{문서ID: {{항목: 값}}}} 형태여야 함")

    gold_cells_index = {(d, f): v for d, f, v in gold_cells}
    gold_docs_all = {d for d, _, _ in gold_cells}

    # 응답표의 문서 ID 도 조용히 버리지 않는다(팀장 2-6).
    # ★정답에 있는 문서 ID 는 형식과 무관하게 정상이다 — 형식 검사는 정답에 없는
    #   문서를 끌어온 경우에만 의미가 있다(정답 자체가 그 표기를 쓰고 있으므로).
    pred_key_errors = [
        f"응답표 문서 ID '{k}' — RFP-000000(6자리) 형식 아님"
        for k in pred_tbl
        if str(k).strip() not in gold_docs_all and not _CMP_DOC_RE.match(str(k).strip())
    ]

    # 응답을 (문서, 필드) → 값 으로 편다. 중첩이 dict 가 아니면 오류로 남긴다.
    pred_cells: dict[tuple[str, str], object] = {}
    for doc, fields in pred_tbl.items():
        d = str(doc).strip()
        if not isinstance(fields, dict):
            struct_errors.append(
                f"응답표 '{d}' 의 값이 dict 아님({type(fields).__name__}) — "
                f"{{항목: 값}} 형태여야 함")
            continue
        for f, v in fields.items():
            pred_cells[(d, str(f).strip())] = v

    gold_keys = set(gold_cells_index)
    gold_docs = gold_docs_all

    matched, missing, wrong_cells = [], [], []
    for doc_id, field, gv in gold_cells:
        if (doc_id, field) not in pred_cells:
            missing.append({"document_id": doc_id, "field": field, "gold": gv})
            continue
        pv = pred_cells[(doc_id, field)]
        ok, _why = match_short(gv, pv if pv is not None else "")
        if ok:
            matched.append({"document_id": doc_id, "field": field})
        else:
            wrong_cells.append({"document_id": doc_id, "field": field,
                                "gold": gv, "pred": pv})

    # 정답에 없는 칸 — 문서가 통째로 남는 경우와 필드만 남는 경우를 갈라 기록한다.
    extra_doc_cells, extra_field_cells = [], []
    for (d, f), v in pred_cells.items():
        if (d, f) in gold_keys:
            continue
        row = {"document_id": d, "field": f, "pred": v}
        (extra_doc_cells if d not in gold_docs else extra_field_cells).append(row)
    extra_total = len(extra_doc_cells) + len(extra_field_cells)

    n_gold = len(gold_cells)
    denom = n_gold + extra_total
    score = (len(matched) / denom) if denom else 0.0
    # ★구조 오류(중첩이 dict 아님 등)면 무엇을 비교했는지 신뢰할 수 없다 —
    #   예외로 실행을 멈추지 않고 이 문항만 0점으로 두고 사유를 남긴다(§4-8).
    if struct_errors:
        score = 0.0
    # 형식이 잘못된 문서 ID 는 이미 '추가 칸'으로 분모에 들어가 만점이 나올 수 없다.

    detail = {
        "cells": n_gold,
        "matched_cells": matched,
        "missing_cells": missing,
        "wrong_cells": wrong_cells,
        "extra_document_cells": extra_doc_cells,   # 정답에 없는 **문서**
        "extra_field_cells": extra_field_cells,    # 정답 문서인데 정답에 없는 **필드**
        "n_extra": extra_total,
        "denominator": denom,
        "cell_accuracy": score,
        "conclusion_at_risk": bool(missing or wrong_cells or extra_total),
        "note": "우열 판단은 채점 대상 아님(2-8-4) — 지정 항목 값 일치까지만. "
                "추가 칸은 분모에 포함한다(묻지 않은 문서·필드를 붙이면 감점).",
    }
    if gold_errors:
        detail["gold_errors"] = gold_errors      # 조용히 버리지 않는다(팀장 2-6)
    if pred_key_errors:
        detail["pred_errors"] = pred_key_errors
    if struct_errors:
        detail["structure_errors"] = struct_errors
    return TaskScore(kind="comparison", score=score, detail=detail)


# ------------------------------------------------------------------ 3-3 기권

def grade_abstention(item: EvaluationItem, response: ModelResponse, cfg: dict | None = None) -> AbstentionResult:
    """기권율을 별도 지표로 분리(2-3 → 3-3).

    v0.2: 기권 대상 문항은 answer_type=unanswerable 로 표시된다(구 v0.1은
    unanswerable_reason 필드). 기권 사유 문자열은 answer_raw 에 담긴다.

    ★ field_tag=critical 문항의 기권은 감점이 아니라 정상 동작일 수 있다(1-5:
    추정 금지, 못 찾으면 기권) — 일반 문항의 과잉 거절과 반대 방향이라 같은 칸에
    합산하지 않는다. critical_abstain 축은 severity weighting(5:3:2)과 **별개**로
    유지한다 — 하나가 다른 하나를 대체하지 않는다.
    """
    should_abstain = item.answer_type == "unanswerable"
    # ★abstained 미전달(None)은 형식 오류(check_format 이 잡는다). 채점 계산에선 False 취급하되
    #   본문 표현으로 추측하지 않는다(팀장 2-4).
    did = bool(response.abstained)
    kind = "ok"
    extra: dict[str, Any] = {}
    if should_abstain and did:
        kind = "ok"
        if response.unanswerable_reason:
            extra["response_reason"] = response.unanswerable_reason
    elif should_abstain and not did:
        kind = "hallucination"
    elif not should_abstain and did:
        # ★문항이 원래 모호해서(unspecified_type) 되묻는 응답이면 "불필요한 거절"이 아니다
        #   (채점기 피드백) — abstained bool 자체는 그대로 두고(팀장 2-4), 분류만 구분한다.
        if item.unspecified_type is not None and _is_clarification(response, cfg):
            kind = "clarification_ok"
        else:
            kind = "critical_abstain" if item.field_tag == "critical" else "over_refusal"

    return AbstentionResult(should_abstain=should_abstain, abstained=did,
                            abstention_kind=kind, **extra)


# ------------------------------------------------------------------ 3-3 형식 계약

def check_format(item: EvaluationItem, response: ModelResponse, cfg: dict) -> FormatStatus:
    """내용과 형식을 분리한다. 여기서 FAIL 이면 내용이 맞아도 최종 PASS가 아니다."""
    violations: list[str] = []
    at = item.answer_type

    # 팀장 2-4: abstained 는 모든 응답 필수 — 키 자체가 없으면 형식 오류
    if response.abstained is None:
        violations.append("응답에 abstained 필드 없음 — 모든 응답 필수(true/false 명시)")

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
    # ★4-5 / 6-1: 시스템 실패는 내용 채점보다 먼저다.
    #   failure 가 있으면 답 문자열이 정답과 **같아도** 0점이다. 실패한 실행이
    #   우연히 정답 문자열을 담고 있었다는 것은 시스템이 답을 낸 것이 아니다.
    #   (예: 타임아웃 후 이전 응답 잔여물, 부분 스트림, 캐시 찌꺼기.)
    #   이걸 정답으로 세면 "실패할수록 점수가 오르는" 구간이 생긴다.
    if (response.failure or "").strip():
        task_score = TaskScore(
            kind="system_failure", score=0.0,
            detail={"failure": response.failure,
                    "declared_answer_type": str(item.answer_type),
                    "note": "시스템 실패 — 내용 채점을 하지 않는다. 오답 0점과 구분되며 "
                            "분모에는 남는다(4-5)."})
        return {
            "task_score": task_score,
            "format_status": check_format(item, response, cfg),
            "abstention": grade_abstention(item, response, cfg),
            "final_status": "FAIL-system",
        }

    abstention = grade_abstention(item, response, cfg)

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
