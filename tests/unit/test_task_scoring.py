from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import (
    check_format,
    combine_status,
    grade_abstention,
    grade_comparison,
    grade_content,
    grade_list,
    grade_selection,
    grade_short_answer,
    grade_summary_checkpoint,
    score_item,
)

DEFAULT_CFG = {"docset_partial_credit": True, "miss_weight": 0.7,
              "require_table_format": True, "grade_citations": True}


def _item(**over):
    # v0.2 기본: extraction / value
    base = dict(
        id="T", question="q", task_type="extraction", answer_type="value",
    )
    base.update(over)
    return EvaluationItem(**base)


def _resp(**over):
    base = dict(id="T", answer="x")
    base.update(over)
    return ModelResponse(**base)


# ------------------------------------------------------------------ document_set (선별형)

def test_selection_zero_hit_case():
    # v0.2: 선별형 정답 문서 배열은 answer_raw 에 담긴다
    it = _item(task_type="selection", answer_type="document_set", answer_raw=[])
    s = grade_selection(it, _resp(selected_document_ids=[]), DEFAULT_CFG)
    assert s.score == 1.0
    s2 = grade_selection(it, _resp(selected_document_ids=["A"]), DEFAULT_CFG)
    assert s2.score == 0.0


def test_selection_missing_weighs_more_than_extra():
    """1-5: 누락 > 오검색."""
    it = _item(task_type="selection", answer_type="document_set", answer_raw=["A", "B", "C"])
    miss = grade_selection(it, _resp(selected_document_ids=["A", "B"]), DEFAULT_CFG).score
    extra = grade_selection(it, _resp(selected_document_ids=["A", "B", "C", "Z"]), DEFAULT_CFG).score
    assert miss < extra


# ------------------------------------------------------------------ value (단일 값)

def test_short_answer_exact():
    it = _item(answer_raw="5억원")
    assert grade_short_answer(it, _resp(answer="5억원"), DEFAULT_CFG).score == 1.0


def test_short_answer_wrong():
    it = _item(answer_raw="5억원")
    assert grade_short_answer(it, _resp(answer="서울특별시입니다"), DEFAULT_CFG).score == 0.0


# ------------------------------------------------------------------ list (exact-all 고정)

def test_list_four_out_of_five_is_zero():
    """★5개 중 4개는 80점이 아니라 오답이다."""
    gold = ["가", "나", "다", "라", "마"]
    it = _item(answer_type="list", answer_normalized=gold)
    resp = _resp(answer="가 나 다 라", structured_answer=["가", "나", "다", "라"])
    s = grade_list(it, resp)
    assert s.score == 0.0
    assert s.detail["completeness_rule"] == "exact_all"
    assert s.detail["n_missing"] == 1


def test_list_all_present_is_one():
    gold = ["가", "나"]
    it = _item(answer_type="list", answer_normalized=gold)
    resp = _resp(answer="가 그리고 나 필요")
    assert grade_list(it, resp).score == 1.0


def test_list_omission_and_addition_are_separated():
    gold = ["가", "나", "다"]
    it = _item(answer_type="list", answer_normalized=gold)
    resp = _resp(answer="가 나 다 그리고 별첨서류", structured_answer=["가", "나", "다", "별첨서류"])
    s = grade_list(it, resp)
    assert s.detail["n_missing"] == 0
    assert s.detail["n_extra"] == 1


# ------------------------------------------------------------------ summary (checkpoint — v0.2: answer_raw)

def test_summary_without_checkpoints_is_not_applicable():
    it = _item(task_type="qa", answer_type="summary", answer_raw="문자열이라 체크포인트 배열 아님")
    s = grade_summary_checkpoint(it, _resp(answer="아무 말"))
    assert s.detail["applicable"] is False


def test_summary_checkpoint_coverage():
    it = _item(task_type="qa", answer_type="summary", answer_raw=["과업 범위", "사업기간"])
    s = grade_summary_checkpoint(it, _resp(answer="과업 범위를 정리하면"))
    assert s.score == 0.5


# ------------------------------------------------------------------ comparison

def test_comparison_one_cell_wrong():
    it = _item(task_type="qa", answer_type="comparison",
              answer_normalized=["A|예산|5억원", "B|예산|7억원"])
    resp = _resp(answer="| A | B |", structured_answer={"A": {"예산": "5억원"}, "B": {"예산": "5억원"}})
    s = grade_comparison(it, resp)
    assert s.score == 0.5
    assert s.detail["conclusion_at_risk"] is True


def test_comparison_real_evalset_format():
    """임현진 평가셋 실제 형식: 행마다 {항목, <RFP-ID>: 값}. (PRAC-QA-003)"""
    it = _item(task_type="qa", answer_type="comparison", answer_raw=[
        {"항목": "예산", "RFP-000038": "230,000천원", "RFP-000043": "248,796천원"},
        {"항목": "사업기간", "RFP-000038": "4개월", "RFP-000043": "5개월"},
    ])
    resp = _resp(answer="| ... |", structured_answer={
        "RFP-000038": {"예산": "230,000천원", "사업기간": "4개월"},
        "RFP-000043": {"예산": "248,796천원", "사업기간": "3개월"},  # 1칸 오답
    })
    s = grade_comparison(it, resp)
    assert s.detail["cells"] == 4
    assert s.score == 0.75


# ------------------------------------------------------------------ 기권 (v0.2: answer_type=unanswerable)

def test_abstention_hallucination():
    it = _item(task_type="qa", answer_type="unanswerable", answer_raw="코퍼스에서 찾을 수 없습니다")
    r = grade_abstention(it, _resp(answer="3억원입니다", abstained=False))
    assert r.abstention_kind == "hallucination"


def test_abstention_over_refusal():
    it = _item(answer_raw="5억원", field_tag="minor")
    r = grade_abstention(it, _resp(abstained=True))
    assert r.abstention_kind == "over_refusal"


def test_abstention_critical_is_separated():
    """★critical 필드는 '못 찾으면 기권'이 정상 동작 — 과잉 거절과 같은 칸에 세면 안 된다."""
    it = _item(answer_raw="2026-09-15", field_tag="critical")
    r = grade_abstention(it, _resp(abstained=True))
    assert r.abstention_kind == "critical_abstain"


def test_abstention_correct_refusal_is_ok():
    it = _item(task_type="qa", answer_type="unanswerable", answer_raw="그런 사업을 찾을 수 없습니다")
    r = grade_abstention(it, _resp(answer="찾을 수 없습니다", abstained=True))
    assert r.abstention_kind == "ok"
    assert r.should_abstain is True


# ------------------------------------------------------------------ 형식 계약 (3-3)

def test_format_violation_for_list_without_structured_answer():
    it = _item(answer_type="list", answer_normalized=["가", "나"])
    fmt = check_format(it, _resp(answer="가 나"), DEFAULT_CFG)
    assert fmt.passed is False


def test_format_violation_for_comparison_without_table():
    it = _item(task_type="qa", answer_type="comparison", answer_normalized=["A|예산|5억원"])
    fmt = check_format(it, _resp(answer="그냥 문장입니다"), DEFAULT_CFG)
    assert fmt.passed is False


def test_content_correct_but_format_fail_is_not_pass():
    """3-3: 내용이 맞아도 output contract 를 어기면 최종 PASS 가 아니다."""
    it = _item(answer_type="list", answer_normalized=["가"])
    resp = _resp(answer="가")  # structured_answer 없음 → format 위반, 내용은 맞음
    result = score_item(it, resp, DEFAULT_CFG)
    assert result["task_score"].score == 1.0
    assert result["format_status"].passed is False
    assert result["final_status"] == "FAIL-format"


def test_binary_task_pass():
    it = _item(answer_raw="5억원")
    resp = _resp(answer="5억원")
    result = score_item(it, resp, DEFAULT_CFG)
    assert result["final_status"] == "PASS"


def test_partial_credit_task_is_pending_threshold():
    """document_set/summary/comparison 처럼 부분점수가 있는 태스크는 baseline 실측 전까지
    임의의 PASS/FAIL 경계를 만들지 않는다."""
    it = _item(task_type="selection", answer_type="document_set", answer_raw=["A"])
    resp = _resp(selected_document_ids=["A"])
    result = score_item(it, resp, DEFAULT_CFG)
    assert result["final_status"] == "PENDING_THRESHOLD"


def test_unanswerable_is_scored_as_abstention_not_content():
    """v0.2: answer_type=unanswerable 은 grade_content 가 아니라 기권 채점으로 분기한다."""
    it = _item(task_type="qa", answer_type="unanswerable", answer_raw="찾을 수 없습니다")
    result = score_item(it, _resp(answer="없습니다", abstained=True), DEFAULT_CFG)
    assert result["task_score"].kind == "abstention"
    assert result["task_score"].score == 1.0
    assert result["final_status"] == "PASS"


def test_dispatch_covers_all_v02_answer_types():
    for at, resp in [
        ("value", _resp(answer="x")),
        ("list", _resp(answer="가", structured_answer=["가"])),
        ("summary", _resp(answer="x")),
        ("comparison", _resp(answer="| a |", structured_answer={})),
        ("document_set", _resp(selected_document_ids=[])),
    ]:
        tt = "selection" if at == "document_set" else ("extraction" if at in ("value", "list") else "qa")
        it = _item(task_type=tt, answer_type=at, answer_raw=(["가"] if at in ("list", "summary") else "x"))
        s = grade_content(it, resp, DEFAULT_CFG)
        assert "알 수 없는" not in str(s.detail)
