from grader.extraction import circularity_flag, grade_doc_selection, grade_extraction_audit
from grader.models import EvaluationItem, ModelResponse


def test_absent_vs_failed_confusion():
    """★'항목 없음'과 '추출 실패'를 뒤바꾸면 부정 조건 질의가 조용히 틀린다(4-9-8)."""
    rows = [
        {"document_id": "A", "column": "region_limit", "gold_state": "absent",
         "pred_state": "failed", "gold_value": None, "pred_value": None},
        {"document_id": "B", "column": "region_limit", "gold_state": "absent",
         "pred_state": "absent", "gold_value": None, "pred_value": None},
    ]
    rep = grade_extraction_audit(rows, {"region_limit": "major"})
    col = rep["columns"]["region_limit"]
    assert col["absent_vs_failed_confusion_rate"] > 0
    assert col["gate_pass"] is False


def test_extraction_accuracy_uses_normalized_amount_match():
    rows = [
        {"document_id": "A", "column": "budget", "gold_state": "value", "gold_value": "5억원",
         "pred_state": "value", "pred_value": "500,000,000원"},
        {"document_id": "B", "column": "budget", "gold_state": "value", "gold_value": "9억원",
         "pred_state": "value", "pred_value": "9천만원"},  # 단위 오추출
    ]
    rep = grade_extraction_audit(rows, {"budget": "major"})
    col = rep["columns"]["budget"]
    assert col["accuracy"] == 0.5
    assert col["error_direction"]["under"] == 1


def test_circularity_alarm():
    """추출 정확도 ↓ + 선별형 점수 ↑ ⇒ 답안지도 같은 테이블에서 나왔을 가능성."""
    f = circularity_flag(0.95, 0.70, ["table", "table"])
    assert f["suspicious"] is True


def test_no_alarm_when_independent_answers():
    f = circularity_flag(0.95, 0.70, ["verified", "metadata"])
    assert f["suspicious"] is False


def _item(**over):
    # v0.2: 문서 미특정 여부는 unspecified_type 존재로 파생 → doc_selection 채점 대상
    base = dict(
        id="T", question="q", task_type="extraction", answer_type="value",
        unspecified_type="abbreviation", intermediate_answer=["A"],
    )
    base.update(over)
    return EvaluationItem(**base)


def test_doc_selection_same_org_confusion():
    it = _item()
    resp = ModelResponse(id="T", answer="x", abstained=False, selected_document_ids=["B"])
    r = grade_doc_selection(it, resp, doc_org={"A": "조달청", "B": "조달청"})
    assert r["error_kind"] == "same_org_confusion"


def test_doc_selection_no_selection():
    it = _item()
    resp = ModelResponse(id="T", answer="x", abstained=False, selected_document_ids=[])
    r = grade_doc_selection(it, resp)
    assert r["error_kind"] == "no_selection"


def test_doc_selection_not_applicable_when_specified():
    # 문서가 특정된 문항 — unspecified_type 없음 → document_unspecified=False
    it = _item(unspecified_type=None, intermediate_answer=None)
    resp = ModelResponse(id="T", answer="x", abstained=False)
    r = grade_doc_selection(it, resp)
    assert r["applicable"] is False
