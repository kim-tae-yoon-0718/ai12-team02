"""§2 field_absent 채점 정책 — "문서에 항목이 안 적혀 있다" ≠ "실제로 그런 조건이 없다".

수정 전에는 EXT-09/10/11 의 정답이 "없음" 한 낱말이라 다음이 모두 만점이었다.
  · "없음" · "지역 제한 없음" · "지역 제한이 없다고 확인했습니다"
  · "원문에 지역제한 항목이 없으므로 실제 지역 제한도 없습니다"
정답 문구를 문서 사실로 바꾸고, 채점기는 **근거 상태**(kind/status/document/field)를
보고 미기재 채점을 적용한다.
"""
import pytest

from grader.models import EvaluationItem, ModelResponse
from grader.normalize import (
    FIELD_ABSENT_INFERRED, FIELD_ABSENT_NOT_VERIFIED, FIELD_ABSENT_OTHER,
    FIELD_ABSENT_STATED, classify_field_absent_answer,
)
from grader.task_scoring import field_absent_target, score_item

ON = {"residual_limit": 20, "accept_natural_absence_phrasing": True}
OFF = {"residual_limit": 20, "accept_natural_absence_phrasing": False}
GOLD = "이 문서에는 지역제한 항목이 별도로 명시되어 있지 않습니다."


def _absent_item(field="지역제한", doc="RFP-000011", gold=GOLD, status="field_absent"):
    return EvaluationItem.model_validate(dict(
        id="FA", question="지역 제한 조건이 있어?", task_type="extraction",
        answer_type="value", answer_source="table", answer_raw=gold, document_id=doc,
        evidence=[{"kind": "extraction_table", "document": doc, "field": field,
                   "status": status, "source": "extraction_table_v3.1-candidate.1"}]))


def _r(answer, abstained=False):
    return ModelResponse(id="FA", answer=answer, abstained=abstained)


def _score(item, answer, cfg=ON, abstained=False):
    return score_item(item, _r(answer, abstained), cfg)["task_score"]


# ── 인정 ────────────────────────────────────────────────────────────
ACCEPT = [
    GOLD,
    "문서에 지역제한 항목이 명시되어 있지 않습니다",
    "원문에 지역제한 항목이 기재되어 있지 않습니다.",
    "해당 문서에는 지역제한에 관한 항목이 따로 적혀 있지 않습니다.",
    "공고문에 지역제한 항목이 나와 있지 않습니다",
    "원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)",
    "해당 필드는 미기재입니다",
]

# ── 거부 (지시서 §2 목록 + 자연어 변형) ──────────────────────────────
REJECT = [
    "없음", "지역 제한 없음", "공동수급 조건 없음",
    "지역 제한이 없다고 확인했습니다",
    "확인할 수 없습니다", "자료가 부족합니다", "검색 결과가 없습니다", "문서를 찾지 못했습니다",
    "원문에 지역제한 항목이 없으므로 실제 지역 제한도 없습니다",
    "문서에 공동수급 항목이 없으므로 공동수급이 허용됩니다",
    "문서에 예산 항목이 없으므로 예산은 0원입니다",
    "문서에 해당 항목이 미기재이므로 제한 없음",
    "없음(확인할 수 없음)", "지역 제한이 없지는 않습니다", "지역 제한이 없을 수도 있습니다",
    "지역 제한이 없다는 뜻은 아닙니다", "제출 서류가 존재하지 않습니다", "예산은 0원입니다",
    "지역 제한 없음(전국 입찰 가능)",
    "해당 항목이 없어 어느 지역 업체든 참여할 수 있습니다",
    "문서에 지역제한이 기재되어 있지 않으므로 전국 업체가 참여 가능합니다",
    "응답이 없습니다", "답변을 생성하지 못했습니다", "해당 없음", "N/A",
]


@pytest.mark.parametrize("text", ACCEPT)
def test_document_absence_statement_is_correct(text):
    item = _absent_item()
    assert classify_field_absent_answer(text) == FIELD_ABSENT_STATED
    assert _score(item, text).score == 1.0, text


@pytest.mark.parametrize("text", REJECT)
def test_bare_absence_or_inference_is_wrong(text):
    """토글 ON/OFF 어디서도 통과하면 안 된다 — '없음' 어절 우회 포함."""
    item = _absent_item()
    assert classify_field_absent_answer(text) != FIELD_ABSENT_STATED, text
    assert _score(item, text, ON).score == 0.0, text
    assert _score(item, text, OFF).score == 0.0, text


def test_reason_separates_failure_from_overreach():
    item = _absent_item()
    assert "not_verified" in _score(item, "확인할 수 없습니다").detail["why"]
    assert "overreach" in _score(
        item, "문서에 공동수급 항목이 없으므로 공동수급이 허용됩니다").detail["why"]
    assert "unclear" in _score(item, "없음").detail["why"]


def test_abstention_and_empty_answer_are_not_absence():
    item = _absent_item()
    assert _score(item, "", abstained=True).score == 0.0
    assert _score(item, "   ").score == 0.0


def test_inference_classes_are_distinguished():
    assert classify_field_absent_answer("확인할 수 없습니다") == FIELD_ABSENT_NOT_VERIFIED
    assert classify_field_absent_answer(
        "원문에 지역제한 항목이 없으므로 실제 지역 제한도 없습니다") == FIELD_ABSENT_INFERRED
    assert classify_field_absent_answer("없음") == FIELD_ABSENT_OTHER


# ── 게이트가 붙는 조건(근거 상태) ────────────────────────────────────
def test_gate_needs_field_absent_evidence_not_the_word_없음():
    """정답 문자열이 아니라 근거 상태로 판단한다."""
    absent = _absent_item()
    assert field_absent_target(absent) == {
        "document": "RFP-000011", "field": "지역제한", "status": "field_absent"}
    # 값이 있는(value_present) 추출표 문항에는 붙지 않는다
    present = _absent_item(status="value_present")
    assert field_absent_target(present) is None
    # 근거가 없는 '없음' 문항도 붙지 않는다(기존 자연어 부재 채점이 담당)
    plain = EvaluationItem.model_validate(dict(
        id="P", question="q", task_type="extraction", answer_type="value",
        answer_source="table", answer_raw="없음"))
    assert field_absent_target(plain) is None


def test_gate_ignores_evidence_of_a_different_document():
    item = EvaluationItem.model_validate(dict(
        id="FA", question="q", task_type="extraction", answer_type="value",
        answer_source="table", answer_raw=GOLD, document_id="RFP-000011",
        evidence=[{"kind": "extraction_table", "document": "RFP-000099",
                   "field": "지역제한", "status": "field_absent"}]))
    assert field_absent_target(item) is None


def test_mixed_evidence_is_not_a_field_absent_item():
    item = EvaluationItem.model_validate(dict(
        id="M", question="q", task_type="extraction", answer_type="value",
        answer_source="table", answer_raw=GOLD, document_id="RFP-000011",
        evidence=[{"kind": "extraction_table", "document": "RFP-000011",
                   "field": "지역제한", "status": "field_absent"},
                  {"kind": "extraction_table", "document": "RFP-000011",
                   "field": "예산", "status": "value_present"}]))
    assert field_absent_target(item) is None


def test_generic_absence_policy_still_holds_for_plain_없음_items():
    """근거 없는 '없음' 문항의 기존 정책(A-3)은 그대로 유지된다."""
    plain = EvaluationItem.model_validate(dict(
        id="P", question="q", task_type="extraction", answer_type="value",
        answer_source="table", answer_raw="없음"))
    assert score_item(plain, _r("없음"), ON)["task_score"].score == 1.0
    assert score_item(plain, _r("확인할 수 없습니다"), ON)["task_score"].score == 0.0
    assert score_item(plain, _r("없음(확인할 수 없음)"), OFF)["task_score"].score == 0.0
