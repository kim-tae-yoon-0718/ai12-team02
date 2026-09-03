import pytest
from pydantic import ValidationError

from grader.models import EvaluationItem


def test_v02_schema_loads():
    """스키마 v0.2 — 15개 필드. answer_type 값은 document_set/value/list/summary/
    comparison/unanswerable."""
    item = EvaluationItem.model_validate(
        {
            "id": "EXT-007",
            "question": "이 사업 사업 금액이 얼마야",
            "task_type": "extraction",
            "answer_type": "value",
            "document_id": "DOC-091",
            "answer_raw": "222,180,200원",
            "answer_normalized": 222180200,
            "field_tag": "critical",
            "location": {"document": "DOC-091", "section": "Ⅲ. 사업 개요", "ref_no": "표 5"},
        }
    )
    assert item.id == "EXT-007"
    assert item.answer_type == "value"
    assert item.document_id == "DOC-091"
    assert item.answer_normalized == 222180200


def test_v02_document_set_answer_in_answer_raw():
    item = EvaluationItem.model_validate(
        {
            "id": "SEL-001", "question": "예산 5억 이상인 IT 사업 알려줘",
            "task_type": "selection", "answer_type": "document_set",
            "answer_raw": ["DOC-014", "DOC-036"], "answer_source": "table",
            "field_tag": "critical",
        }
    )
    assert item.answer_raw == ["DOC-014", "DOC-036"]


def test_v02_unanswerable_reason_in_answer_raw():
    item = EvaluationItem.model_validate(
        {
            "id": "QA-039", "question": "한국미래산업진흥원에서 낸 스마트팜 사업 있어?",
            "task_type": "qa", "answer_type": "unanswerable",
            "answer_raw": "그런 사업을 찾을 수 없습니다",
        }
    )
    # 별도 필드가 아니라 answer_raw 뷰
    assert item.unanswerable_reason == "그런 사업을 찾을 수 없습니다"
    assert item.checkpoints == []


def test_v02_summary_checkpoints_in_answer_raw():
    item = EvaluationItem.model_validate(
        {
            "id": "QA-050", "question": "이 사업 요약해줘", "task_type": "qa",
            "answer_type": "summary", "answer_raw": ["과업 범위", "사업기간", "예산"],
        }
    )
    assert item.checkpoints == ["과업 범위", "사업기간", "예산"]
    assert item.unanswerable_reason is None


def test_removed_v01_and_split_out_fields_are_rejected_as_extra():
    """v0.1 필드 + v0.2에서 answer_raw 로 흡수된 checkpoints/unanswerable_reason 을
    입력으로 주면 extra=forbid 로 걸러져야 한다 — 조용히 무시되면 마이그레이션 누락을 못 잡는다."""
    for legacy_field, value in [
        ("document_unspecified", False),
        ("conversational", False),
        ("time_dependent", False),
        ("difficulty", "hard"),
        ("schema_version", "v0.1"),
        ("checkpoints", ["a", "b"]),
        ("unanswerable_reason", "not_in_corpus"),
        ("completeness_rule", "exact_all"),
    ]:
        with pytest.raises(ValidationError):
            EvaluationItem.model_validate(
                {
                    "id": "Q1", "question": "q", "task_type": "qa",
                    "answer_type": "value", legacy_field: value,
                }
            )


def test_document_unspecified_is_derived_from_unspecified_type():
    with_type = EvaluationItem.model_validate(
        {
            "id": "Q2", "question": "q", "task_type": "extraction", "answer_type": "list",
            "unspecified_type": "abbreviation", "intermediate_answer": "DOC-058",
            "answer_raw": ["중소기업 인증 보유"],
        }
    )
    without_type = EvaluationItem.model_validate(
        {"id": "Q3", "question": "q", "task_type": "extraction", "answer_type": "value"}
    )
    assert with_type.document_unspecified is True
    assert without_type.document_unspecified is False


def test_time_dependent_is_derived_from_reference_time():
    item = EvaluationItem.model_validate(
        {
            "id": "Q4", "question": "q", "task_type": "qa", "answer_type": "value",
            "reference_time": "2024-06-01", "answer_raw": "가능",
        }
    )
    assert item.time_dependent is True


def test_conversational_is_derived_from_scenario_type():
    item = EvaluationItem.model_validate(
        {
            "id": "Q5", "question": "이거 지금 지원 가능해?", "task_type": "qa",
            "answer_type": "value", "scenario_type": "workflow_chain",
            "active_document_id": "DOC-014", "answer_raw": "가능",
        }
    )
    assert item.conversational is True


def test_unconfirmed_fields_can_be_empty():
    item = EvaluationItem.model_validate(
        {
            "id": "Q6", "question": "지역제한 있는 사업 알려줘",
            "task_type": "selection", "answer_type": "document_set",
            "answer_raw": [], "answer_source": "verified",
        }
    )
    assert item.field_tag is None
