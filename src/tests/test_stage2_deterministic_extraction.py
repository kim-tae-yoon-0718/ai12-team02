from __future__ import annotations

from answer_pipeline import (
    Answer,
    _deterministic_list_units,
    _stage2_deterministic_extraction,
)
from stage2_agent import Stage2Decision


def _decision(*, mode="list", task="extract", assessment="sufficient"):
    return Stage2Decision(
        action="finalize",
        task_type=task,
        document_scope="specific_documents",
        result_assessment=assessment,
        document_ids=["RFP-000001"],
        requested_fields=["과업 범위"],
        answer_mode=mode,
        used_observation_ids=["O1"],
        used_evidence_ids=["O1:E1"],
        final_answer="모델이 임의로 다시 쓴 문장",
        final_items=["모델이 임의로 나눈 항목"],
    )


def _record(structured, *, abstained=False):
    public = {
        "observation_id": "O1",
        "tool": "table_lookup",
        "requested_fields": ["과업 범위"],
    }
    answer = Answer(
        text="도구 원문",
        task_type="extract",
        route="추출테이블_값조회",
        abstained=abstained,
        structured_answer=structured,
        selected_document_ids=["RFP-000001"],
        sources=["RFP-000001 (extraction_table: 과업 범위)"],
        used_sources=[{"source": "extraction_table", "extraction_version": "v4"}],
        citations=[{"document": "RFP-000001", "field": "과업 범위"}],
        condition_query=[{"field": "과업 범위", "status": "value_present"}],
        condition_result_doc_ids=["RFP-000001"],
    )
    return [(public, answer)]


def test_hierarchy_keeps_top_level_items_without_question_parsing():
    value = {
        "answer_raw": (
            "1) 정보시스템 운영 및 유지관리\n"
            "가) 하위 시스템 A\n"
            "나) 하위 시스템 B\n"
            "2) 연계 데이터 처리\n"
            "3) 개발·운영 업무"
        )
    }
    assert _deterministic_list_units(value) == [
        "정보시스템 운영 및 유지관리", "연계 데이터 처리", "개발·운영 업무"]


def test_complete_list_uses_table_units_not_llm_rewrite():
    result = _stage2_deterministic_extraction(
        _decision(),
        _record({"과업 범위": {
            "status": "value_present",
            "answer_raw": "1) 첫 과업\n2) 둘째 과업",
            "answer_normalized": '["1) 첫 과업", "2) 둘째 과업"]',
        }}),
    )
    assert result is not None
    assert result.text == "첫 과업\n둘째 과업"
    assert result.structured_answer == ["첫 과업", "둘째 과업"]
    assert "모델이" not in result.text
    assert result.citation_diagnostics["ignored_free_form_final_answer"] is True


def test_complete_value_uses_exact_table_value():
    result = _stage2_deterministic_extraction(
        _decision(mode="value"),
        _record({"과업 범위": {
            "status": "value_present",
            "answer_raw": "다. 과업 범위 : 확정된 값",
            "answer_normalized": "확정된 값",
        }}),
    )
    assert result is not None
    assert result.text == "확정된 값"


def test_field_absent_uses_status_template_without_abstaining():
    result = _stage2_deterministic_extraction(
        _decision(mode="value"),
        _record({"과업 범위": {
            "status": "field_absent", "answer_raw": "", "answer_normalized": "",
        }}),
    )
    assert result is not None
    assert "별도로 명시되어 있지 않습니다" in result.text
    assert result.abstained is False


def test_typed_table_list_overrides_incorrect_value_label():
    result = _stage2_deterministic_extraction(
        _decision(mode="value"),
        _record({"과업 범위": {
            "status": "value_present",
            "answer_raw": "가. 첫 방법\n나. 둘째 방법",
            "answer_normalized": '["가. 첫 방법", "나. 둘째 방법"]',
        }}),
    )
    assert result is not None
    assert result.structured_answer == ["첫 방법", "둘째 방법"]
    assert result.citation_diagnostics["llm_answer_mode"] == "value"
    assert result.citation_diagnostics["effective_answer_mode"] == "list"


def test_confirmed_absence_overrides_incorrect_unanswerable_label():
    result = _stage2_deterministic_extraction(
        _decision(mode="unanswerable"),
        _record({"과업 범위": {
            "status": "field_absent", "answer_raw": "", "answer_normalized": "",
        }}),
    )
    assert result is not None
    assert result.abstained is False
    assert result.citation_diagnostics["effective_answer_mode"] == "value"


def test_unresolved_or_non_extraction_keeps_existing_finalizer():
    unresolved = _stage2_deterministic_extraction(
        _decision(),
        _record({"과업 범위": {
            "status": "review_required", "answer_raw": "", "answer_normalized": "",
        }}),
    )
    qa = _stage2_deterministic_extraction(
        _decision(task="qa"),
        _record({"과업 범위": {
            "status": "value_present", "answer_raw": "값", "answer_normalized": "값",
        }}),
    )
    assert unresolved is None
    assert qa is None


def test_table_lookup_is_required_for_deterministic_assembly():
    records = _record({"과업 범위": {
        "status": "value_present", "answer_raw": "값", "answer_normalized": "값",
    }})
    records[0][0]["tool"] = "vector_search"
    assert _stage2_deterministic_extraction(_decision(), records) is None
