"""grader 쪽 평가셋 처리 — validate_evaluation_set(파서) + check_evalset_integrity(어댑터).

★ 계약 검사 규칙 자체는 checks.check_evalset 단일 출처 → tests/checks/test_check_evalset.py.
  여기서는 grader.validation 어댑터가 그걸 제대로 호출하는지 + 주석 키 처리만 본다.
"""

import json

import pytest

from grader.models import EvaluationItem
from grader.validation import check_evalset_integrity, validate_evaluation_set


def _item(**over):
    base = dict(id="Q1", question="q", task_type="extraction", answer_type="value", answer_raw="x")
    base.update(over)
    return EvaluationItem.model_validate(base)


# ── 어댑터: EvaluationItem 을 받아 checks.check_evalset 로 넘긴다 ──────────

def test_clean_item_passes():
    it = _item(task_type="extraction", answer_type="value", answer_raw="지역 제한 없음",
               field_tag="critical")
    assert it.answer_type == "value"
    assert check_evalset_integrity([it]) == []


def test_state_field_on_item_is_rejected_by_model():
    """field_absent/conflict 같은 상태를 문항 필드로 넣으면 extra=forbid 로 거부."""
    with pytest.raises(Exception):
        EvaluationItem.model_validate({
            "id": "Q1", "question": "q", "task_type": "extraction",
            "answer_type": "value", "answer_raw": "x", "extract_state": "field_absent",
        })


def test_adapter_flags_excluded_duplicate_doc():
    excluded = {"RFP-000006", "RFP-000017"}
    it = _item(id="Q9", document_id="RFP-000006")
    problems = check_evalset_integrity([it], retrieval_excluded_ids=excluded)
    assert any("Q9" in p for p in problems)
    ok = _item(id="Q10", document_id="RFP-000075")
    assert check_evalset_integrity([ok], retrieval_excluded_ids=excluded) == []


def test_adapter_flags_scenario_without_active_doc():
    it = EvaluationItem.model_validate({
        "id": "QA1", "question": "그거 지금 가능해?", "task_type": "qa", "answer_type": "value",
        "answer_raw": "가능", "scenario_type": "anaphora",
    })
    assert any("active_document_id" in p for p in check_evalset_integrity([it]))


def test_bad_field_tag_is_rejected_by_model():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvaluationItem.model_validate({
            "id": "X", "question": "q", "task_type": "qa", "answer_type": "value",
            "answer_raw": "x", "field_tag": "잠정",
        })


# ── practice 세트 주석 키 (validate_evaluation_set) ─────────────────────

def test_practice_meta_key_is_dropped_by_default(tmp_path):
    p = tmp_path / "practice_items.jsonl"
    p.write_text(json.dumps({
        "id": "PRAC-001", "question": "개발 확인용", "task_type": "extraction",
        "answer_type": "value", "answer_raw": "1억원",
        "_source_note": "최종셋 미사용 문서에서 발췌",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    items = validate_evaluation_set(p)  # strict_meta=False
    assert len(items) == 1 and items[0].id == "PRAC-001"


def test_final_set_rejects_meta_key_under_strict(tmp_path):
    p = tmp_path / "final.jsonl"
    p.write_text(json.dumps({
        "id": "Q0001", "question": "q", "task_type": "extraction",
        "answer_type": "value", "answer_raw": "1억원", "_source_note": "x",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="주석 키"):
        validate_evaluation_set(p, strict_meta=True)
