"""평가셋 무결성 (2-17) — 팀 결정 반영분.

- C/D 팀 결정(2026-08-31): field_absent/conflict 는 문항 스키마 상태 필드가 아니다.
  "없음이 곧 정답"은 answer_type=value + answer_raw="지역 제한 없음". conflict 문항은 없음.
- 1-9-1: 수집 중복 2건(RFP-000006/RFP-000017)은 검색 대상이 아니므로 정답 근거로 금지.
- 임현진 2026-08-30: practice 세트의 `_source_note` 는 허용, 최종셋에서는 FAIL.
"""

import json

import pytest

from grader.models import EvaluationItem
from grader.validation import check_evalset_integrity, validate_evaluation_set


def _item(**over):
    base = dict(id="Q1", question="q", task_type="extraction", answer_type="value", answer_raw="x")
    base.update(over)
    return EvaluationItem.model_validate(base)


# ── C: "없음이 곧 정답" 은 평범한 value 문항 ─────────────────────────────

def test_absence_answer_is_a_normal_value_item():
    it = _item(task_type="extraction", answer_type="value", answer_raw="지역 제한 없음",
               field_tag="critical")
    # 스키마 통과, 별도 상태 필드 없음
    assert it.answer_type == "value"
    assert it.unanswerable_reason is None  # unanswerable 이 아님
    assert check_evalset_integrity([it]) == []


def test_state_field_on_item_is_rejected():
    """field_absent/conflict 같은 상태를 문항 필드로 넣으면 extra=forbid 로 거부."""
    with pytest.raises(Exception):
        EvaluationItem.model_validate({
            "id": "Q1", "question": "q", "task_type": "extraction",
            "answer_type": "value", "answer_raw": "x", "extract_state": "field_absent",
        })


# ── 1-9-1: 수집 중복 문서를 정답 근거로 쓰면 실패 ──────────────────────────

def test_excluded_duplicate_doc_as_gold_is_flagged():
    excluded = {"RFP-000006", "RFP-000017"}
    it = _item(id="Q9", document_id="RFP-000006")
    problems = check_evalset_integrity([it], retrieval_excluded_ids=excluded)
    assert any("중복제외" in p and "Q9" in p for p in problems)

    ok = _item(id="Q10", document_id="RFP-000075")  # 대표본은 허용
    assert check_evalset_integrity([ok], retrieval_excluded_ids=excluded) == []


def test_excluded_check_covers_answer_raw_and_location():
    excluded = {"RFP-000017"}
    sel = EvaluationItem.model_validate({
        "id": "S1", "question": "q", "task_type": "selection", "answer_type": "document_set",
        "answer_raw": ["RFP-000098", "RFP-000017"], "answer_source": "table",
    })
    problems = check_evalset_integrity([sel], retrieval_excluded_ids=excluded)
    assert any("중복제외" in p for p in problems)


# ── practice 세트 주석 키 ────────────────────────────────────────────────

def test_practice_meta_key_is_dropped_by_default(tmp_path):
    p = tmp_path / "practice_items.jsonl"
    p.write_text(json.dumps({
        "id": "PRAC-001", "question": "개발 확인용", "task_type": "extraction",
        "answer_type": "value", "answer_raw": "1억원",
        "_source_note": "최종셋 미사용 문서에서 발췌",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    items = validate_evaluation_set(p)  # strict_meta=False
    assert len(items) == 1
    assert items[0].id == "PRAC-001"
    assert not hasattr(items[0], "_source_note")


def test_final_set_rejects_meta_key_under_strict(tmp_path):
    p = tmp_path / "final.jsonl"
    p.write_text(json.dumps({
        "id": "Q0001", "question": "q", "task_type": "extraction",
        "answer_type": "value", "answer_raw": "1억원", "_source_note": "x",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="주석 키"):
        validate_evaluation_set(p, strict_meta=True)
