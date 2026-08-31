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


# ── 임현진 FIELD_SPEC 정합 (2026-08-31 HJ 브랜치 대조) ──────────────────

def test_scenario_type_requires_active_document_id():
    it = EvaluationItem.model_validate({
        "id": "QA1", "question": "그거 지금 가능해?", "task_type": "qa", "answer_type": "value",
        "answer_raw": "가능", "scenario_type": "anaphora",  # active_document_id 없음
    })
    assert any("active_document_id" in p for p in check_evalset_integrity([it]))


def test_reference_time_must_be_the_constant():
    it = EvaluationItem.model_validate({
        "id": "QA2", "question": "지금 지원 가능한 사업?", "task_type": "qa",
        "answer_type": "value", "answer_raw": "네", "reference_time": "2025-01-01",
    })
    assert any("2024-06-01" in p for p in check_evalset_integrity([it]))


def test_answer_raw_is_required_for_every_answer_type():
    it = EvaluationItem.model_validate({
        "id": "EXT9", "question": "예산이 얼마?", "task_type": "extraction",
        "answer_type": "value", "document_id": "RFP-000001",  # answer_raw 없음
    })
    assert any("answer_raw 누락" in p for p in check_evalset_integrity([it]))


def test_document_id_list_is_flagged():
    it = EvaluationItem.model_validate({
        "id": "EXT8", "question": "예산이 얼마?", "task_type": "extraction",
        "answer_type": "value", "answer_raw": "5억", "document_id": ["RFP-000001"],
    })
    assert any("document_id 는 문자열" in p for p in check_evalset_integrity([it]))


def test_bad_field_tag_is_rejected_by_model():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvaluationItem.model_validate({
            "id": "X", "question": "q", "task_type": "qa", "answer_type": "value",
            "answer_raw": "x", "field_tag": "잠정",
        })


# ── 유출 검사 (임현진 check_no_leakage 와 같은 목적) ────────────────────

def test_leakage_finds_question_in_tracked_file(tmp_path):
    from grader.validation import check_leakage
    import subprocess
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "generate_v1.txt").write_text(
        "few-shot 예: 이 사업 사업 금액이 얼마야\n답: ...", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    it = EvaluationItem.model_validate({
        "id": "EXT-007", "question": "이 사업 사업 금액이 얼마야",
        "task_type": "extraction", "answer_type": "value", "answer_raw": "x",
    })
    problems = check_leakage([it], tmp_path)
    assert any("유출" in p and "EXT-007" in p for p in problems)
