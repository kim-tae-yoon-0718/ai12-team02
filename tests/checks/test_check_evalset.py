"""checks.check_evalset — 평가셋 계약 검사 (2-17, origin/HJ 벤더링 + grader 부가분)."""

import json
import subprocess

from checks.check_evalset import (
    check_dup_ids,
    check_excluded_as_gold,
    check_leak,
    check_quota,
    check_ref_intg,
    check_schema,
    load_jsonl,
    run_all,
    scan_tracked_files,
)


def _rec(**over):
    base = dict(id="Q1", question="예산이 얼마인지 알려줘",
               task_type="extraction", answer_type="value", answer_raw="5억원")
    base.update(over)
    return base


# ── C1 스키마 ─────────────────────────────────────────────────────────

def test_clean_item_passes():
    assert check_schema(_rec(), 1) == []


def test_missing_essential_field():
    r = _rec()
    del r["answer_raw"]
    assert any("missing field 'answer_raw'" in e for e in check_schema(r, 1))


def test_bad_task_answer_combo():
    r = _rec(task_type="selection", answer_type="value", answer_source="table")
    assert any("cannot have answer_type" in e for e in check_schema(r, 1))


def test_deprecated_field_rejected():
    assert any("deprecated field 'schema_version'" in e
               for e in check_schema(_rec(schema_version="0.2"), 1))


def test_reference_time_must_be_constant():
    assert any("2024-06-01" in e for e in check_schema(_rec(task_type="qa", reference_time="2025-01-01"), 1))


def test_scenario_type_requires_active_document_id():
    assert any("active_document_id" in e
               for e in check_schema(_rec(task_type="qa", scenario_type="anaphora"), 1))


def test_selection_needs_answer_source():
    r = _rec(task_type="selection", answer_type="document_set", answer_raw=["RFP-000001"])
    assert any("answer source" in e for e in check_schema(r, 1))


def test_explicit_null_rejected():
    assert any("null" in e for e in check_schema(_rec(field_tag=None), 1))


# ── location: 단일 객체 + 비교형 배열 ─────────────────────────────────

def test_location_single_object_ok():
    r = _rec(location={"document": "RFP-000001", "section": "3장", "ref_no": "3장 · 표 5", "line": 88})
    assert check_schema(r, 1) == []


def test_location_metadata_line_exempt():
    """answer_source=metadata (CSV 답변) 는 line 면제 (임현진 09-02)."""
    r = _rec(answer_source="metadata",
             location={"document": "RFP-000038", "section": "CSV", "ref_no": "CSV: bid_deadline"})
    assert check_schema(r, 1) == []


def test_location_array_ok():
    r = _rec(task_type="qa", answer_type="comparison", location=[
        {"document": "RFP-000038", "section": "A", "ref_no": "A · 문단 3", "line": 51},
        {"document": "RFP-000043", "section": "B", "ref_no": "B · 문단 5", "line": 62},
    ])
    assert check_schema(r, 1) == []


def test_location_missing_line_flagged():
    r = _rec(location={"document": "RFP-000038", "section": "A", "ref_no": "A · 표 1"})  # line 없음
    assert any("line" in e for e in check_schema(r, 1))


def test_location_array_missing_key_flagged():
    r = _rec(task_type="qa", answer_type="comparison", location=[
        {"document": "RFP-000038", "section": "A", "line": 5},  # ref_no 없음
    ])
    assert any("location missing key" in e for e in check_schema(r, 1))


# ── C2 / C3 ──────────────────────────────────────────────────────────

def test_dup_ids():
    assert any("Q1" in e for e in check_dup_ids([_rec(), _rec()]))


def test_quota_unknown_task_type():
    assert any("unknown task_type" in e for e in check_quota([_rec(task_type="translation")]))


def test_quota_strict():
    items = [_rec(id=f"S{i}", task_type="selection", answer_type="document_set",
                  answer_raw=["RFP-000001"], answer_source="table") for i in range(3)]
    assert any("selection expected 25, got 3" in e for e in check_quota(items, strict=True))


# ── C4 참조 무결성 (배열 좌표) ───────────────────────────────────────

def test_ref_intg_flags_unknown_doc(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps(["RFP-000001"]), encoding="utf-8")
    it = _rec(location=[{"document": "RFP-999999", "section": "x", "ref_no": "table 1"}])
    assert any("RFP-999999" in e for e in check_ref_intg([it], p))


def test_ref_intg_skips_without_file():
    assert check_ref_intg([_rec()], "/no/such/path.json") == []


# ── 1-9-1 수집중복 문서 (grader 부가분) ──────────────────────────────

def test_excluded_doc_as_gold():
    it = _rec(id="Q9", document_id="RFP-000006")
    assert any("Q9" in e and "RFP-000006" in e for e in check_excluded_as_gold([it]))


# ── C6 유출 ──────────────────────────────────────────────────────────

def test_leak_prac_prefix():
    assert any("PRAC-" in e for e in check_leak([_rec(id="PRAC-001")], None))


def test_leak_practice_doc_overlap(tmp_path):
    prac = tmp_path / "practice_items.jsonl"
    prac.write_text(json.dumps(_rec(id="PRAC-1", document_id="RFP-000038")) + "\n", encoding="utf-8")
    it = _rec(id="F1", location=[{"document": "RFP-000038", "section": "x", "ref_no": "table 1"}])
    assert any("practice document RFP-000038" in e for e in check_leak([it], prac))


# ── 【25】 추적 파일 유출 (grader 부가분) ────────────────────────────

def test_scan_tracked_files(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "generate_v1.txt").write_text(
        "few-shot 예: 이 사업 사업 금액이 얼마야\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    it = _rec(id="EXT-007", question="이 사업 사업 금액이 얼마야")
    assert any("EXT-007" in e for e in scan_tracked_files([it], tmp_path))


# ── 로더 ─────────────────────────────────────────────────────────────

def test_load_jsonl_accepts_pretty_printed(tmp_path):
    p = tmp_path / "practice_items.jsonl"
    p.write_text('{\n "id": "A"\n}\n{\n "id": "B"\n}\n', encoding="utf-8")
    assert [r["id"] for r in load_jsonl(p)] == ["A", "B"]


# ── 오케스트레이터 ───────────────────────────────────────────────────

def test_run_all_clean():
    assert run_all([_rec()]) == []


def test_run_all_final_set_flags_prac(tmp_path):
    probs = run_all([_rec(id="PRAC-9")], final_set=True)
    assert any("PRAC-" in p for p in probs)
    # practice 세트 검사(final_set=False)일 땐 안 잡힌다
    assert run_all([_rec(id="PRAC-9")]) == []
