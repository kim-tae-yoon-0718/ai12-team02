"""checks.check_evalset — 평가셋 계약 검사 단일 출처 (2-17).

원본 origin/HJ `src/checks/check_evalset.py` 벤더링본. raw dict 위에서 돈다.
"""

import json
import subprocess

from checks.check_evalset import (
    check_dup_ids,
    check_leak,
    check_location_coords,
    check_quota,
    check_ref_intg,
    check_schema,
    load_jsonl,
    run_all,
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
    r = _rec(schema_version="0.2")
    assert any("deprecated field 'schema_version'" in e for e in check_schema(r, 1))


def test_reference_time_must_be_constant():
    r = _rec(task_type="qa", reference_time="2025-01-01")
    assert any("2024-06-01" in e for e in check_schema(r, 1))


def test_scenario_type_requires_active_document_id():
    r = _rec(task_type="qa", scenario_type="anaphora")
    assert any("active_document_id" in e for e in check_schema(r, 1))


def test_selection_needs_answer_source():
    r = _rec(task_type="selection", answer_type="document_set", answer_raw=["RFP-000001"])
    assert any("answer_source" in e for e in check_schema(r, 1))


def test_explicit_null_is_rejected():
    r = _rec(field_tag=None)
    assert any("is null" in e for e in check_schema(r, 1))


def test_document_id_must_be_string():
    r = _rec(document_id=["RFP-000001"])
    assert any("document_id" in e and "string" in e for e in check_schema(r, 1))


# ── C2 / C3 ──────────────────────────────────────────────────────────

def test_dup_ids():
    assert any("Q1" in e for e in check_dup_ids([_rec(), _rec()]))


def test_quota_unknown_task_type():
    assert any("unknown task_type" in e for e in check_quota([_rec(task_type="translation")]))


def test_quota_strict_counts():
    items = [_rec(id=f"S{i}", task_type="selection", answer_type="document_set",
                  answer_raw=["RFP-000001"], answer_source="table") for i in range(3)]
    assert any("selection expected 25, got 3" in e for e in check_quota(items, strict=True))


# ── C4 참조 무결성 ───────────────────────────────────────────────────

def test_ref_intg_missing_doc():
    it = _rec(document_id="RFP-999999")
    assert any("코퍼스에 없는 문서" in e for e in check_ref_intg([it], valid_ids={"RFP-000001"}))


def test_ref_intg_excluded_doc_as_gold():
    it = _rec(id="Q9", document_id="RFP-000006")
    problems = check_ref_intg([it], excluded_ids={"RFP-000006", "RFP-000017"})
    assert any("검색 대상 아닌 문서" in e and "Q9" in e for e in problems)


def test_ref_intg_covers_selection_answer_raw():
    sel = _rec(id="S1", task_type="selection", answer_type="document_set",
               answer_raw=["RFP-000098", "RFP-000017"], answer_source="table")
    problems = check_ref_intg([sel], excluded_ids={"RFP-000017"})
    assert any("검색 대상 아닌 문서" in e for e in problems)


def test_ref_intg_skips_without_ids():
    assert check_ref_intg([_rec(document_id="RFP-999999")]) == []


# ── 좌표 (2-9) ───────────────────────────────────────────────────────

def test_location_placeholder_ref_no_flagged():
    it = _rec(location={"document": "RFP-000001", "section": "3장", "ref_no": ""})
    assert any("ref_no" in e for e in check_location_coords([it]))


def test_location_part_of_warned():
    it = _rec(location={"document": "RFP-000001", "section": "4. 추진일정", "ref_no": "표 7 (2/3)"})
    assert any("(N/M)" in e for e in check_location_coords([it]))


def test_location_clean_passes():
    it = _rec(location={"document": "RFP-000001", "section": "Ⅲ. 개요", "ref_no": "표 5"})
    assert check_location_coords([it]) == []


# ── C6 유출 ──────────────────────────────────────────────────────────

def test_leak_prac_prefix_in_final():
    assert any("PRAC-" in e for e in check_leak([_rec(id="PRAC-001")], final_set=True))


def test_leak_practice_only_doc_as_gold():
    it = _rec(id="Q5", document_id="RFP-000038")  # PRACTICE_ONLY_DOCS
    assert any("practice 전용 문서" in e for e in check_leak([it], final_set=True))


def test_leak_practice_checks_off_by_default():
    """practice 세트 자체를 검사할 땐 PRAC 접두어·전용문서 검사가 오탐이므로 꺼져 있어야 한다."""
    it = _rec(id="PRAC-001", document_id="RFP-000038")
    assert check_leak([it]) == []


def test_leak_question_in_tracked_file(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "generate_v1.txt").write_text(
        "few-shot 예: 이 사업 사업 금액이 얼마야\n답: ...", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    it = _rec(id="EXT-007", question="이 사업 사업 금액이 얼마야")
    problems = check_leak([it], repo_root=tmp_path)
    assert any("EXT-007" in e and "추적 파일" in e for e in problems)


def test_leak_practice_id_overlap(tmp_path):
    prac = tmp_path / "practice_items.jsonl"
    prac.write_text(json.dumps(_rec(id="SHARED-1"), ensure_ascii=False) + "\n", encoding="utf-8")
    problems = check_leak([_rec(id="SHARED-1")], practice_path=prac, final_set=True)
    assert any("교집합" in e for e in problems)


# ── 오케스트레이터 ───────────────────────────────────────────────────

# ── 실제 팀 데이터에서 나온 케이스 ─────────────────────────────────────

def test_load_jsonl_accepts_pretty_printed_concatenated_json(tmp_path):
    """임현진 evalset/practice_items.jsonl 은 줄바꿈 있는 JSON 을 이어 붙인 형식이다."""
    p = tmp_path / "practice_items.jsonl"
    p.write_text('{\n "id": "A",\n "x": 1\n}\n{\n "id": "B",\n "x": 2\n}\n', encoding="utf-8")
    recs = load_jsonl(p)
    assert [r["id"] for r in recs] == ["A", "B"]


def test_ref_intg_splits_multi_doc_comparison_location():
    """comparison 문항은 location.document 가 'RFP-000038, RFP-000043' 처럼 이어붙인 문자열."""
    it = _rec(id="CMP", task_type="qa", answer_type="comparison", answer_raw=[{"항목": "예산"}],
              location={"document": "RFP-000038, RFP-000043",
                        "section": "RFP-000038: A / RFP-000043: B", "ref_no": "문단 3"})
    # 둘 다 유효 → 통과
    assert check_ref_intg([it], valid_ids={"RFP-000038", "RFP-000043"}) == []
    # 하나가 코퍼스에 없으면 그 ID 만 지목
    probs = check_ref_intg([it], valid_ids={"RFP-000038"})
    assert any("RFP-000043" in e for e in probs)
    assert not any("RFP-000038, RFP-000043" in e for e in probs)  # 통짜 문자열로 잡지 않음


def test_run_all_clean():
    assert run_all([_rec()]) == []


def test_run_all_collects_multiple():
    bad = _rec(id="B1", task_type="selection", answer_type="value", reference_time="2020-01-01")
    problems = run_all([bad, bad])
    assert len(problems) >= 3  # combo + reference_time + dup id
