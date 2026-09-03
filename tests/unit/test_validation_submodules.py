"""grader.validation 하위 모듈 — provenance / assets(1층 데이터 정상성).

평가셋 계약 검사(스키마·참조무결성·좌표·유출)는 tests/checks/test_check_evalset.py 로 이동.
"""

from grader.validation import (
    check_data_sanity,
    check_provenance,
    check_reproducible,
)


# ── provenance ────────────────────────────────────────────────────────

def test_provenance_missing_field_flagged():
    prov = {"corpus": "v2", "preprocess": "v2", "table": "v2", "index": "v1", "evalset": "v1"}
    assert any("scorer" in p for p in check_provenance(prov))


def test_provenance_none_value_flagged():
    prov = {"corpus": None, "preprocess": "v2", "table": "v2",
            "index": "v1", "evalset": "v1", "scorer": "v1"}
    assert any("None" in p for p in check_provenance(prov))


def test_provenance_full_unknown_is_ok_shape():
    prov = {k: "UNKNOWN" for k in ("corpus", "preprocess", "table", "index", "evalset", "scorer")}
    assert check_provenance(prov) == []


def test_reproducible_flags_dirty_and_unknown():
    m = {"git_dirty": True, "provenance": {"corpus": "UNKNOWN", "preprocess": "v2",
         "table": "v2", "index": "v1", "evalset": "v1", "scorer": "v1"}}
    problems = check_reproducible(m)
    assert any("git_dirty" in p for p in problems)
    assert any("미상" in p for p in problems)


# ── 1층 데이터 정상성 ─────────────────────────────────────────────────

def test_data_sanity_still_works_from_assets():
    clean = {"doc_count": 10, "corpus_version": "v2", "extraction_table_version": "v2"}
    assert check_data_sanity(clean, {"expected_docs": 10}) == []
    assert any("예상" in p for p in check_data_sanity({**clean, "doc_count": 9}, {"expected_docs": 10}))
