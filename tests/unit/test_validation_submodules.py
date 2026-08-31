"""validation/ 하위 모듈 — location / provenance / assets(references)."""

from grader.models import EvaluationItem
from grader.validation import (
    check_data_sanity,
    check_locations,
    check_provenance,
    check_references,
    check_reproducible,
)


def _item(**over):
    base = dict(id="Q1", question="q", task_type="qa", answer_type="value", answer_raw="x")
    base.update(over)
    return EvaluationItem.model_validate(base)


# ── location ──────────────────────────────────────────────────────────

def test_location_placeholder_ref_no_is_flagged():
    it = _item(location={"document": "DOC-1", "section": "3장", "ref_no": ""})
    problems = check_locations([it])
    assert any("ref_no" in p for p in problems)


def test_location_full_coords_pass():
    it = _item(location={"document": "DOC-1", "section": "Ⅲ. 개요", "ref_no": "표 5"})
    assert check_locations([it]) == []


def test_no_location_is_skipped():
    assert check_locations([_item()]) == []


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


# ── references ────────────────────────────────────────────────────────

def test_reference_integrity_missing_doc():
    it = _item(document_id="DOC-999")
    assert any("미존재" in p for p in check_references([it], corpus_doc_ids={"DOC-1"}))


def test_data_sanity_still_works_from_assets():
    clean = {"doc_count": 10, "corpus_version": "v2", "extraction_table_version": "v2"}
    assert check_data_sanity(clean, {"expected_docs": 10}) == []
    assert any("예상" in p for p in check_data_sanity({**clean, "doc_count": 9}, {"expected_docs": 10}))
