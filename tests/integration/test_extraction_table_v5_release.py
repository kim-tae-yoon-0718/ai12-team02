"""공식 추출표 v5·평가셋 v3·채점기 v3의 정렬과 보존 계약."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V4_DIR = ROOT / "data" / "preprocessed" / "rfp_extraction_table_v4"
V5_DIR = ROOT / "data" / "preprocessed" / "rfp_extraction_table_v5"
V2_DIR = ROOT / "data" / "evalsets" / "final" / "v2"
V3_DIR = ROOT / "data" / "evalsets" / "final" / "v3"

EXPECTED = {
    "EXT-01": ("RFP-000002", "참가 자격(면허·실적)"),
    "EXT-02": ("RFP-000005", "지역제한"),
    "EXT-03": ("RFP-000003", "컨소시엄 요건"),
    "EXT-06": ("RFP-000008", "평가 배점"),
    "EXT-08": ("RFP-000010", "제출 방식"),
    "EXT-12": ("RFP-000030", "필수 제출 서류"),
    "EXT-14": ("RFP-000003", "컨소시엄 요건"),
}


def _items(path: Path) -> dict[str, dict]:
    return {item["id"]: item for item in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )}


def _normalized(row: dict):
    value = row["answer_normalized"]
    if isinstance(value, str) and value.startswith("["):
        return json.loads(value)
    return value


def test_current_versions_are_v5_v3_v3_and_index_lineage_stays_v3():
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    grader = yaml.safe_load((ROOT / "config" / "grader.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "gold" / "data_manifest.json").read_text(
        encoding="utf-8"))

    assert cfg["table"] == cfg["extraction_version"] == "v5"
    assert cfg["evalset"] == "v3"
    assert cfg["scorer"] == "v3"
    assert cfg["index_source_extraction_version"] == "v3"
    assert grader["provenance"]["table"] == "v5"
    assert grader["provenance"]["scorer"] == "v3"
    assert manifest["extraction_table_version"] == "v5"
    assert (ROOT / "data" / "evalsets" / "final" / "CURRENT").read_text(
        encoding="utf-8").strip() == "v3"
    assert "scorer: v3" in (ROOT / "src" / "grader" / "VERSION.txt").read_text(
        encoding="utf-8")


def test_v5_changes_exactly_six_semantic_rows_and_preserves_v4():
    v4 = json.loads((V4_DIR / "extraction_table_v4.json").read_text(encoding="utf-8"))
    v5 = json.loads((V5_DIR / "extraction_table_v5.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((V4_DIR / "extraction_table_v4.json").read_bytes()).hexdigest() == \
        "44c973d79d7fe4d7b5693fdad070572e846c11a15b8bf8bcaca66c4e16d2a1df"
    assert v5["extraction_version"] == "v5"
    assert len(v5["rows"]) == len(v4["rows"]) == 1200

    def semantic(row):
        return {k: v for k, v in row.items() if k != "extraction_version"}

    before = {(r["document_id"], r["field_name"]): semantic(r) for r in v4["rows"]}
    after = {(r["document_id"], r["field_name"]): semantic(r) for r in v5["rows"]}
    changed = {key for key in before if before[key] != after[key]}
    assert changed == set(EXPECTED.values())


def test_v5_canonical_values_match_all_seven_source_verified_gold_answers():
    table = json.loads((V5_DIR / "extraction_table_v5.json").read_text(encoding="utf-8"))
    rows = {(r["document_id"], r["field_name"]): r for r in table["rows"]}
    items = _items(V3_DIR / "items.jsonl")

    for item_id, key in EXPECTED.items():
        assert _normalized(rows[key]) == items[item_id]["answer_raw"]


def test_v5_canonical_outputs_receive_full_content_score_for_seven_items():
    from grader.models import EvaluationItem, ModelResponse
    from grader.task_scoring import grade_content

    table = json.loads((V5_DIR / "extraction_table_v5.json").read_text(encoding="utf-8"))
    rows = {(r["document_id"], r["field_name"]): r for r in table["rows"]}
    items = _items(V3_DIR / "items.jsonl")
    cfg = {"allow_partial": False, "residual_limit": 20,
           "accept_natural_absence_phrasing": True}
    for item_id, key in EXPECTED.items():
        item = EvaluationItem.model_validate(items[item_id])
        value = _normalized(rows[key])
        response = ModelResponse(
            id=item_id,
            answer="\n".join(value) if isinstance(value, list) else str(value),
            structured_answer=value if isinstance(value, list) else None,
            abstained=False,
        )
        assert grade_content(item, response, cfg).score == 1.0, item_id


def test_evalset_v3_changes_only_extraction_table_source_version():
    before = _items(V2_DIR / "items.jsonl")
    after = _items(V3_DIR / "items.jsonl")
    assert before.keys() == after.keys()
    for item_id in before:
        left = json.loads(json.dumps(before[item_id], ensure_ascii=False))
        right = json.loads(json.dumps(after[item_id], ensure_ascii=False))
        for evidence in left.get("evidence", []):
            if evidence.get("kind") == "extraction_table":
                evidence.pop("source", None)
        for evidence in right.get("evidence", []):
            if evidence.get("kind") == "extraction_table":
                assert evidence["source"] == "extraction_table_v5"
                evidence.pop("source", None)
        assert left == right, item_id


def test_generators_are_deterministic(tmp_path: Path):
    from tools.evalset.build_evalset_v3 import build as build_evalset
    from tools.evalset.build_extraction_table_v5 import build as build_table

    table_out = tmp_path / "table"
    eval_out = tmp_path / "evalset"
    build_table(table_out)
    build_evalset(eval_out)
    for name in ("extraction_table_v5.json", "extraction_table_v5.csv",
                 "ALIGNMENT_REPORT.json", "extraction_metadata.json"):
        assert (table_out / name).read_bytes() == (V5_DIR / name).read_bytes()
    for name in ("items.jsonl", "VERSION.txt", "README.md", "ALIGNMENT_REPORT.json"):
        assert (eval_out / name).read_bytes() == (V3_DIR / name).read_bytes()
