"""공식 추출표 v4와 연결 설정이 한 버전을 가리키는지 검사한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "data" / "preprocessed" / "rfp_extraction_table_v4"
TABLE_JSON = TABLE_DIR / "extraction_table_v4.json"
EVAL_DIR = ROOT / "data" / "evalsets" / "final" / "v2"


def test_v4_release_assets_and_versions_are_consistent():
    """현재 버전이 올라가도 과거 공식 v4 자체는 그대로 재현돼야 한다."""
    table = json.loads(TABLE_JSON.read_text(encoding="utf-8"))
    metadata = json.loads((TABLE_DIR / "extraction_metadata.json").read_text(
        encoding="utf-8"))

    assert table["extraction_version"] == metadata["extraction_version"] == "v4"
    assert table["schema_version"] == metadata["schema_version"] == "1-12-2/v3"
    assert table["row_count"] == len(table["rows"]) == 1200
    assert len({row["document_id"] for row in table["rows"]}) == 100
    assert all(row["extraction_version"] == "v4" for row in table["rows"])
    assert hashlib.sha256(TABLE_JSON.read_bytes()).hexdigest() == \
        "44c973d79d7fe4d7b5693fdad070572e846c11a15b8bf8bcaca66c4e16d2a1df"


def test_v4_contains_only_the_two_approved_consoritum_corrections():
    table = json.loads(TABLE_JSON.read_text(encoding="utf-8"))
    rows = {(row["document_id"], row["field_name"]): row for row in table["rows"]}
    r67 = rows[("RFP-000067", "컨소시엄 요건")]
    r81 = rows[("RFP-000081", "컨소시엄 요건")]

    assert r67["status"] == "value_present"
    assert "공동수급 형태로 제안할 경우" in r67["answer_raw"]
    assert r81["status"] == "value_present"
    assert "10%를 초과하여 하도급" in r81["answer_raw"]


def test_evalset_v2_provenance_points_to_v4_without_changing_answers():
    """과거 공식 평가셋 v2도 손대지 않고 그대로 보존한다."""
    items_path = EVAL_DIR / "items.jsonl"
    raw = items_path.read_bytes()
    version_text = (EVAL_DIR / "VERSION.txt").read_text(encoding="utf-8")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    assert len(items) == 50
    assert "extraction_table: v4" in version_text
    assert f"evalset_sha256: {hashlib.sha256(raw).hexdigest()}" in version_text
    assert "candidate" not in raw.decode("utf-8")
    table_sources = {
        ev.get("source")
        for item in items
        for ev in item.get("evidence", [])
        if ev.get("kind") == "extraction_table"
    }
    assert table_sources == {"extraction_table_v4"}
