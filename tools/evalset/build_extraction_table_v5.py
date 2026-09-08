#!/usr/bin/env python3
#@title 공식 추출표 v5 생성
#@markdown 기존 v4를 보존한 채 원문 대조가 끝난 6개 행의 정리값만 반영하여 v5 JSON·CSV·검토 기록을 만듭니다.
"""Build official extraction table v5 deterministically from preserved v4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "preprocessed" / "rfp_extraction_table_v4"
SOURCE_JSON = SOURCE_DIR / "extraction_table_v4.json"
DECISIONS = Path(__file__).with_name("extraction_table_v5_decisions.json")
DEFAULT_OUT = ROOT / "data" / "preprocessed" / "rfp_extraction_table_v5"
SOURCE_SHA256 = "44c973d79d7fe4d7b5693fdad070572e846c11a15b8bf8bcaca66c4e16d2a1df"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_cell(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, quoting=csv.QUOTE_ALL,
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: normalized_cell(row.get(k, "")) for k in columns})


def build(out_dir: Path) -> None:
    actual = sha256(SOURCE_JSON)
    if actual != SOURCE_SHA256:
        raise SystemExit(f"v4 source hash mismatch: expected {SOURCE_SHA256}, got {actual}")

    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    decision_doc = json.loads(DECISIONS.read_text(encoding="utf-8"))
    if decision_doc["source_table_sha256"] != actual:
        raise SystemExit("decision file points to a different v4 table")

    rows = source["rows"]
    by_key = {(row["document_id"], row["field_name"]): row for row in rows}
    if len(by_key) != len(rows):
        raise SystemExit("source table contains duplicate document/field keys")

    report_rows = []
    for decision in decision_doc["decisions"]:
        key = (decision["document_id"], decision["field_name"])
        if key not in by_key:
            raise SystemExit(f"missing target row: {key}")
        row = by_key[key]
        before = {name: row.get(name) for name in decision["updates"]}
        for name, value in decision["updates"].items():
            row[name] = normalized_cell(value)
        after = {name: row.get(name) for name in decision["updates"]}
        if before == after:
            raise SystemExit(f"decision does not change the source row: {key}")
        report_rows.append({
            "document_id": key[0],
            "field_name": key[1],
            "affected_items": decision["affected_items"],
            "source_refs": decision["source_refs"],
            "reason": decision["reason"],
            "changed_fields": sorted(decision["updates"]),
            "before": before,
            "after": after,
        })

    source["extraction_version"] = "v5"
    for row in rows:
        row["extraction_version"] = "v5"

    out_dir.mkdir(parents=True, exist_ok=True)
    table_json = out_dir / "extraction_table_v5.json"
    table_json.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    with (SOURCE_DIR / "extraction_table_v4.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        columns = list(csv.DictReader(fh).fieldnames or [])
    write_csv(out_dir / "extraction_table_v5.csv", rows, columns)

    shutil.copyfile(SOURCE_DIR / "semantic_decisions_v4.csv",
                    out_dir / "semantic_decisions_v4_inherited.csv")
    shutil.copyfile(SOURCE_DIR / "field_alias_candidates_v4.csv",
                    out_dir / "field_alias_candidates_v4_inherited.csv")
    shutil.copyfile(DECISIONS, out_dir / "alignment_decisions_v5.json")

    report = {
        "source_table": "v4",
        "source_table_sha256": actual,
        "released_table": "v5",
        "changed_row_count": len(report_rows),
        "affected_item_count": len({x for row in report_rows for x in row["affected_items"]}),
        "changes": report_rows,
    }
    (out_dir / "ALIGNMENT_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "schema_version": source["schema_version"],
        "extraction_version": "v5",
        "supersedes": "v4",
        "status": "official",
        "corpus_version": source["corpus_version"],
        "registry_version": source["registry_version"],
        "row_count": source["row_count"],
        "document_count": source["document_count"],
        "field_count": source["field_count"],
        "source_table_sha256": actual,
        "table_sha256": sha256(table_json),
        "generator": "tools/evalset/build_extraction_table_v5.py",
        "decisions": "tools/evalset/extraction_table_v5_decisions.json",
        "changed_rows": len(report_rows),
        "affected_evalset_items": sorted({x for row in report_rows for x in row["affected_items"]}),
        "released_at": "2026-09-08",
        "note": "원문 값은 보존하고 6개 행의 누락 정보·목록 경계·정리값만 바로잡았다.",
    }
    (out_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out_dir / "README.md").write_text(
        "# rfp_extraction_table_v5 (공식)\n\n"
        "v4의 1,200행·100문서·12필드와 스키마를 그대로 보존하면서, 원문 대조가 끝난 "
        "6개 행의 정리값만 바로잡은 공식 버전입니다. v4는 삭제하거나 덮어쓰지 않았습니다.\n\n"
        "- 목록 경계 정정: RFP-000002, RFP-000010, RFP-000030\n"
        "- 누락 정보 복원: RFP-000003, RFP-000008\n"
        "- 원문 조건의 간결한 정리: RFP-000005\n"
        "- 영향 평가 문항: EXT-01, EXT-02, EXT-03, EXT-06, EXT-08, EXT-12, EXT-14\n"
        "- 인덱스·청크는 변경하지 않았으며 재임베딩도 필요하지 않습니다.\n\n"
        "세부 전후 값과 원문 근거 위치는 `ALIGNMENT_REPORT.json`에 기록했습니다.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
