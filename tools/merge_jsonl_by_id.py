"""Replace selected JSONL records by ID while preserving the base file order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def row_id(row: dict) -> str:
    value = row.get("id") or row.get("question_id")
    if not value:
        raise ValueError("JSONL row has no id or question_id")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = read_rows(Path(args.base))
    replacement_rows = read_rows(Path(args.replacement))
    replacements = {row_id(row): row for row in replacement_rows}
    base_ids = {row_id(row) for row in base}
    unknown = sorted(set(replacements) - base_ids)
    if unknown:
        raise SystemExit(f"replacement ids absent from base: {unknown}")
    merged = [replacements.get(row_id(row), row) for row in base]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    print(json.dumps({
        "base_count": len(base),
        "replacement_count": len(replacements),
        "output_count": len(merged),
        "replaced_ids": sorted(replacements),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
