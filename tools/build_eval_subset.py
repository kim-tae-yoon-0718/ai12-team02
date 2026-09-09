"""Copy selected evaluation rows without modifying the source evaluation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    wanted = list(dict.fromkeys(args.ids))
    rows = [json.loads(line) for line in
            Path(args.source).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row.get("id") or row.get("question_id")): row for row in rows}
    missing = [item_id for item_id in wanted if item_id not in by_id]
    if missing:
        raise SystemExit(f"missing ids: {missing}")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(by_id[item_id], ensure_ascii=False) + "\n"
                for item_id in wanted),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
