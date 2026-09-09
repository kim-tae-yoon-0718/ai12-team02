"""Combine API-free Stage 3 answers with Stage 2-3 answers for a preflight score."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "outputs/stage3_preflight_blocked"
FALLBACK = ROOT / "outputs/stage2_experiment3_run50_final"
OUT = ROOT / "outputs/stage3_preflight_composite"


def rows(path: Path, key: str) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            result[str(row[key])] = row
    return result


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values),
        encoding="utf-8",
    )


def main() -> None:
    details = rows(PREFLIGHT / "details.jsonl", "question_id")
    fast_ids = {
        qid for qid, row in details.items()
        if (row.get("stage3_fast_path") or {}).get("accepted") is True
    }
    preflight_responses = rows(PREFLIGHT / "responses.jsonl", "id")
    fallback_responses = rows(FALLBACK / "responses.jsonl", "id")
    all_ids = sorted(fallback_responses)
    combined = [
        preflight_responses[qid] if qid in fast_ids else fallback_responses[qid]
        for qid in all_ids
    ]
    write_jsonl(OUT / "responses.jsonl", combined)
    (OUT / "COMPOSITION.json").write_text(
        json.dumps({
            "purpose": "offline preflight; not an observed end-to-end Stage 3 run",
            "question_count": len(combined),
            "api_free_fast_path_count": len(fast_ids),
            "stage2_3_fallback_count": len(combined) - len(fast_ids),
            "fast_path_ids": sorted(fast_ids),
            "expected_incremental_api_cost_usd": 0.0,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": len(combined), "fast": len(fast_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
