#!/usr/bin/env python3
#@title 공식 평가셋 v3 생성
#@markdown 기존 v2의 질문과 정답은 그대로 보존하고, 새 추출표 v5·채점기 v3를 가리키는 출처 기록만 갱신해 v3를 만듭니다.
"""Build evalset v3 from preserved v2 without changing questions or answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "evalsets" / "final" / "v2"
SOURCE_ITEMS = SOURCE_DIR / "items.jsonl"
DEFAULT_OUT = ROOT / "data" / "evalsets" / "final" / "v3"
SOURCE_SHA256 = "0cf0868aaa1466b91fee79cba9ac7eb0fe935b32254a9e6b8272d3ca4bbeb007"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out_dir: Path) -> None:
    actual = sha256(SOURCE_ITEMS)
    if actual != SOURCE_SHA256:
        raise SystemExit(f"v2 source hash mismatch: expected {SOURCE_SHA256}, got {actual}")

    before = [json.loads(line) for line in SOURCE_ITEMS.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    after = json.loads(json.dumps(before, ensure_ascii=False))
    changed_evidence = 0
    for item in after:
        for evidence in item.get("evidence", []):
            if (evidence.get("kind") == "extraction_table"
                    and evidence.get("source") == "extraction_table_v4"):
                evidence["source"] = "extraction_table_v5"
                changed_evidence += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    items_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in after),
        encoding="utf-8",
    )
    output_hash = sha256(items_path)

    def without_provenance(item: dict) -> dict:
        copy = json.loads(json.dumps(item, ensure_ascii=False))
        for evidence in copy.get("evidence", []):
            if evidence.get("kind") == "extraction_table":
                evidence.pop("source", None)
        return copy

    if [without_provenance(x) for x in before] != [without_provenance(x) for x in after]:
        raise SystemExit("v3 changed question, answer, or evidence content beyond source version")

    report = {
        "source_evalset": "v2",
        "source_sha256": actual,
        "released_evalset": "v3",
        "released_sha256": output_hash,
        "item_count": len(after),
        "question_changes": 0,
        "answer_changes": 0,
        "evidence_source_changes": changed_evidence,
        "from": "extraction_table_v4",
        "to": "extraction_table_v5",
    }
    (out_dir / "ALIGNMENT_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out_dir / "VERSION.txt").write_text(
        "evalset: v3\n"
        "status: official\n"
        "supersedes: v2\n"
        "source_evalset: v2\n"
        "scorer: v3\n"
        "extraction_table: v5\n"
        "corpus: v2\n"
        "preprocess: v2\n"
        "chunking: v3\n"
        "registry: v2\n"
        "identity: v2\n"
        "index: v2\n"
        "reference_time: 2024-06-01\n"
        "items: 50 (selection 25 / extraction 15 / qa 10)\n"
        f"evalset_sha256: {output_hash}\n"
        "source_base_dev_commit: 15867c007c14b8aeb0de4029f1532e84f8a2915b\n"
        "promoted: 2026-09-08\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# 공식 평가셋 v3\n\n"
        "v2의 50개 질문·정답·근거 내용은 그대로 유지하고, 구조화 추출표 출처 이름만 "
        "공식 v5로 갱신한 버전입니다. v2는 삭제하거나 덮어쓰지 않았습니다.\n\n"
        "- 질문 변경: 0개\n"
        "- 정답 변경: 0개\n"
        f"- 추출표 출처 기록 변경: {changed_evidence}개 근거\n"
        "- 함께 쓰는 채점기: v3\n"
        "- 함께 쓰는 추출표: v5\n\n"
        "자세한 기계 검증 결과는 `ALIGNMENT_REPORT.json`에 있습니다.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
