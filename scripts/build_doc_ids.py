#!/usr/bin/env python3
"""문서 등록부 → grader 참조 무결성 입력 (corpus_doc_ids.json / excluded_doc_ids.json).

grader run --corpus-doc-ids / --excluded-doc-ids 가 먹는 JSON 배열을 만든다.
등록부 CSV 는 BOM 이 있어 utf-8-sig 로 읽는다(박예진 C청킹 §3-1).

  RAG_ROOT=/srv/rfp python scripts/build_doc_ids.py --out-dir data/gold
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def build(registry_csv: Path, out_dir: Path) -> dict:
    with registry_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    active = [r["document_id"] for r in rows if str(r.get("active", "")).lower() == "true"]
    excluded = [r["document_id"] for r in rows
                if str(r.get("retrieval_eligible", "")).lower() == "false"]
    eligible = [d for d in active if d not in set(excluded)]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_doc_ids.json").write_text(json.dumps(sorted(active), ensure_ascii=False, indent=0))
    (out_dir / "excluded_doc_ids.json").write_text(json.dumps(sorted(excluded), ensure_ascii=False, indent=0))
    return {"active": len(active), "excluded": len(excluded), "retrieval_eligible": len(eligible)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    root = os.environ.get("RAG_ROOT", "/srv/rfp")
    ap.add_argument("--registry", default=f"{root}/shared_data/processed/document_registry_v2/document_registry_v2.csv")
    ap.add_argument("--out-dir", default="data/gold")
    a = ap.parse_args(argv)
    stats = build(Path(a.registry), Path(a.out_dir))
    print(json.dumps(stats, ensure_ascii=False))
    print(f"→ {a.out_dir}/corpus_doc_ids.json, {a.out_dir}/excluded_doc_ids.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
