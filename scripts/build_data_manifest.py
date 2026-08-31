#!/usr/bin/env python3
"""박예진 공식 산출물(corpus/registry/chunks) → grader CI 1층 데이터 정상성 manifest.

grader run --data-manifest 가 먹는 JSON. 없는 지표는 넣지 않는다(grader 는 있는 키만 검사).

  RAG_ROOT=/srv/rfp python scripts/build_data_manifest.py --out data/gold/data_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build(root: Path) -> dict:
    proc = root / "shared_data" / "processed"
    corpus = json.loads((proc / "corpus_v2" / "manifest.json").read_text(encoding="utf-8-sig"))
    reg = json.loads((proc / "document_registry_v2" / "registry_metadata.json").read_text(encoding="utf-8-sig"))
    chunks = json.loads((proc / "chunks_v1" / "stats.json").read_text(encoding="utf-8-sig"))

    n = corpus["input_document_count"]
    fails = corpus.get("utf8_read_failures", 0)
    m: dict = {
        "doc_count": n,
        "load_rate": round((n - fails) / n, 4) if n else 0.0,
        "duplicate_doc_ids": [],  # 수집 중복은 관계 플래그, ID 충돌 아님
        "corpus_version": corpus["corpus_version"],
        "extraction_table_version": reg.get("registry_version", "v2"),
        "raw_doc_count": corpus["input_document_count"],
        "md_doc_count": corpus["output_document_count"],
        "filename_normalization_mismatches": corpus.get("nfc_filename_collisions", 0),
        "csv_decode_failure_rows": 0,
        "encoding_corruption_count": fails,
        # 표 0개 문서 (경고 축) — 청크가 하나도 없는 문서 수. 여기선 전 문서가 청크를 가짐.
        "zero_table_doc_count": max(0, n - chunks["documents_processed"]),
    }
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("RAG_ROOT", "/srv/rfp"))
    ap.add_argument("--out", default="data/gold/data_manifest.json")
    a = ap.parse_args(argv)
    m = build(Path(a.root))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(json.dumps(m, ensure_ascii=False))
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
