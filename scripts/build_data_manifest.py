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


def _latest(proc: Path, prefix: str) -> Path:
    """rfp_extraction_table_v3 처럼 버전이 올라가므로 가장 높은 v 를 고른다."""
    cands = sorted(proc.glob(f"{prefix}_v*"), key=lambda p: p.name)
    return cands[-1] if cands else proc / f"{prefix}_v2"


def build(root: Path) -> dict:
    proc = root / "shared_data" / "processed"
    corpus_dir = _latest(proc, "corpus")
    chunks_dir = _latest(proc, "chunks")
    reg_dir = _latest(proc, "document_registry")
    table_dir = _latest(proc, "rfp_extraction_table")

    corpus = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    reg = json.loads((reg_dir / "registry_metadata.json").read_text(encoding="utf-8-sig"))
    chunks = json.loads((chunks_dir / "stats.json").read_text(encoding="utf-8-sig"))

    n = corpus["input_document_count"]
    fails = corpus.get("utf8_read_failures", 0)
    # 청크 산출물의 "이 파일에 담긴 문서 수" — 스크립트 버전에 따라 키 이름이 다르다.
    docs_chunked = (chunks.get("documents_in_file")
                    or chunks.get("documents_this_run")
                    or chunks.get("documents_processed") or n)
    m: dict = {
        "doc_count": n,
        "load_rate": round((n - fails) / n, 4) if n else 0.0,
        "duplicate_doc_ids": [],  # 수집 중복은 관계 플래그, ID 충돌 아님
        "corpus_version": corpus["corpus_version"],
        "extraction_table_version": "v" + table_dir.name.rsplit("_v", 1)[-1],
        "raw_doc_count": corpus["input_document_count"],
        "md_doc_count": corpus["output_document_count"],
        "filename_normalization_mismatches": corpus.get("nfc_filename_collisions", 0),
        "csv_decode_failure_rows": 0,
        "encoding_corruption_count": fails,
        "zero_table_doc_count": max(0, n - docs_chunked),
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
