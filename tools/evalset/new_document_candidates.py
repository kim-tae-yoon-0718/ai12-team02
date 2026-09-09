#!/usr/bin/env python3
"""새 문서 1건에 대한 12필드 추출 후보 미리보기 — 사람이 정리하기 전 1차 스크리닝.

select/extract 라우트는 LLM이 아니라 extraction_table_v5(구조화 추출표)를 코드로
조회한다. 그 표는 지금 공식 100건 고정이고(⭐ 확정), 새 문서를 넣으려면 결국
사람이 문서를 읽고 12필드 값·근거 위치를 정리해 `semantic_decisions_v4_inherited.csv`와
같은 형식의 결정 파일을 만들어야 한다(v5는 v4의 결정 계약을 그대로 이어받았다).

이 스크립트는 그 사람 작업을 대신하지 않는다. 대신 `build_extraction_table.py`가
100건에 이미 쓰고 있는 같은 규칙 엔진(find_hits/decide/discover_aliases)을 새
문서 1건에 돌려서 "규칙이 어디서 무엇을 찾았는지" 후보를 먼저 보여준다 —
사람은 문서 전체를 처음부터 읽는 대신 이 후보를 확인·수정만 하면 된다.

⚠️ 공식 산출물(extraction_table_v5, semantic_decisions_v4_inherited.csv, document_registry_v2)은
   전혀 건드리지 않는다. 결과는 항상 --out 폴더의 별도 파일로만 나간다.

사용:
    python3 new_document_candidates.py --md path/to/new_doc.md \
        --document-id RFP-NEW-001 [--announce-date 2026-09-01] [--out out_dir]

산출물(--out 폴더):
    <document_id>_candidate_decisions.csv   규칙 후보 + 사람이 채울 빈 검토 칸
    <document_id>_alias_candidates.csv      기존 12필드 목록 밖 표현 후보(있을 때만)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extraction_rules_v3 as R          # noqa: E402  (순수 규칙, 100건 표와 동일 모듈)
import build_extraction_table as B       # noqa: E402  (규칙 엔진 재사용, 수정하지 않음)

# 사람이 검토 후 채우는 칸 — semantic_decisions_v4_inherited.csv와 같은 계약을 쓴다.
# 나중에 이 문서를 정식으로 편입하기로 하면 이 칸들만 채워서 그대로 결정
# 파일로 합칠 수 있게 하기 위함(형식을 새로 만들지 않는다).
#
# 이 중 reviewer/source_review_method는 "누가 무슨 방법으로 검토하나"라는
# 세션 전체에 동일한 값(예: 기존 semantic_decisions_v4_inherited.csv의 "태윤님 측 LLM" /
# "태윤님 측 LLM 원문 검수")이라 --reviewer/--review-method로 한 번에 채운다.
# source_review_completed·source_context_note·discarded_*는 실제로 그 필드를
# 검토했는지에 따라 필드마다 달라야 하므로 항상 사람이 직접 채운다(자동 채움 없음).
REVIEWER_IDENTITY_FIELDS = ["reviewer", "source_review_method"]
REVIEW_ONLY_FIELDS = ["source_review_completed", "source_context_note",
                      "discarded_candidates", "discarded_reason"]


def default_org_allowlist(registry_dir: Path | None) -> set[str]:
    """기존 등록부에서 발주기관 이름을 모은다(PII 오탐 방지용, 없으면 빈 집합)."""
    if registry_dir is None:
        return set()
    path = registry_dir / "document_registry_v2.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["source_filename_nfc"].split("_")[0].strip()
                for row in csv.DictReader(fh) if row.get("source_filename_nfc")}


def build_candidate_rows(document_id: str, filename: str, text: str,
                         announce_date: str, aliases_out: list,
                         reviewer: str = "", review_method: str = "") -> list[dict]:
    """process_document()와 같은 흐름을 새 문서 1건(경로 무관)에 대해 돌린다."""
    lines = text.split("\n")
    section_index = B.build_section_index(lines)
    att = B.attachment_start({}, len(lines), lines)   # sidecar 없음 — 제목 기반으로만 판단
    front_limit = max(40, int(len(lines) * B.FRONT_RATIO))
    reg_row = {"document_id": document_id, "source_filename_nfc": filename}
    aliases_out.extend(B.discover_aliases(lines, reg_row, att, front_limit))

    rows = []
    for field in R.FIELDS:
        if field == "공고일":
            if announce_date:
                out = {"status": "value_present", "answer_raw": announce_date,
                       "answer_normalized": announce_date.split(" ")[0],
                       "matched_expression": "공고일(입력값)",
                       "rep": {"source_type": "metadata_csv", "line": 0,
                               "heading": "--announce-date 입력값", "value": announce_date},
                       "additional": [], "confidence": "rule_high", "review_reason": ""}
            else:
                out = {"status": "field_absent", "answer_raw": "", "answer_normalized": "",
                       "matched_expression": "", "rep": None, "additional": [],
                       "confidence": "rule_high",
                       "review_reason": "--announce-date 입력 없음 — 원본 공고문에서 확인 필요"}
        else:
            hits = B.find_hits(lines, field, att, front_limit, section_index)
            out = B.decide(field, hits, text)

        rep = out.get("rep")
        rep_type = rep["source_type"] if rep else ""
        rep_loc = ""
        if rep:
            rep_loc = B.jdump({"document_id": document_id, "file": filename,
                               **B.location_payload(rep)})
        excerpt = R.redact((rep or {}).get("value", ""))[:B.MAX_EXCERPT_CHARS]

        row = {
            "document_id": document_id, "field_name": field, "status": out["status"],
            "answer_raw": R.redact(out["answer_raw"]),
            "answer_normalized": R.redact(out["answer_normalized"]),
            "matched_expression": out["matched_expression"],
            "representative_source_type": rep_type,
            "representative_location": rep_loc,
            "source_excerpt_redacted": excerpt,
            "additional_locations": B.jdump(out["additional"]),
            "confidence": out["confidence"], "review_reason": out["review_reason"],
            "reviewer": reviewer, "source_review_method": review_method,
            **{k: "" for k in REVIEW_ONLY_FIELDS},
        }
        row["representative_location"] = B.jdump(B.enrich_saved_location(
            row["representative_location"], section_index, document_id, filename,
            row["representative_source_type"]))
        row["additional_locations"] = B.jdump(B.enrich_saved_location(
            row["additional_locations"], section_index, document_id, filename))
        rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    from collections import Counter
    print(json.dumps({
        "status": dict(Counter(r["status"] for r in rows).most_common()),
        "review_required": [r["field_name"] for r in rows if r["status"] == "review_required"],
        "field_absent": [r["field_name"] for r in rows if r["status"] == "field_absent"],
    }, ensure_ascii=False, indent=2))
    print()
    for r in rows:
        value = (r["answer_raw"] or "").replace("\n", " ")[:80]
        print(f"[{r['status']:16}] {r['field_name']:20} {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description="새 문서 1건의 12필드 추출 후보 미리보기")
    ap.add_argument("--md", required=True, help="전처리된 문서 본문(.md) 경로")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--filename", default=None,
                    help="원본 파일명(기본: --md 파일 stem)")
    ap.add_argument("--announce-date", default=None, help="공고일(YYYY-MM-DD). 원본 공고문의 "
                    "단일 공식 출처이므로 규칙이 자동으로 찾지 않는다 — 사람이 넣는다.")
    ap.add_argument("--out", default=None,
                    help="결과 폴더(기본: --md와 같은 폴더의 <document_id>_candidates/)")
    ap.add_argument("--registry-dir", default=None,
                    help="발주기관 이름 목록을 읽어올 기존 등록부 폴더(선택, PII 오탐 감소용)")
    ap.add_argument("--reviewer", default="",
                    help="검토 주체 — 기본값은 빈 문자열이다. 이 스크립트가 뽑은 후보는 "
                    "아직 사람이 확인하지 않았으므로, 검토자를 미리 채워두면(예: "
                    "'태윤님 측 LLM') 검토를 안 거친 자료에 검토 기록이 남는 것처럼 "
                    "보인다(회귀: 코드리뷰 지적). 실제로 검토를 마친 뒤에만 이 옵션으로 채운다.")
    ap.add_argument("--review-method", default="",
                    help="검토 방법 — 위와 같은 이유로 기본값은 빈 문자열이다.")
    args = ap.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        raise SystemExit(f"❌ --md 파일이 없습니다: {md_path}")
    text = md_path.read_text(encoding="utf-8")
    filename = args.filename or md_path.stem
    out_dir = Path(args.out) if args.out else md_path.parent / f"{args.document_id}_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_dir = Path(args.registry_dir) if args.registry_dir else None
    R.set_org_allowlist(default_org_allowlist(registry_dir) | {filename.split("_")[0].strip()})

    aliases: list = []
    rows = build_candidate_rows(args.document_id, filename, text, args.announce_date, aliases,
                                reviewer=args.reviewer, review_method=args.review_method)

    decisions_path = out_dir / f"{args.document_id}_candidate_decisions.csv"
    with decisions_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=B.DECISION_FIELDS, quoting=csv.QUOTE_ALL,
                           lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in B.DECISION_FIELDS} for r in rows)
    print(f"후보 결정 파일(사람이 검토·확정): {decisions_path}")

    if aliases:
        afields = ["document_id", "field_name", "matched_expression", "source_type",
                   "location", "heading", "basis", "confidence"]
        alias_path = out_dir / f"{args.document_id}_alias_candidates.csv"
        with alias_path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=afields, quoting=csv.QUOTE_ALL,
                               lineterminator="\n")
            w.writeheader()
            w.writerows({k: a.get(k, "") for k in afields} for a in aliases)
        print(f"목록 밖 표현 후보: {alias_path}")

    print()
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
