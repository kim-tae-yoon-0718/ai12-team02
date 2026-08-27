#!/usr/bin/env python3
"""
[연쇄 체크리스트 1-12] 표 비율 전수 계산 + 육안 표본 추출

측정만 한다. 판정하지 않는다.
- 유형 임계값("표 비율 몇 % 이상이면 표 중심")은 여기서 정하지 않는다.
  분포를 보고 사람이 정한다.
- 스캔본 여부(1-12 A'·B')는 여기서 다루지 않는다 — 김태윤 판정 대기.
- 뽑은 16건이 실제로 어떻게 생겼는지는 눈으로 본다(임무 ②).

출력: $RAG_ROOT/docs/measurement_1-12.{md,json}

규약 준수:
- 경로는 $RAG_ROOT 환경변수로 조립 (절대경로 하드코딩 금지)
- 측정 함수와 출력 함수를 분리 → 나중에 check_data.py로 승격
- shared_data/raw 는 읽기만 한다
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# 계수 규칙 (이 두 줄이 모든 숫자의 정의다. 바꾸면 앞의 값과 비교 불가)
# --------------------------------------------------------------------------
# - 글자 수 = HTML 태그 제거 + 엔티티 복원 + 공백문자 전부 제외한 문자 수
# - 표 글자 수 = 최상위 <table> 블록 안의 글자 수 (중첩 표는 바깥 표에 포함)
COUNT_RULE = "tags_stripped, entities_unescaped, whitespace_excluded"

# 분포를 보기 위한 구간. 유형 분류 임계값이 아니다 (임계값은 사람이 정한다).
BINS = [(0.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]

TAG_RE = re.compile(r"<[^>]+>")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
TABLE_TOKEN_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def visible_chars(fragment: str) -> int:
    """태그·엔티티·공백을 걷어낸 실제 글자 수."""
    text = MD_IMAGE_RE.sub("", fragment)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return len(WS_RE.sub("", text))


def top_level_tables(text: str) -> list[str]:
    """
    최상위 <table> 블록만 잘라낸다.

    ⚠️ 단순 non-greedy 정규식(<table>.*?</table>)을 쓰면 중첩 표에서
       첫 </table>에 끊겨 바깥 표를 과소 계산한다.
       1-13 실측에서 중첩 표가 확인됐으므로 깊이를 세며 훑는다.
    """
    blocks: list[str] = []
    depth = 0
    start = 0
    for m in TABLE_TOKEN_RE.finditer(text):
        opening = not m.group(0).lstrip("<").startswith("/")
        if opening:
            if depth == 0:
                start = m.start()
            depth += 1
        else:
            if depth > 0:
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:m.end()])
    if depth != 0:
        # 닫히지 않은 표 — 파싱 이상 신호. 호출부에서 경고로 올린다.
        blocks.append(text[start:])
    return blocks


def unbalanced_table_tags(text: str) -> bool:
    depth = 0
    for m in TABLE_TOKEN_RE.finditer(text):
        depth += 1 if not m.group(0).lstrip("<").startswith("/") else -1
        if depth < 0:
            return True
    return depth != 0


# --------------------------------------------------------------------------
# 원본 확장자 매핑 (raw 는 읽기 전용)
# --------------------------------------------------------------------------
def build_origin_map(raw_dir: Path) -> dict[str, str]:
    """md 파일 stem -> 원본 확장자. 파일명은 NFC로 정규화해 맞춘다."""
    origin: dict[str, str] = {}
    if not raw_dir.is_dir():
        return origin
    for p in raw_dir.iterdir():
        if p.is_file():
            key = unicodedata.normalize("NFC", p.stem)
            origin[key] = p.suffix.lower().lstrip(".")
    return origin


# --------------------------------------------------------------------------
# 측정
# --------------------------------------------------------------------------
def measure(md_dir: Path, raw_dir: Path) -> tuple[list[dict], list[str]]:
    origin = build_origin_map(raw_dir)
    rows: list[dict] = []
    warnings: list[str] = []

    files = sorted(md_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"[중단] md 파일이 없습니다: {md_dir}")

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.append(f"UTF-8 디코딩 실패: {path.name}")
            continue

        stem_nfc = unicodedata.normalize("NFC", path.stem)
        total = visible_chars(text)
        tables = top_level_tables(text)
        table_chars = sum(visible_chars(t) for t in tables)

        if unbalanced_table_tags(text):
            warnings.append(f"<table> 태그 짝이 안 맞음: {path.name}")
        if table_chars > total:
            warnings.append(f"표 글자 수 > 전체 글자 수 (계수 규칙 점검 필요): {path.name}")

        rows.append({
            "file": path.name,
            "origin_ext": origin.get(stem_nfc, "UNKNOWN"),
            "total_chars": total,
            "table_chars": table_chars,
            "table_ratio": round(table_chars / total, 4) if total else 0.0,
            "table_count": len(tables),
        })

    missing = [r["file"] for r in rows if r["origin_ext"] == "UNKNOWN"]
    if missing:
        warnings.append(
            f"원본 확장자를 못 찾은 md {len(missing)}건 — raw 경로/파일명 정규화 확인 필요: "
            + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        )
    return rows, warnings


# --------------------------------------------------------------------------
# C' 층화 표본 16건
# --------------------------------------------------------------------------
def pick_sample(rows: list[dict]) -> list[dict]:
    """
    PDF 전수 4 + 표비율 상위 3 + 표비율 하위 3 + 글자수 하위 3 + 중앙값 근처 3

    무작위 없음. 같은 입력이면 항상 같은 16건이 나온다(재현성).
    앞 층에서 이미 뽑힌 문서는 뒤 층에서 건너뛴다.
    """
    picked: dict[str, str] = {}   # file -> 뽑힌 이유
    order: list[dict] = []

    def take(row: dict, reason: str) -> bool:
        if row["file"] in picked:
            return False
        picked[row["file"]] = reason
        order.append({**row, "reason": reason})
        return True

    def take_n(candidates: list[dict], n: int, reason: str) -> None:
        taken = 0
        for row in candidates:
            if taken >= n:
                break
            if take(row, reason):
                taken += 1

    # 1) PDF 전수
    for row in [r for r in rows if r["origin_ext"] == "pdf"]:
        take(row, "PDF 전수")

    by_ratio = sorted(rows, key=lambda r: r["table_ratio"])
    by_chars = sorted(rows, key=lambda r: r["total_chars"])

    # 2) 표 비율 상위 / 하위
    take_n(list(reversed(by_ratio)), 3, "표 비율 상위")
    take_n(by_ratio, 3, "표 비율 하위")

    # 3) 글자 수 하위
    take_n(by_chars, 3, "글자 수 하위")

    # 4) 중앙값 근처 (글자 수 기준, 중앙 인덱스에서 바깥으로 번갈아 확장)
    mid = len(by_chars) // 2
    spiral: list[dict] = []
    for offset in range(len(by_chars)):
        for idx in (mid + offset, mid - offset) if offset else (mid,):
            if 0 <= idx < len(by_chars):
                spiral.append(by_chars[idx])
    take_n(spiral, 3, "중앙값 근처")

    return order


# --------------------------------------------------------------------------
# 출력
# --------------------------------------------------------------------------
def pct(values: list[float], q: float) -> float:
    """선형보간 없는 최근접 순위 백분위."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def report(rows: list[dict], warnings: list[str], md_dir: Path) -> None:
    ratios = [r["table_ratio"] for r in rows]
    print("=" * 68)
    print("[1-12 D] 표 비율 전수 계산")
    print(f"  대상       : {md_dir}")
    print(f"  문서       : {len(rows)}건 "
          f"(hwp {sum(1 for r in rows if r['origin_ext'] == 'hwp')} / "
          f"pdf {sum(1 for r in rows if r['origin_ext'] == 'pdf')} / "
          f"미상 {sum(1 for r in rows if r['origin_ext'] == 'UNKNOWN')})")
    print(f"  계수 규칙  : {COUNT_RULE}")
    print("-" * 68)
    print("  표 비율 분포")
    for label, q in [("min", 0.0), ("p25", .25), ("median", .5),
                     ("p75", .75), ("p95", .95), ("max", 1.0)]:
        print(f"    {label:>6} : {pct(ratios, q):6.1%}")
    print("-" * 68)
    print("  구간별 문서 수 (임계값은 아직 미확정 — 분포를 보고 정한다)")
    for lo, hi in BINS:
        n = sum(1 for r in ratios if lo <= r < hi)
        bar = "█" * round(n / max(1, len(rows)) * 40)
        print(f"    {lo:.0%}~{min(hi,1.0):.0%} : {n:3d}건 {bar}")
    print("-" * 68)
    print("  표 개수: 합계 "
          f"{sum(r['table_count'] for r in rows)}개 · "
          f"문서당 중앙값 {int(pct([float(r['table_count']) for r in rows], .5))}개")

    if warnings:
        print("-" * 68)
        print(f"  ⚠️ 경고 {len(warnings)}건")
        for w in warnings[:20]:
            print(f"    - {w}")
        if len(warnings) > 20:
            print(f"    ... 외 {len(warnings) - 20}건")
    print("=" * 68)


def report_sample(sample: list[dict]) -> None:
    print()
    print("=" * 68)
    print(f"[1-12 C'] 육안 표본 {len(sample)}건 — 무작위 없음, 재현 가능")
    print("=" * 68)
    print(f"{'뽑은 이유':<14} {'표비율':>7} {'글자수':>9}  파일")
    print("-" * 68)
    for row in sample:
        print(f"{row['reason']:<14} {row['table_ratio']:>7.1%} "
              f"{row['total_chars']:>9,}  {row['file']}")
    print("-" * 68)
    print("  훑을 때 적을 것 (임무 ②) — 반복 머리글·바닥글 / 목차 덩어리 /")
    print("  표 밖으로 흘러나온 조각 / 그 밖에 '이게 왜 여기 있지' 싶은 것")
    print("  ⚠️ 유형 분류가 맞는지는 보지 않는다 (임무 ①은 ⓕ에서 제외됨)")
    print("=" * 68)


def build_summary(rows: list[dict], sample: list[dict],
                  warnings: list[str], md_dir: Path) -> dict:
    """측정 결과를 그대로 담는다. 판정·분류는 넣지 않는다."""
    ratios = [r["table_ratio"] for r in rows]
    counts = [float(r["table_count"]) for r in rows]
    return {
        "item": "1-12",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "target": str(md_dir),
        "count_rule": COUNT_RULE,
        "documents": {
            "total": len(rows),
            "hwp": sum(1 for r in rows if r["origin_ext"] == "hwp"),
            "pdf": sum(1 for r in rows if r["origin_ext"] == "pdf"),
            "unknown": sum(1 for r in rows if r["origin_ext"] == "UNKNOWN"),
        },
        "table_ratio": {
            "min": pct(ratios, 0.0), "p25": pct(ratios, .25),
            "median": pct(ratios, .5), "p75": pct(ratios, .75),
            "p95": pct(ratios, .95), "max": pct(ratios, 1.0),
        },
        "table_ratio_bins": {
            f"{int(lo*100)}-{int(min(hi,1.0)*100)}%":
                sum(1 for r in ratios if lo <= r < hi)
            for lo, hi in BINS
        },
        "table_count": {
            "sum": sum(r["table_count"] for r in rows),
            "median_per_doc": int(pct(counts, .5)),
        },
        "sample": sample,
        "warnings": warnings,
        "per_document": rows,
    }


def write_outputs(summary: dict, docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = docs_dir / "measurement_1-12.json"
    md_path = docs_dir / "measurement_1-12.md"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 측정 결과 — 연쇄 체크리스트 1-12 (표 비율 · 육안 표본)",
        "",
        f"- 측정 시각: {summary['measured_at']}",
        f"- 대상: `{summary['target']}`",
        f"- 계수 규칙: `{summary['count_rule']}`",
        f"- 문서: {summary['documents']['total']}건 "
        f"(hwp {summary['documents']['hwp']} / pdf {summary['documents']['pdf']}"
        + (f" / 미상 {summary['documents']['unknown']}"
           if summary['documents']['unknown'] else "") + ")",
        "",
        "> ⚠️ 유형 임계값은 이 문서에 없다. 분포를 보고 사람이 정한다.",
        "",
        "## 표 비율 분포",
        "",
        "| 지점 | 값 |", "| --- | --- |",
    ]
    for k, v in summary["table_ratio"].items():
        lines.append(f"| {k} | {v:.1%} |")
    lines += ["", "## 구간별 문서 수", "", "| 구간 | 문서 수 |", "| --- | --- |"]
    for k, v in summary["table_ratio_bins"].items():
        lines.append(f"| {k} | {v}건 |")
    lines += [
        "",
        f"표 개수 — 합계 {summary['table_count']['sum']}개 · "
        f"문서당 중앙값 {summary['table_count']['median_per_doc']}개",
        "",
        f"## 육안 표본 {len(summary['sample'])}건",
        "",
        "무작위 없음. 같은 입력이면 항상 같은 표본이 나온다.",
        "",
        "| 뽑은 이유 | 표 비율 | 글자 수 | 파일 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in summary["sample"]:
        lines.append(f"| {row['reason']} | {row['table_ratio']:.1%} | "
                     f"{row['total_chars']:,} | {row['file']} |")
    lines += [
        "",
        "훑을 때 적을 것 — 반복 머리글·바닥글 / 목차 덩어리 / "
        "표 밖으로 흘러나온 조각 / 그 밖에 이상한 것",
        "",
        "⚠️ 유형 분류가 맞는지는 보지 않는다 (임무 ① 제외).",
        "",
    ]
    if summary["warnings"]:
        lines += [f"## ⚠️ 경고 {len(summary['warnings'])}건", ""]
        lines += [f"- {w}" for w in summary["warnings"]]
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [md_path, json_path]


def main() -> int:
    ap = argparse.ArgumentParser(description="1-12 표 비율 전수 계산 + 육안 표본 추출")
    ap.add_argument("--md-dir", default=None,
                    help="기본 $RAG_ROOT/shared_data/interim/md")
    ap.add_argument("--raw-dir", default=None,
                    help="기본 $RAG_ROOT/shared_data/raw/files (읽기 전용)")
    ap.add_argument("--docs-dir", default=None,
                    help="기본 $RAG_ROOT/docs")
    ap.add_argument("--csv", default=None,
                    help="문서별 결과를 CSV로도 남길 경로 (선택)")
    ap.add_argument("--dry-run", action="store_true",
                    help="파일을 쓰지 않고 화면 출력만")
    args = ap.parse_args()

    root_env = os.environ.get("RAG_ROOT")
    needs_root = (args.md_dir is None or args.raw_dir is None
                  or (args.docs_dir is None and not args.dry_run))
    if not root_env and needs_root:
        print("[중단] RAG_ROOT 가 설정돼 있지 않습니다.", file=sys.stderr)
        print("       export RAG_ROOT=/srv/rfp  후 다시 실행하거나", file=sys.stderr)
        print("       --md-dir / --raw-dir 를 직접 지정하세요.", file=sys.stderr)
        return 2

    root = Path(root_env) if root_env else None
    md_dir = Path(args.md_dir) if args.md_dir else root / "shared_data" / "interim" / "md"
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / "shared_data" / "raw" / "files"

    if not md_dir.is_dir():
        print(f"[중단] md 폴더를 찾을 수 없습니다: {md_dir}", file=sys.stderr)
        return 2
    if not raw_dir.is_dir():
        print(f"[경고] raw 폴더를 찾을 수 없습니다: {raw_dir}", file=sys.stderr)
        print("       원본 확장자를 못 읽으므로 PDF 전수 층이 비게 됩니다.", file=sys.stderr)

    rows, warnings = measure(md_dir, raw_dir)
    sample = pick_sample(rows)

    report(rows, warnings, md_dir)
    report_sample(sample)

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았습니다.")
        return 0

    docs_dir = Path(args.docs_dir) if args.docs_dir else root / "docs"
    written = write_outputs(build_summary(rows, sample, warnings, md_dir), docs_dir)
    for p in written:
        print(f"[저장] {p}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[저장] {csv_path} ({len(rows)}행)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
