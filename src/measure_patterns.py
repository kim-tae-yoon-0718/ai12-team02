#!/usr/bin/env python3
"""
[연쇄 체크리스트 1-12] 육안 발견 패턴 전수 적용

측정만 한다. 판정하지 않는다.
- "몇 건에서 몇 번 나오는가"만 센다. 지울지 말지는 1-17에서 정한다.
- 육안 16건에서 나온 후보를 100건에 적용해 실제 분포를 확인한다.
- 육안은 후보를 만들고, 숫자는 여기서 낸다.

⚠️ 이 스크립트의 숫자는 정규식이 잡은 것이다. 육안 판정과 다를 수 있고,
   오탐 가능성이 있는 패턴은 sample을 함께 출력하니 눈으로 확인해야 한다.
   (1-13에서 자모분리 33건이 원문 마스킹 오탐이었던 선례)

출력: $RAG_ROOT/docs/measurement_1-12_patterns.{md,json}

규약 준수:
- 경로는 $RAG_ROOT 환경변수로 조립 (절대경로 하드코딩 금지)
- 측정 함수와 출력 함수를 분리
- shared_data 는 읽기만 한다
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# 발견 번호는 1-12 육안 훑기 기록과 맞춘다.
# confidence: high  = 패턴이 명확, 오탐 가능성 낮음
#             mid   = 오탐 가능, sample 확인 필요
#             low   = 후보 탐지용. 숫자를 그대로 믿지 말 것
# --------------------------------------------------------------------------
PATTERNS = [
    # ── 35: 장·절 제목이 표 한 줄로 (좌표 형식 복구 열쇠) ────────────────
    dict(
        id="35a", name="장 제목이 표 한 줄 (로마숫자)",
        note="| Ⅰ |  | 사업 개요 |  형태. 1-13-1 ⑤ 좌표 재료가 표 안에 갇힘",
        confidence="high",
        regex=r"^\s*\|\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]{1,5}\s*\|\s*\|\s*[^|\n]{2,40}\s*\|\s*$",
    ),
    dict(
        id="35b", name="절 제목이 표 한 줄 (아라비아 숫자)",
        note="| 1 |  | 사업개요 |  형태",
        confidence="mid",
        regex=r"^\s*\|\s*\d{1,2}\s*\|\s*\|\s*[^|\n]{2,40}\s*\|\s*$",
    ),

    # ── 6+7: 별첨/서식 구간 ──────────────────────────────────────────────
    dict(
        id="7", name="별첨/서식 구간 경계",
        note="이 표기 이후가 제출 양식. 빈 서식이 여기 몰려 있다",
        confidence="high",
        regex=r"(?:^|\n)[^\n]{0,20}(?:\[\s*(?:붙임|별지|별표|서식|별첨)[^\]]{0,20}\]"
              r"|#+\s*별첨|#+\s*[^\n]{0,15}관련\s*서식|【\s*별첨[^】]{0,10}】"
              r"|【\s*붙임[^】]{0,10}】)",
    ),
    dict(
        id="6a", name="완전히 빈 표 행 (HTML)",
        note="<td></td> 만으로 이루어진 행. 데이터 0행 서식",
        confidence="high",
        regex=r"<tr>(?:\s*<td[^>]*>\s*</td>\s*)+</tr>",
    ),
    dict(
        id="6b", name="완전히 빈 표 행 (마크다운)",
        note="| | | | 형태",
        confidence="mid",
        regex=r"^\s*\|(?:\s*\|)+\s*$",
    ),
    dict(
        id="6c", name="자리표시자 텍스트",
        note="년월일 · 20XX년 · 년 개월 · (한자) 등. 값처럼 보이나 값이 아님",
        confidence="mid",
        regex=r"(?:년\s*월\s*일|20\s*[XxＸ]{2}\s*년|[XxＸ]{2,}\s*월|년\s+개월|\(한자\))",
    ),

    # ── 2: 목차 잔존 ─────────────────────────────────────────────────────
    dict(
        id="2a", name="목차 점선 리더",
        note="··········· 또는 ----------- 로 남은 목차 줄",
        confidence="high",
        regex=r"(?:[·․‥…]{5,}|-{10,}|\.{8,})",
    ),
    dict(
        id="2b", name="목차 표기",
        note="'목 차' 라는 말 자체. 몇 번 나오는지도 본다(문서당 2회 사례 있음)",
        confidence="high",
        regex=r"목\s*차",
    ),

    # ── 14: 단어 중간 공백 ───────────────────────────────────────────────
    dict(
        id="14", name="단어 중간 공백 (자간 벌림)",
        note="사 업 명 · 구 분 · 성 명 처럼 한 글자씩 띄어짐. 검색이 안 됨",
        confidence="mid",
        regex=r"(?<![가-힣])(?:[가-힣]\s){2,}[가-힣](?![가-힣])",
    ),

    # ── 5: 제목 레벨 무의미 ──────────────────────────────────────────────
    dict(
        id="5a", name="# 제목이 날짜만",
        note="# 2024. 8.  처럼 날짜가 제목 레벨 1",
        confidence="high",
        regex=r"^#{1,3}\s*\d{4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]?\s*$",
    ),
    dict(
        id="5b", name="# 제목이 5자 이하",
        note="# 제 / # 안요 / # I  처럼 조각난 제목",
        confidence="mid",
        regex=r"^#{1,6}\s*\S{1,5}\s*$",
    ),
    dict(
        id="5c", name="제목 안에 HTML 태그",
        note="# <u>합의각서</u> 형태",
        confidence="high",
        regex=r"^#{1,6}\s*[^\n]*<[a-zA-Z][^>]*>",
    ),

    # ── 24: 빈 셀 위치가 정보 (1-17 반례) ────────────────────────────────
    dict(
        id="24", name="일정·조직 표시 기호",
        note="● ■ ★ ▲ ▶ ○ ◎ 가 셀에 단독으로. 위치 자체가 값 → 빈 셀 건너뛰기 금지",
        confidence="mid",
        regex=r"<td[^>]*>\s*[●■★▲▶○◎◇◆□]\s*</td>",
    ),

    # ── 32: 요구사항 코드 ────────────────────────────────────────────────
    dict(
        id="32", name="요구사항 고유번호",
        note="SFR-001 등. 0패딩 누락·표기 불일치를 보려면 뒤의 코드별 집계를 볼 것",
        confidence="high",
        regex=r"\b([A-Z]{2,4})-(\d{1,4})\b",
    ),

    # ── 26: LaTeX 수식 (제거 금지) ───────────────────────────────────────
    dict(
        id="26", name="LaTeX 수식",
        note="가격평가 산출식 등. ⚠️ 제거 금지 — 값이 여기 담김",
        confidence="high",
        regex=r"\\(?:frac|times|left|right|bullet|div)\b",
    ),

    # ── 10: 체크박스 (제거 금지) ─────────────────────────────────────────
    dict(
        id="10", name="체크박스 기호",
        note="□ ■ □∨ 등. ⚠️ ∨ 하나가 적용/미적용을 뒤집음 — 제거 금지",
        confidence="mid",
        regex=r"[□■☐☑]\s*[∨✓v]?",
    ),

    # ── 15/29: 개인정보 (1-19) ───────────────────────────────────────────
    dict(
        id="29a", name="이메일 주소",
        note="담당자 연락처. 1-12-1 ⑤ 필드이나 공개 시 주의 → 1-19",
        confidence="high",
        regex=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    ),
    dict(
        id="29b", name="전화번호",
        note="같은 이유로 1-19",
        confidence="mid",
        regex=r"0\d{1,2}[)\-\s]\s*\d{3,4}\s*-\s*\d{4}",
    ),

    # ── 19: 인쇄 규격 (명백한 제거 대상) ─────────────────────────────────
    dict(
        id="19", name="인쇄 규격·용지 표기",
        note="210mm×297mm(백상지 80g/㎡) · (뒤쪽). 값이 아님",
        confidence="high",
        regex=r"(?:\d{2,3}\s*[mm㎜]{1,2}\s*[×xX]\s*\d{2,3}\s*[mm㎜]{1,2}|백상지|\(뒤쪽\))",
    ),

    # ── 4: 페이지 번호 잔존 ──────────────────────────────────────────────
    dict(
        id="4", name="페이지 번호 잔존",
        note="-8- · 페이지:4/19 형태",
        confidence="mid",
        regex=r"(?:^|\|)\s*-\s*\d{1,3}\s*-\s*(?:\||$)|페이지\s*:\s*\d+\s*/\s*\d+",
    ),

    # ── 25: 이스케이프 문자 (1-14 경계) ──────────────────────────────────
    dict(
        id="25", name="백슬래시 이스케이프",
        note="\\* · \\~ 등. 1-14(문자 수준) 경계 — 여기서는 건수만",
        confidence="high",
        regex=r"\\[*~_\[\]()#+\-.!]",
    ),

    # ── 27: 볼드가 제목 대신 ─────────────────────────────────────────────
    dict(
        id="27", name="볼드만으로 된 줄",
        note="**1. 항목** 처럼 제목인데 # 이 아님 → 좌표에 안 잡힘",
        confidence="mid",
        regex=r"^\s*\*\*[^*\n]{2,60}\*\*\s*$",
    ),

    # ── 13: 표 값 뒤섞임 (PDF 특유) ──────────────────────────────────────
    dict(
        id="13", name="조각난 표 셀 (PDF 특유 의심)",
        note="⚠️ 오탐 많음. 한 줄에 2자 이하 셀이 3개 이상. sample 필수 확인",
        confidence="low",
        regex=r"^\s*\|(?:\s*[^|\n]{0,2}\s*\|){3,}\s*$",
    ),
]


# --------------------------------------------------------------------------
# 측정
# --------------------------------------------------------------------------
def strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def measure(md_dir: Path, sample_n: int) -> tuple[list[dict], dict, list[str]]:
    files = sorted(md_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"[중단] md 파일이 없습니다: {md_dir}")

    compiled = [(p, re.compile(p["regex"], re.MULTILINE)) for p in PATTERNS]
    results = {p["id"]: dict(docs=0, hits=0, samples=[]) for p in PATTERNS}
    req_codes: dict[str, set[str]] = {}   # 32번용: 접두어 -> 번호 표기 집합
    warnings: list[str] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.append(f"UTF-8 디코딩 실패: {path.name}")
            continue

        for spec, rx in compiled:
            found = list(rx.finditer(text))
            if not found:
                continue
            r = results[spec["id"]]
            r["docs"] += 1
            r["hits"] += len(found)
            if len(r["samples"]) < sample_n:
                snippet = found[0].group(0)
                snippet = re.sub(r"\s+", " ", snippet)[:110]
                r["samples"].append({"file": path.name, "text": snippet})

            # 32번은 코드 표기 자체를 모은다
            if spec["id"] == "32":
                for m in found:
                    req_codes.setdefault(m.group(1), set()).add(m.group(2))

    # 요구사항 코드 표기 불일치 판정 (0패딩 혼용 여부)
    code_report = []
    for prefix, nums in sorted(req_codes.items()):
        widths = {len(n) for n in nums}
        code_report.append({
            "prefix": prefix,
            "distinct_numbers": len(nums),
            "digit_widths": sorted(widths),
            "mixed_padding": len(widths) > 1,
        })

    rows = []
    for spec in PATTERNS:
        r = results[spec["id"]]
        rows.append({
            "id": spec["id"], "name": spec["name"], "note": spec["note"],
            "confidence": spec["confidence"],
            "documents": r["docs"], "total_hits": r["hits"],
            "doc_ratio": round(r["docs"] / len(files), 3),
            "samples": r["samples"],
        })

    return rows, {"total_files": len(files), "req_codes": code_report}, warnings


# --------------------------------------------------------------------------
# 출력
# --------------------------------------------------------------------------
def write_outputs(rows: list[dict], meta: dict, warnings: list[str],
                  md_dir: Path, docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "item": "1-12",
        "subject": "육안 발견 패턴 전수 적용",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "target": str(md_dir),
        "total_files": meta["total_files"],
        "patterns": rows,
        "requirement_codes": meta["req_codes"],
        "warnings": warnings,
    }
    json_path = docs_dir / "measurement_1-12_patterns.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    n = meta["total_files"]
    lines = [
        "# 측정 결과 — 1-12 육안 발견 패턴 전수 적용",
        "",
        f"- 측정 시각: {summary['measured_at']}",
        f"- 대상: `{md_dir}` · {n}건",
        "",
        "> ⚠️ 이 숫자는 정규식이 잡은 것이다. 지울지 말지는 1-17에서 정한다.",
        "> confidence가 mid/low인 항목은 sample을 눈으로 확인해야 한다.",
        "",
        "## 패턴별 출현",
        "",
        "| ID | 패턴 | 신뢰도 | 문서 | 비율 | 총 출현 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: -x["documents"]):
        lines.append(f"| {r['id']} | {r['name']} | {r['confidence']} | "
                     f"{r['documents']} | {r['doc_ratio']:.0%} | {r['total_hits']:,} |")

    lines += ["", "## 패턴별 상세와 표본", ""]
    for r in sorted(rows, key=lambda x: -x["documents"]):
        lines += [
            f"### {r['id']} — {r['name']}",
            "",
            f"- {r['note']}",
            f"- **{r['documents']}건 / {n}건 ({r['doc_ratio']:.0%}) · "
            f"총 {r['total_hits']:,}회** · 신뢰도 `{r['confidence']}`",
            "",
        ]
        if r["samples"]:
            lines.append("표본:")
            lines.append("")
            for s in r["samples"]:
                lines.append(f"- `{s['text']}`  ← {s['file'][:40]}")
            lines.append("")

    codes = meta["req_codes"]
    if codes:
        mixed = [c for c in codes if c["mixed_padding"]]
        lines += [
            "## 32 — 요구사항 코드 표기",
            "",
            f"접두어 {len(codes)}종. **0패딩이 혼용된 접두어 {len(mixed)}종**",
            "",
            "| 접두어 | 서로 다른 번호 | 자릿수 | 패딩 혼용 |",
            "| --- | ---: | --- | --- |",
        ]
        for c in codes:
            flag = "⚠️ 예" if c["mixed_padding"] else "아니오"
            lines.append(f"| {c['prefix']} | {c['distinct_numbers']} | "
                         f"{c['digit_widths']} | {flag} |")
        lines.append("")

    if warnings:
        lines += [f"## ⚠️ 경고 {len(warnings)}건", ""]
        lines += [f"- {w}" for w in warnings] + [""]

    md_path = docs_dir / "measurement_1-12_patterns.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [md_path, json_path]


def report(rows: list[dict], meta: dict) -> None:
    n = meta["total_files"]
    print("=" * 74)
    print(f"[1-12] 육안 발견 패턴 전수 적용 — {n}건")
    print("=" * 74)
    print(f"{'ID':<5}{'신뢰':<6}{'문서':>5}{'비율':>7}{'출현':>9}  패턴")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: -x["documents"]):
        print(f"{r['id']:<5}{r['confidence']:<6}{r['documents']:>5}"
              f"{r['doc_ratio']:>7.0%}{r['total_hits']:>9,}  {r['name']}")
    print("-" * 74)
    mixed = [c for c in meta["req_codes"] if c["mixed_padding"]]
    if meta["req_codes"]:
        print(f"요구사항 코드 접두어 {len(meta['req_codes'])}종 · "
              f"0패딩 혼용 {len(mixed)}종"
              + (f" ({', '.join(c['prefix'] for c in mixed[:8])})" if mixed else ""))
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description="1-12 육안 발견 패턴 전수 적용")
    ap.add_argument("--md-dir", default=None, help="기본 $RAG_ROOT/shared_data/interim/md")
    ap.add_argument("--docs-dir", default=None, help="기본 $RAG_ROOT/docs")
    ap.add_argument("--samples", type=int, default=3, help="패턴당 표본 수 (기본 3)")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 화면 출력만")
    args = ap.parse_args()

    root_env = os.environ.get("RAG_ROOT")
    needs_root = args.md_dir is None or (args.docs_dir is None and not args.dry_run)
    if not root_env and needs_root:
        print("[중단] RAG_ROOT 가 설정돼 있지 않습니다.", file=sys.stderr)
        print("       export RAG_ROOT=/srv/rfp  후 다시 실행하세요.", file=sys.stderr)
        return 2

    root = Path(root_env) if root_env else None
    md_dir = Path(args.md_dir) if args.md_dir else root / "shared_data" / "interim" / "md"
    if not md_dir.is_dir():
        print(f"[중단] md 폴더를 찾을 수 없습니다: {md_dir}", file=sys.stderr)
        return 2

    rows, meta, warnings = measure(md_dir, args.samples)
    report(rows, meta)

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았습니다.")
        return 0

    docs_dir = Path(args.docs_dir) if args.docs_dir else root / "docs"
    for p in write_outputs(rows, meta, warnings, md_dir, docs_dir):
        print(f"[저장] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
