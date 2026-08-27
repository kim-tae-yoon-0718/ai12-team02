#!/usr/bin/env python3
"""
[연쇄 체크리스트 1-18] 0단계 — 제공받은 메타데이터 컬럼 파악

무엇을 하는가:
    shared_data/raw/data_list.csv 를 읽어 컬럼 12개의
    이름 / 추정 타입 / 결측률 / 고유값 수 / 값 예시를 뽑는다.
    결측이 있는 컬럼끼리 "같은 행에서 비었는가"도 대조한다. (3-1 재료)

무엇을 하지 않는가:
    - 파일을 쓰지 않는다. shared_data 는 읽기만 한다.
    - 본문(md) 대조를 하지 않는다. 그건 2단계.
    - 값을 판정하지 않는다. 여기는 "파악"이고 "검증"은 1단계 이후.

규약:
    - RAG_ROOT 환경변수 + 코드 조립 (절대경로 하드코딩 없음)
    - shared_data 읽기 전용
    - 무작위 없음 (예시는 앞에서부터 순서대로)
    - 컬럼명 NFC 정규화 여부를 함께 기록 (1-11 ④)

사용:
    export RAG_ROOT=/srv/rfp        # 이미 잡혀 있으면 생략
    python src/measure_metadata.py

출력:
    docs/measurement_1-18_columns.md
    docs/measurement_1-18_columns.json
"""

import csv
import json
import os
import re
import statistics
import sys
import unicodedata
from datetime import date
from pathlib import Path

CSV_RELPATH = ("shared_data", "raw", "data_list.csv")
ENCODING = "utf-8-sig"          # 1-11 ⑤ 에서 확인된 값
EXAMPLE_N = 3                    # 앞에서부터 N건

DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\d{4}\.\s?\d{1,2}\.\s?\d{1,2}"),
    re.compile(r"^\d{4}/\d{1,2}/\d{1,2}"),
    re.compile(r"^\d{4}년\s?\d{1,2}월\s?\d{1,2}일"),
]
INT_RE = re.compile(r"^-?[\d,]+$")
FLOAT_RE = re.compile(r"^-?[\d,]*\.\d+$")


def rag_root() -> Path:
    root = os.environ.get("RAG_ROOT")
    if not root:
        sys.exit("RAG_ROOT 환경변수가 없습니다. 규약 §2-3 — 절대경로를 코드에 박지 않습니다.")
    p = Path(root)
    if not p.is_dir():
        sys.exit(f"RAG_ROOT 경로가 없습니다: {p}")
    return p


def is_missing(v: str) -> bool:
    """빈 문자열과 공백만 있는 값을 결측으로 본다. 'null' 같은 문자열은 값으로 둔다."""
    return v is None or v.strip() == ""


def guess_type(values):
    """비결측 값 전체를 보고 타입을 추정한다. 하나라도 어긋나면 mixed."""
    if not values:
        return "unknown(전량 결측)"
    kinds = set()
    for v in values:
        s = v.strip()
        if any(p.match(s) for p in DATE_PATTERNS):
            kinds.add("date")
        elif INT_RE.match(s):
            kinds.add("int")
        elif FLOAT_RE.match(s):
            kinds.add("float")
        else:
            kinds.add("text")
    if kinds == {"int"}:
        return "int"
    if kinds <= {"int", "float"}:
        return "number"
    if kinds == {"date"}:
        return "date"
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed(" + "/".join(sorted(kinds)) + ")"


def nfc_note(name: str) -> str:
    nfc = unicodedata.normalize("NFC", name)
    return "NFC" if nfc == name else f"⚠️비NFC → NFC로는 '{nfc}'"


def main():
    csv_path = rag_root().joinpath(*CSV_RELPATH)
    if not csv_path.is_file():
        sys.exit(f"CSV가 없습니다: {csv_path}")

    with csv_path.open(encoding=ENCODING, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            sys.exit("빈 파일입니다.")
        rows = [r for r in reader]

    n_rows = len(rows)
    n_cols = len(header)

    # 컬럼 수가 어긋나는 행 (1-11 ⑤ 에서 0건이었음 — 대조용)
    ragged = [i for i, r in enumerate(rows, start=2) if len(r) != n_cols]

    # 완전히 같은 행
    seen, dup_rows = set(), []
    for i, r in enumerate(rows, start=2):
        key = tuple(r)
        if key in seen:
            dup_rows.append(i)
        seen.add(key)

    columns = []
    missing_sets = {}

    for ci, raw_name in enumerate(header):
        col = [(r[ci] if ci < len(r) else "") for r in rows]
        present = [v for v in col if not is_missing(v)]
        missing_idx = [i for i, v in enumerate(col) if is_missing(v)]
        missing_sets[raw_name] = set(missing_idx)

        lengths = [len(v.strip()) for v in present]
        examples = [v.strip() for v in present[:EXAMPLE_N]]

        shortest = min(present, key=lambda v: len(v.strip())).strip() if present else None
        longest = max(present, key=lambda v: len(v.strip())).strip() if present else None

        columns.append({
            "순서": ci + 1,
            "컬럼명": raw_name,
            "정규화": nfc_note(raw_name),
            "추정타입": guess_type(present),
            "결측건수": len(missing_idx),
            "결측률(%)": round(len(missing_idx) / n_rows * 100, 1) if n_rows else None,
            "고유값수": len(set(v.strip() for v in present)),
            "값길이_min": min(lengths) if lengths else None,
            "값길이_median": int(statistics.median(lengths)) if lengths else None,
            "값길이_max": max(lengths) if lengths else None,
            "예시_앞에서3건": examples,
            "가장짧은값": shortest,
            "가장긴값앞80자": (longest[:80] + "…") if longest and len(longest) > 80 else longest,
            "결측행번호(csv기준)": [i + 2 for i in missing_idx],
        })

    # ⭐ 3-1 재료 — 결측이 있는 컬럼끼리 같은 행에서 비었는지
    overlaps = []
    holed = [name for name, s in missing_sets.items() if s]
    for a in range(len(holed)):
        for b in range(a + 1, len(holed)):
            na, nb = holed[a], holed[b]
            sa, sb = missing_sets[na], missing_sets[nb]
            inter = sa & sb
            if not inter:
                continue
            overlaps.append({
                "컬럼A": na, "A결측": len(sa),
                "컬럼B": nb, "B결측": len(sb),
                "교집합": len(inter),
                "완전일치": sa == sb,
            })

    result = {
        "항목": "1-18 0단계 — 메타데이터 컬럼 파악",
        "측정일": date.today().isoformat(),
        "대상": str(csv_path),
        "인코딩": ENCODING,
        "데이터행수": n_rows,
        "컬럼수": n_cols,
        "컬럼수불일치행": ragged,
        "완전중복행": dup_rows,
        "컬럼": columns,
        "결측겹침": overlaps,
    }

    out_dir = rag_root() / "docs"
    out_dir.mkdir(exist_ok=True)

    with (out_dir / "measurement_1-18_columns.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with (out_dir / "measurement_1-18_columns.md").open("w", encoding="utf-8") as f:
        f.write(render_md(result))

    print(render_md(result))
    print(f"\n→ {out_dir/'measurement_1-18_columns.md'}")
    print(f"→ {out_dir/'measurement_1-18_columns.json'}")


def render_md(r) -> str:
    L = []
    L.append(f"# 1-18 0단계 — 메타데이터 컬럼 파악 ({r['측정일']})\n")
    L.append(f"- 대상: `{r['대상']}`")
    L.append(f"- 인코딩: {r['인코딩']}")
    L.append(f"- 데이터 행: **{r['데이터행수']}** / 컬럼: **{r['컬럼수']}**")
    L.append(f"- 컬럼 수 불일치 행: {len(r['컬럼수불일치행'])}건 {r['컬럼수불일치행'] or ''}")
    L.append(f"- 완전 중복 행: {len(r['완전중복행'])}건 {r['완전중복행'] or ''}\n")

    L.append("## 컬럼 일람\n")
    L.append("| # | 컬럼명 | 추정타입 | 결측 | 결측률 | 고유값 | 길이(min/med/max) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for c in r["컬럼"]:
        L.append(
            f"| {c['순서']} | {c['컬럼명']} | {c['추정타입']} | {c['결측건수']} | "
            f"{c['결측률(%)']}% | {c['고유값수']} | "
            f"{c['값길이_min']}/{c['값길이_median']}/{c['값길이_max']} |"
        )

    L.append("\n## 컬럼별 값 예시 (앞에서부터 — 무작위 없음)\n")
    for c in r["컬럼"]:
        L.append(f"### {c['순서']}. {c['컬럼명']}  ({c['정규화']})")
        for e in c["예시_앞에서3건"]:
            L.append(f"- `{e[:120]}`" + ("…" if len(e) > 120 else ""))
        L.append(f"- 가장 짧은 값: `{c['가장짧은값']}`")
        L.append(f"- 가장 긴 값: `{c['가장긴값앞80자']}`")
        if c["결측건수"]:
            L.append(f"- ⚠️ 결측 행(csv 줄번호): {c['결측행번호(csv기준)']}")
        L.append("")

    L.append("## ⭐ 결측 겹침 — 같은 행에서 비었는가 (3단계 재료)\n")
    if not r["결측겹침"]:
        L.append("겹치는 결측 없음.")
    else:
        L.append("| 컬럼 A | A결측 | 컬럼 B | B결측 | 교집합 | 완전일치 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for o in r["결측겹침"]:
            L.append(
                f"| {o['컬럼A']} | {o['A결측']} | {o['컬럼B']} | {o['B결측']} | "
                f"{o['교집합']} | {'✅' if o['완전일치'] else ''} |"
            )
    return "\n".join(L)


if __name__ == "__main__":
    main()
