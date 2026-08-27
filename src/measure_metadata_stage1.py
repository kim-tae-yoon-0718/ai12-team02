#!/usr/bin/env python3
"""
[연쇄 체크리스트 1-18] 1단계 — 식별자 체계 · 값 분포 하단 · 표기 갈림

0단계에서 나온 신호 다섯 개를 각각 숫자로 만든다.

  A. 공고번호 체계 분류 (11자리 / R형식 / 기타)      ← 잠정. 태민 님 결과 후 재집계
  B. 사업금액 분포 하단 (0.0 및 극단 소액) + 8건과의 교집합
  C. 사업명·공개일자 중복 1쌍의 위치 (1-14로 넘길 단서)
  D. 발주기관 표기 갈림 (정규화 그룹 + 포함관계)
  E. 파일명 ↔ 발주기관 일치 여부 + 태민 님 부탁용 8건 목록

무엇을 하지 않는가:
    - 본문(md) 대조를 하지 않는다. 그건 2단계.
    - 0.0 을 `항목 없음`으로 판정하지 않는다. 별도 통으로 세기만 한다.
    - 정정공고 쌍을 확정하지 않는다. 단서 위치만 넘긴다 (1-14 소관).
    - 약칭 사전을 만들지 않는다. 사례만 낸다 (임현진 2-2-2 소관).

규약:
    - RAG_ROOT 환경변수 + 코드 조립, shared_data 읽기 전용
    - 무작위 없음. 파일 미변경.

사용:
    python src/measure_metadata_stage1.py

출력:
    docs/measurement_1-18_stage1.{md,json}
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

CSV_RELPATH = ("shared_data", "raw", "data_list.csv")
ENCODING = "utf-8-sig"

C_NO, C_CHA, C_NAME, C_AMT = "공고 번호", "공고 차수", "사업명", "사업 금액"
C_ORG, C_OPEN, C_START, C_END = "발주 기관", "공개 일자", "입찰 참여 시작일", "입찰 참여 마감일"
C_FILE = "파일명"

# 소액 하단을 몇 건까지 보여줄지. 기준선이 아니라 표시 개수다.
LOW_SHOW = 15

NUM_ID_RE = re.compile(r"^\d{11}$")
R_ID_RE = re.compile(r"^R\d{2}[A-Z]{2}\d+$")

ORG_STRIP = re.compile(r"[\s()\[\]（）·ㆍ,]|주식회사|재단법인|사단법인|\(주\)|㈜|\(사\)|\(재\)")


def rag_root() -> Path:
    root = os.environ.get("RAG_ROOT")
    if not root:
        sys.exit("RAG_ROOT 환경변수가 없습니다. (규약 §2-3)")
    p = Path(root)
    if not p.is_dir():
        sys.exit(f"RAG_ROOT 경로가 없습니다: {p}")
    return p


def load():
    path = rag_root().joinpath(*CSV_RELPATH)
    with path.open(encoding=ENCODING, newline="") as f:
        rows = list(csv.DictReader(f))
    # csv 줄번호(헤더=1) 부여 — 0단계 출력과 행 번호를 맞추기 위함
    for i, r in enumerate(rows, start=2):
        r["_line"] = i
    return path, rows


def val(r, k):
    return (r.get(k) or "").strip()


def to_num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def norm_org(s):
    return ORG_STRIP.sub("", unicodedata.normalize("NFC", s))


# ── A. 공고번호 체계 ────────────────────────────────────────────
def section_a(rows):
    buckets = defaultdict(list)
    for r in rows:
        v = val(r, C_NO)
        if not v:
            buckets["결측"].append(r["_line"])
        elif NUM_ID_RE.match(v):
            buckets["11자리 숫자"].append(r["_line"])
        elif R_ID_RE.match(v):
            buckets["R형식"].append(r["_line"])
        else:
            buckets[f"기타: {v}"].append(r["_line"])

    cha = defaultdict(list)
    for r in rows:
        v = val(r, C_CHA)
        cha[v if v else "(결측)"].append(r["_line"])

    return {
        "체계별": {k: {"건수": len(v), "행": v} for k, v in buckets.items()},
        "차수분포": {k: len(v) for k, v in cha.items()},
        "비고": "결측 18건 확보 전 잠정치. 태민 님 결과 수령 후 재집계.",
    }


# ── B. 사업금액 하단 ────────────────────────────────────────────
def section_b(rows, end_missing_lines):
    zeros, missing, vals = [], [], []
    for r in rows:
        s = val(r, C_AMT)
        if not s:
            missing.append(r["_line"])
            continue
        n = to_num(s)
        if n is None:
            missing.append(r["_line"])
        elif n == 0:
            zeros.append(r["_line"])
        else:
            vals.append((n, r["_line"], val(r, C_NAME)))

    vals.sort()
    low = [{"금액": n, "행": ln, "사업명": nm[:40]} for n, ln, nm in vals[:LOW_SHOW]]

    zset, mset = set(zeros), set(missing)
    return {
        "0원": {"건수": len(zeros), "행": zeros},
        "결측": {"건수": len(missing), "행": missing},
        "하단정렬": low,
        "마감일결측8건과의교집합": {
            "0원∩8건": sorted(zset & end_missing_lines),
            "결측∩8건": sorted(mset & end_missing_lines),
        },
        "비고": "0원은 판정하지 않고 별도 통으로만 셈. `항목 없음`/`추출 실패` 구분은 2단계 본문 대조 후.",
    }


# ── C. 중복 단서 ────────────────────────────────────────────────
def section_c(rows):
    out = {}
    for col in (C_NAME, C_OPEN):
        groups = defaultdict(list)
        for r in rows:
            v = val(r, col)
            if v:
                groups[v].append(r["_line"])
        dups = {k: v for k, v in groups.items() if len(v) > 1}
        out[col] = [{"값": k, "행": v} for k, v in dups.items()]

    name_lines = {tuple(d["행"]) for d in out[C_NAME]}
    open_lines = {tuple(d["행"]) for d in out[C_OPEN]}
    out["같은쌍인가"] = bool(name_lines & open_lines)
    out["비고"] = "1-14 정정공고 식별의 단서. 여기서 확정하지 않는다."
    return out


# ── D. 발주기관 표기 갈림 ───────────────────────────────────────
def section_d(rows):
    raw2lines = defaultdict(list)
    for r in rows:
        v = val(r, C_ORG)
        if v:
            raw2lines[v].append(r["_line"])

    norm2raw = defaultdict(set)
    for raw in raw2lines:
        norm2raw[norm_org(raw)].add(raw)

    same_after_norm = [
        {"정규화": k, "원표기": sorted(v)}
        for k, v in norm2raw.items() if len(v) > 1
    ]

    # 포함관계 — 약칭/정식명 후보 (한영대학 ⊂ 한영대학교)
    names = sorted(norm2raw.keys())
    contains = []
    for a in names:
        for b in names:
            if a != b and len(a) >= 3 and a in b:
                contains.append({"짧은쪽": a, "긴쪽": b})

    return {
        "고유표기수": len(raw2lines),
        "정규화후_같아지는_묶음": same_after_norm,
        "포함관계_후보": contains,
        "비고": "약칭 사전을 만들지 않는다. 사례만 임현진 2-2-2로 넘긴다.",
    }


# ── E. 파일명 대조 + 부탁용 목록 ────────────────────────────────
def section_e(rows, end_missing_lines):
    mismatch, no_us = [], []
    for r in rows:
        fn, org = val(r, C_FILE), val(r, C_ORG)
        stem = fn.rsplit(".", 1)[0]
        if "_" not in stem:
            no_us.append({"행": r["_line"], "파일명": fn})
            continue
        head = stem.split("_", 1)[0]
        if norm_org(head) != norm_org(org):
            mismatch.append({"행": r["_line"], "파일명앞": head, "발주기관": org})

    ask = []
    for r in rows:
        if r["_line"] in end_missing_lines:
            ask.append({
                "행": r["_line"],
                "파일명": val(r, C_FILE),
                "사업명": val(r, C_NAME),
                "사업금액": val(r, C_AMT) or "(결측)",
                "발주기관": val(r, C_ORG),
            })
    ask.sort(key=lambda d: to_num(d["사업금액"]) if to_num(d["사업금액"]) is not None else -1)

    return {
        "구분자없는파일명": no_us,
        "파일명앞≠발주기관": mismatch,
        "태민님_대상8건_금액오름차순": ask,
    }


def render(res):
    L = [f"# 1-18 1단계 ({res['측정일']})\n", f"- 대상: `{res['대상']}`\n"]

    a = res["A_공고번호체계"]
    L.append("## A. 공고번호 체계 (잠정 — 18건 확보 전)\n")
    L.append("| 체계 | 건수 |")
    L.append("| --- | --- |")
    for k, v in a["체계별"].items():
        L.append(f"| {k} | {v['건수']} |")
    L.append(f"\n- 차수 분포: {a['차수분포']}")
    L.append(f"- {a['비고']}\n")

    b = res["B_사업금액하단"]
    L.append("## B. 사업금액 하단\n")
    L.append(f"- **0원: {b['0원']['건수']}건** → 행 {b['0원']['행']}")
    L.append(f"- 결측: {b['결측']['건수']}건 → 행 {b['결측']['행']}")
    L.append(f"- 마감일결측 8건과의 교집합: {b['마감일결측8건과의교집합']}\n")
    L.append("| 금액 | 행 | 사업명 |")
    L.append("| --- | --- | --- |")
    for d in b["하단정렬"]:
        L.append(f"| {d['금액']:,.0f} | {d['행']} | {d['사업명']} |")
    L.append(f"\n⚠️ {b['비고']}\n")

    c = res["C_중복단서"]
    L.append("## C. 중복 단서 (1-14로)\n")
    L.append(f"- 사업명 중복: {c[C_NAME]}")
    L.append(f"- 공개일자 중복: {c[C_OPEN]}")
    L.append(f"- **같은 쌍인가: {c['같은쌍인가']}**\n")

    d = res["D_발주기관"]
    L.append("## D. 발주기관 표기 갈림\n")
    L.append(f"- 고유 표기 {d['고유표기수']}개")
    L.append(f"- 정규화 후 같아지는 묶음: {len(d['정규화후_같아지는_묶음'])}건")
    for x in d["정규화후_같아지는_묶음"]:
        L.append(f"  - {x['원표기']}")
    L.append(f"- 포함관계 후보: {len(d['포함관계_후보'])}건")
    for x in d["포함관계_후보"]:
        L.append(f"  - `{x['짧은쪽']}` ⊂ `{x['긴쪽']}`")
    L.append("")

    e = res["E_파일명"]
    L.append("## E. 파일명 대조\n")
    L.append(f"- 구분자(_) 없는 파일명: {len(e['구분자없는파일명'])}건 {e['구분자없는파일명']}")
    L.append(f"- 파일명 앞 ≠ 발주기관: {len(e['파일명앞≠발주기관'])}건")
    for x in e["파일명앞≠발주기관"]:
        L.append(f"  - {x['행']}행 `{x['파일명앞']}` vs `{x['발주기관']}`")
    L.append("\n### 태민 님 부탁 대상 8건 (금액 오름차순)\n")
    L.append("| 행 | 사업금액 | 발주기관 | 파일명 |")
    L.append("| --- | --- | --- | --- |")
    for x in e["태민님_대상8건_금액오름차순"]:
        L.append(f"| {x['행']} | {x['사업금액']} | {x['발주기관']} | {x['파일명']} |")
    return "\n".join(L)


def main():
    path, rows = load()
    end_missing = {r["_line"] for r in rows if not val(r, C_END)}

    res = {
        "항목": "1-18 1단계",
        "측정일": date.today().isoformat(),
        "대상": str(path),
        "마감일결측행": sorted(end_missing),
        "A_공고번호체계": section_a(rows),
        "B_사업금액하단": section_b(rows, end_missing),
        "C_중복단서": section_c(rows),
        "D_발주기관": section_d(rows),
        "E_파일명": section_e(rows, end_missing),
    }

    out_dir = rag_root() / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "measurement_1-18_stage1.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(res)
    (out_dir / "measurement_1-18_stage1.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"\n→ {out_dir/'measurement_1-18_stage1.md'}")


if __name__ == "__main__":
    main()
