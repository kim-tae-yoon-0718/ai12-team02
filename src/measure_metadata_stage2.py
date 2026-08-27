#!/usr/bin/env python3
"""
[연쇄 체크리스트 1-18] 2단계 — CSV 값 ↔ md 본문 전수 대조

대상 4컬럼 × 100건:
    사업명 / 사업 금액 / 발주 기관 / 입찰 참여 마감일

판정 등급:
    high      정규화 후 정확히 일치
    mid       변환·포함 관계로 일치
    low       근접·부분 일치만
    불일치     본문에 다른 값만 존재
    미발견     후보 자체가 없음

  → 육안 대상은 low / 불일치 / 미발견 세 통.
  → high·mid 도 컬럼당 3건씩 열어 등급이 맞는지 확인할 것 (1-12 13번 재발 방지).

무엇을 하지 않는가:
    - 금액 한글 표기(일금 일억…)는 잡지 않는다. 미발견 30건 초과 시 재검토.
    - 마감일 시각은 등급 판정에 넣지 않는다. 별도 열로만 기록.
    - 0원/1원을 판정하지 않는다. 미확인 8건은 별도 추적표로만 낸다.
    - 약칭 사전을 만들지 않는다. mid 목록(사례)만 낸다.

규약:
    - RAG_ROOT 환경변수 + 코드 조립, shared_data 읽기 전용
    - 무작위 없음. 파일 미변경. 문서를 하나씩 열고 닫는다(162MB).

사용:
    python3 src/measure_metadata_stage2.py

출력:
    $RAG_ROOT/docs/measurement_1-18_stage2.{md,json}
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

CSV_RELPATH = ("shared_data", "raw", "data_list.csv")
MD_RELPATH = ("shared_data", "interim", "md")
ENCODING = "utf-8-sig"

C_NAME, C_AMT, C_ORG, C_END, C_FILE = "사업명", "사업 금액", "발주 기관", "입찰 참여 마감일", "파일명"

# 미확인 8건 (1단계 실측). 두 통을 합산하지 않는다.
UNKNOWN_IN_CLUSTER = {14, 15, 18, 36, 73}      # 18건 묶음 안 · 0원
UNKNOWN_OUTSIDE = {43, 85, 61}                  # 묶음 밖 · 0원1/1원1/결측1

EYE_CHECK_N = 3          # high·mid 에서 표본 확인할 건수 (앞에서부터)
MISS_ALARM = 30          # 미발견이 이 수를 넘으면 규칙을 의심하라는 경고선

TAG_RE = re.compile(r"<[^>]+>")
ESCAPE_RE = re.compile(r"\\([*_\[\]()#+\-.!|~`>])")
SPACED_RE = re.compile(r"(?<=[가-힣])\s+(?=[가-힣])")
WS_RE = re.compile(r"\s+")

NUM_RE = re.compile(r"[\d][\d,]{2,}")
DATE_RE = re.compile(r"(20\d{2})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})")
DATE_SHORT_RE = re.compile(r"['’](\d{2})[.\-/]\s*(\d{1,2})[.\-/]\s*(\d{1,2})")
ORG_STRIP = re.compile(r"[\s()\[\]（）·ㆍ,]|주식회사|재단법인|사단법인|㈜")


# ── 공통 ────────────────────────────────────────────────────────
def rag_root() -> Path:
    root = os.environ.get("RAG_ROOT")
    if not root:
        sys.exit("RAG_ROOT 환경변수가 없습니다. (규약 §2-3)")
    p = Path(root)
    if not p.is_dir():
        sys.exit(f"RAG_ROOT 경로가 없습니다: {p}")
    return p


def norm_text(s: str) -> str:
    """본문·CSV 값 양쪽에 같은 정규화를 건다. 한쪽만 하면 전부 미발견이 된다."""
    s = unicodedata.normalize("NFC", s)
    s = TAG_RE.sub(" ", s)          # HTML 표 태그 제거 (표 비율 median 58%)
    s = ESCAPE_RE.sub(r"\1", s)     # 1-12 25번 역슬래시
    s = SPACED_RE.sub("", s)        # 1-12 14번 글자 사이 공백
    return WS_RE.sub(" ", s)


def squeeze(s: str) -> str:
    """비교 전용 — 공백을 전부 없앤다."""
    return WS_RE.sub("", norm_text(s))


def to_num(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, AttributeError):
        return None


# ── 사업명 ──────────────────────────────────────────────────────
def match_name(csv_v, body_sq):
    v = squeeze(csv_v)
    if not v:
        return "결측", {}
    if v in body_sq:
        return "high", {"위치": body_sq.find(v)}
    # 앞/뒤 절단 (파일명 절단과 같은 현상)
    for cut in (0.8, 0.6):
        head, tail = v[:int(len(v) * cut)], v[-int(len(v) * cut):]
        if len(head) >= 8 and head in body_sq:
            return "mid", {"형태": "앞부분", "조각": head[:30]}
        if len(tail) >= 8 and tail in body_sq:
            return "mid", {"형태": "뒷부분", "조각": tail[:30]}
    # 어절 절반 이상
    toks = [t for t in norm_text(csv_v).split() if len(t) >= 2]
    if toks:
        hit = sum(1 for t in toks if squeeze(t) in body_sq)
        if hit / len(toks) >= 0.5:
            return "low", {"어절적중": f"{hit}/{len(toks)}"}
    return "미발견", {}


# ── 사업금액 ────────────────────────────────────────────────────
def match_amount(csv_v, body_norm):
    n = to_num(csv_v)
    if n is None:
        return "결측", {}
    if n <= 1:
        return "미확인", {"값": n}          # 0원·1원은 판정하지 않는다

    cands = set()
    for m in NUM_RE.finditer(body_norm):
        raw = m.group()
        x = to_num(raw)
        if x is None or x < 100:
            continue
        tail = body_norm[m.end():m.end() + 6]
        cands.add(x)
        if "천원" in tail:
            cands.add(x * 1_000)
        if "백만" in tail:
            cands.add(x * 1_000_000)
    if not cands:
        return "미발견", {}
    if n in cands:
        return "high", {}
    # 단위 환산 일치
    if any(abs(c - n) < 1 for c in cands):
        return "mid", {"형태": "단위환산"}
    # ⭐ 부가세 — 불일치가 아니라 둘 다 맞음
    vat = [c for c in cands if abs(c - n * 1.1) < max(n * 0.001, 1)]
    if vat:
        return "mid", {"형태": "부가세추정", "본문값": vat[0]}
    near = [c for c in cands if 0.9 <= c / n <= 1.11]
    if near:
        return "low", {"근접값": sorted(near)[:3]}
    return "불일치", {"본문최대값": max(cands)}


# ── 발주기관 ────────────────────────────────────────────────────
def match_org(csv_v, body_sq):
    v = ORG_STRIP.sub("", squeeze(csv_v))
    if not v:
        return "결측", {}
    body = ORG_STRIP.sub("", body_sq)
    if v in body:
        return "high", {}
    # ⭐ 포함 관계 — 이게 임현진 2-2-2 약칭 사례가 된다
    for ext in ("교", "청", "원", "공사", "공단", "재단", "센터", "대학교"):
        if (v + ext) in body:
            return "mid", {"형태": "확장", "본문표기": v + ext}
    for cut in range(len(v) - 1, 2, -1):
        if v[:cut] in body:
            if cut >= len(v) - 2:
                return "mid", {"형태": "축약", "본문표기": v[:cut]}
            if cut >= 3:
                return "low", {"공통부분": v[:cut]}
            break
    return "미발견", {}


# ── 마감일 ──────────────────────────────────────────────────────
def dates_in(body_norm):
    out = set()
    for m in DATE_RE.finditer(body_norm):
        y, mo, d = m.groups()
        out.add((int(y), int(mo), int(d)))
    for m in DATE_SHORT_RE.finditer(body_norm):
        y, mo, d = m.groups()
        out.add((2000 + int(y), int(mo), int(d)))
    return out


TIME_PATTERNS = [
    lambda h: re.compile(rf"\b{h}\s*:\s*\d{{2}}"),
    lambda h: re.compile(rf"\b{h}\s*시"),
]


def match_deadline(csv_v, body_norm):
    """등급은 날짜만. 시각은 별도 열."""
    s = (csv_v or "").strip()
    if not s:
        return "결측", {}
    try:
        d_part, t_part = s.split(" ")
        y, mo, dd = (int(x) for x in d_part.split("-"))
        hh = int(t_part.split(":")[0])
    except (ValueError, IndexError):
        return "형식불명", {"원값": s}

    found = dates_in(body_norm)
    extra = {"본문날짜후보수": len(found), "csv시각": t_part}

    if t_part == "00:00:00":
        extra["시각"] = "CSV에 시각 없음(00:00:00) — 대조 제외"
    if (y, mo, dd) not in found:
        return ("미발견" if not found else "불일치"), extra

    if t_part != "00:00:00":
        hit = any(p(hh).search(body_norm) for p in TIME_PATTERNS)
        extra["시각"] = "일치" if hit else "본문에 시각 없음"
    return "high", extra


# ── 본체 ────────────────────────────────────────────────────────
def main():
    root = rag_root()
    csv_path = root.joinpath(*CSV_RELPATH)
    md_dir = root.joinpath(*MD_RELPATH)
    if not md_dir.is_dir():
        sys.exit(f"md 폴더가 없습니다: {md_dir}")

    with csv_path.open(encoding=ENCODING, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, start=2):
        r["_line"] = i

    # 파일명 → md 경로 (NFC 매칭, glob 안 씀 — 1-11 ④ 특수문자 사고)
    md_index = {unicodedata.normalize("NFC", p.stem): p
                for p in md_dir.iterdir() if p.suffix == ".md"}

    grades = {c: Counter() for c in (C_NAME, C_AMT, C_ORG, C_END)}
    details, org_mid, vat_pairs, unmatched_files = [], [], [], []
    csv_time_dist = Counter()

    for r in rows:
        stem = unicodedata.normalize("NFC", (r.get(C_FILE) or "").rsplit(".", 1)[0])
        md_path = md_index.get(stem)
        if md_path is None:
            unmatched_files.append({"행": r["_line"], "파일명": r.get(C_FILE)})
            continue

        # 문서를 하나씩 열고 닫는다 (총 162MB — 한 번에 올리지 않는다)
        body_norm = norm_text(md_path.read_text(encoding="utf-8", errors="replace"))
        body_sq = WS_RE.sub("", body_norm)

        end_raw = (r.get(C_END) or "").strip()
        csv_time_dist[end_raw.split(" ")[1] if " " in end_raw else "(결측)"] += 1

        row_out = {"행": r["_line"], "파일명": r.get(C_FILE)}
        for col, fn, arg in (
            (C_NAME, match_name, body_sq),
            (C_AMT, match_amount, body_norm),
            (C_ORG, match_org, body_sq),
            (C_END, match_deadline, body_norm),
        ):
            g, extra = fn(r.get(col), arg)
            grades[col][g] += 1
            row_out[col] = {"등급": g, **extra}

            if col == C_ORG and g == "mid":
                org_mid.append({"행": r["_line"], "csv": r.get(col),
                                "본문": extra.get("본문표기")})
            if col == C_AMT and extra.get("형태") == "부가세추정":
                vat_pairs.append({"행": r["_line"], "csv": to_num(r.get(col)),
                                  "본문": extra.get("본문값")})
        details.append(row_out)

        del body_norm, body_sq

    # 미확인 8건 — 두 통을 합산하지 않는다
    by_line = {d["행"]: d for d in details}
    unknown = {
        "묶음안(0원 5건)": [by_line[l] for l in sorted(UNKNOWN_IN_CLUSTER) if l in by_line],
        "묶음밖(3건)": [by_line[l] for l in sorted(UNKNOWN_OUTSIDE) if l in by_line],
    }

    # 육안 대상 / 표본 확인 대상
    eye, spot = defaultdict(list), defaultdict(list)
    for d in details:
        for col in (C_NAME, C_AMT, C_ORG, C_END):
            g = d[col]["등급"]
            if g in ("low", "불일치", "미발견", "형식불명"):
                eye[col].append({"행": d["행"], "등급": g, **d[col]})
            elif g in ("high", "mid") and len(spot[col]) < EYE_CHECK_N:
                spot[col].append({"행": d["행"], "등급": g})

    res = {
        "항목": "1-18 2단계 — 본문 전수 대조",
        "측정일": date.today().isoformat(),
        "대상": {"csv": str(csv_path), "md": str(md_dir), "대조건수": len(details)},
        "md미매칭": unmatched_files,
        "등급분포": {c: dict(v) for c, v in grades.items()},
        "csv마감시각분포": dict(csv_time_dist),
        "발주기관_mid목록": org_mid,
        "금액_부가세쌍": vat_pairs,
        "미확인8건": unknown,
        "육안대상": {c: v for c, v in eye.items()},
        "표본확인대상": {c: v for c, v in spot.items()},
        "상세": details,
    }

    out_dir = rag_root() / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "measurement_1-18_stage2.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(res)
    (out_dir / "measurement_1-18_stage2.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n→ {out_dir/'measurement_1-18_stage2.md'}")


def render(r):
    L = [f"# 1-18 2단계 — 본문 전수 대조 ({r['측정일']})\n"]
    L.append(f"- 대조: **{r['대상']['대조건수']}건** × 4컬럼")
    if r["md미매칭"]:
        L.append(f"- ⚠️ md 미매칭: {len(r['md미매칭'])}건 {r['md미매칭']}")
    L.append("")

    L.append("## 등급 분포\n")
    keys = ["high", "mid", "low", "불일치", "미발견", "결측", "미확인", "형식불명"]
    L.append("| 컬럼 | " + " | ".join(keys) + " |")
    L.append("| --- " * (len(keys) + 1) + "|")
    for col, dist in r["등급분포"].items():
        L.append(f"| {col} | " + " | ".join(str(dist.get(k, 0)) for k in keys) + " |")
    L.append("")

    for col, dist in r["등급분포"].items():
        miss = dist.get("미발견", 0)
        if miss > MISS_ALARM:
            L.append(f"⚠️ **{col} 미발견 {miss}건** — {MISS_ALARM}건 초과. "
                     f"데이터가 아니라 **대조 규칙을 먼저 의심**할 것.")
    L.append("")

    L.append("## 마감일 — CSV 시각 분포 (대조 없이 CSV만)\n")
    for k, v in sorted(r["csv마감시각분포"].items(), key=lambda x: -x[1]):
        flag = "  ← ⚠️ 시각 소실" if k == "00:00:00" else ""
        L.append(f"- `{k}` : {v}건{flag}")
    L.append("")

    L.append(f"## ⭐ 발주기관 mid 목록 — 약칭 사례 ({len(r['발주기관_mid목록'])}건) → 임현진 2-2-2\n")
    if r["발주기관_mid목록"]:
        L.append("| 행 | CSV 표기 | 본문 표기 |")
        L.append("| --- | --- | --- |")
        for x in r["발주기관_mid목록"]:
            L.append(f"| {x['행']} | {x['csv']} | {x['본문']} |")
    else:
        L.append("없음")
    L.append("")

    L.append(f"## 금액 — 부가세 추정 쌍 ({len(r['금액_부가세쌍'])}건)\n")
    L.append("⭐ 불일치가 아니라 **둘 다 맞음**. 1-12-1 필드 정의에 넘길 것.\n")
    for x in r["금액_부가세쌍"]:
        L.append(f"- {x['행']}행 · CSV {x['csv']:,.0f} / 본문 {x['본문']:,.0f}")
    L.append("")

    L.append("## 미확인 8건 — 두 통을 합산하지 않는다\n")
    for bucket, items in r["미확인8건"].items():
        L.append(f"### {bucket}")
        for d in items:
            L.append(f"- {d['행']}행 · 금액 {d['사업 금액']} · 마감일 {d['입찰 참여 마감일']}")
        L.append("")

    L.append("## 육안 대상 (low · 불일치 · 미발견)\n")
    for col, items in r["육안대상"].items():
        L.append(f"- **{col}**: {len(items)}건 → 행 {[i['행'] for i in items]}")
    L.append("")

    L.append("## ⭐ 표본 확인 대상 (high·mid 에서 앞 3건)\n")
    L.append("오탐은 실패 통이 아니라 **성공 통에 숨는다**. 1-12 13번 재발 방지선.\n")
    for col, items in r["표본확인대상"].items():
        L.append(f"- **{col}**: {[(i['행'], i['등급']) for i in items]}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
