#!/usr/bin/env python3
"""
[연쇄 체크리스트 1-9] 코퍼스 실물·규모 실측

측정만 한다. 판정하지 않는다.
- 파일이 열리는지(1-11), 스캔본인지(1-12), 값이 맞는지(1-18),
  변환 손실(1-13), 페이지/토큰 수(1-13, 1-15)는 여기서 다루지 않는다.

출력: $RAG_ROOT/docs/measurement_1-9.{md,json}

규약 준수:
- 경로는 $RAG_ROOT 환경변수로 조립 (절대경로 하드코딩 금지)
- 측정 함수와 출력 함수를 분리 → 나중에 check_data.py로 승격
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, date
from pathlib import Path

csv.field_size_limit(sys.maxsize)  # 텍스트(본문) 컬럼이 큼

# ── 경로 조립 ────────────────────────────────────────────────

def get_paths():
    root = os.environ.get("RAG_ROOT")
    if not root:
        sys.exit("환경변수 RAG_ROOT가 없습니다. ~/.bashrc 확인 후 source 하세요.")
    root = Path(root)
    return {
        "root": root,
        "files": root / "shared_data" / "raw" / "files",
        "csv": root / "shared_data" / "raw" / "data_list.csv",
        "md": root / "shared_data" / "interim" / "md",
        "out": root / "docs",
    }

# ── 공통 도구 ────────────────────────────────────────────────

def norm(s):
    """유니코드 정규화 + 공백 정리. 한글 파일명 NFC/NFD 차이 흡수."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()

def pct(values, p):
    """정렬된 수치 목록의 p분위수 (0~100). 외부 라이브러리 없이."""
    if not values:
        return None
    vs = sorted(values)
    k = (len(vs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)

def mb(n):
    return round(n / 1024 / 1024, 2)

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
                "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y.%m.%d %H:%M", "%Y/%m/%d %H:%M"]

def parse_date(s):
    """여러 형식을 시도. 실패하면 None — 추측하지 않는다."""
    s = norm(s)
    if not s:
        return None
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None

# ── 1단계: 실물 확인 ──────────────────────────────────────────

def measure_files(files_dir):
    r = {"stage": "1. 실물 확인", "dir": str(files_dir)}
    if not files_dir.is_dir():
        r["error"] = "폴더 없음"
        return r

    entries = [p for p in files_dir.iterdir() if p.is_file()]
    sizes = [p.stat().st_size for p in entries]

    r["total_count"] = len(entries)
    r["extensions"] = dict(Counter(p.suffix.lower() for p in entries).most_common())
    r["total_bytes"] = sum(sizes)
    r["total_mb"] = mb(sum(sizes))
    r["size_bytes"] = {
        "min": min(sizes) if sizes else None,
        "p25": int(pct(sizes, 25)) if sizes else None,
        "median": int(pct(sizes, 50)) if sizes else None,
        "p75": int(pct(sizes, 75)) if sizes else None,
        "max": max(sizes) if sizes else None,
    }
    # 깨졌을 가능성 — 판정이 아니라 눈에 띄게 하는 것
    r["zero_byte"] = sorted(p.name for p in entries if p.stat().st_size == 0)
    r["under_10kb"] = sorted(
        (p.name, p.stat().st_size) for p in entries if 0 < p.stat().st_size < 10 * 1024
    )
    # 파일명 중복 (정규화 후 — 매칭에 쓸 키라서 중요)
    dup = Counter(norm(p.name) for p in entries)
    r["duplicate_names"] = {k: v for k, v in dup.items() if v > 1}
    return r

# ── 2단계: CSV 점검 ──────────────────────────────────────────

WANT = {
    "공고번호": ["공고 번호", "공고번호"],
    "공고차수": ["공고 차수", "공고차수"],
    "사업명": ["사업명"],
    "사업금액": ["사업 금액", "사업금액"],
    "발주기관": ["발주 기관", "발주기관"],
    "공개일자": ["공개 일자", "공개일자"],
    "시작일": ["입찰 참여 시작일", "입찰참여시작일"],
    "마감일": ["입찰 참여 마감일", "입찰참여마감일"],
    "사업요약": ["사업 요약", "사업요약"],
    "파일형식": ["파일형식", "파일 형식"],
    "파일명": ["파일명", "파일 명"],
    "본문": ["텍스트(본문)", "텍스트", "본문"],
}

def resolve_columns(header):
    """실제 헤더를 표준 키에 매핑. 못 찾으면 None으로 남긴다."""
    lookup = {norm(h).replace(" ", ""): h for h in header}
    out = {}
    for key, cands in WANT.items():
        out[key] = next(
            (lookup[norm(c).replace(" ", "")] for c in cands
             if norm(c).replace(" ", "") in lookup), None
        )
    return out

def measure_csv(csv_path):
    r = {"stage": "2. CSV 점검", "path": str(csv_path)}
    if not csv_path.is_file():
        r["error"] = "파일 없음"
        return r

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        header = list(rows[0].keys()) if rows else []

    r["row_count"] = len(rows)
    r["header"] = header
    cols = resolve_columns(header)
    r["column_map"] = cols
    r["unmapped"] = [k for k, v in cols.items() if v is None]

    # 컬럼별 결측률 — 실제 헤더 전부에 대해
    r["missing_rate"] = {
        h: round(sum(1 for row in rows if not norm(row.get(h))) / len(rows) * 100, 1)
        for h in header
    } if rows else {}

    def col(row, key):
        c = cols.get(key)
        return norm(row.get(c)) if c else ""

    # 공고번호 유일성 / 차수 조합 → 정정 쌍 후보
    if cols["공고번호"]:
        nums = [col(row, "공고번호") for row in rows]
        cnt = Counter(n for n in nums if n)
        r["notice_no"] = {
            "unique": len(cnt),
            "blank": sum(1 for n in nums if not n),
            "duplicated": {k: v for k, v in cnt.items() if v > 1},
        }
        if cols["공고차수"]:
            grp = {}
            for row in rows:
                n = col(row, "공고번호")
                if n:
                    grp.setdefault(n, []).append(col(row, "공고차수"))
            multi = {k: sorted(v) for k, v in grp.items() if len(v) > 1}
            r["correction_pairs"] = {"count": len(multi), "samples": dict(list(multi.items())[:5])}
            r["round_values"] = dict(Counter(
                col(row, "공고차수") for row in rows).most_common())

    # 본문 길이 분포
    if cols["본문"]:
        lens = [len(row.get(cols["본문"]) or "") for row in rows]
        r["text_length"] = {
            "min": min(lens) if lens else None,
            "median": int(pct(lens, 50)) if lens else None,
            "p95": int(pct(lens, 95)) if lens else None,
            "max": max(lens) if lens else None,
            "zero": sum(1 for x in lens if x == 0),
            "under_500": sum(1 for x in lens if 0 < x < 500),
        }

    # 마감일 분포 — 기준일이 미정이라 '경과 건수'는 세지 않는다
    # 결측은 채우지 않는다. 세는 것 자체가 측정 결과.
    if cols["마감일"]:
        parsed, unparsed, blank = [], [], 0
        for row in rows:
            raw = col(row, "마감일")
            if not raw:
                blank += 1
                continue
            d = parse_date(raw)
            parsed.append(d) if d else unparsed.append(raw)
        r["deadline"] = {
            "blank": blank,
            "blank_rate": round(blank / len(rows) * 100, 1) if rows else None,
            "parsed": len(parsed),
            "unparsed_count": len(unparsed),
            "unparsed_samples": unparsed[:10],
            "min": str(min(parsed)) if parsed else None,
            "max": str(max(parsed)) if parsed else None,
            "by_month": dict(sorted(Counter(f"{d:%Y-%m}" for d in parsed).items())),
            "note": "기준일 미확정(1-9 ④)이므로 '마감 경과 건수'는 세지 않음",
        }

    # 참고용: 오늘 기준으로 보면 몇 건인지 (확정값 아님)
    if cols["마감일"] and parsed:
        today = date.today()
        r["deadline"]["reference_only_past_as_of_today"] = {
            "as_of": str(today),
            "count": sum(1 for d in parsed if d < today),
            "warning": "참고값. 기준일 확정 전까지 확정 정리에 쓰지 말 것",
        }

    if cols["파일형식"]:
        r["csv_filetype"] = dict(Counter(col(row, "파일형식") for row in rows).most_common())

    r["_rows"] = rows
    r["_cols"] = cols
    return r

# ── 0단계: 코퍼스 버전 확인 ──────────────────────────────────

def read_version(md_dir):
    """VERSION.txt를 찾아 그대로 읽는다. 없으면 '없음'으로 기록한다.

    ⚠️ 없는데 'v1'이라고 적지 않는다 — 아무도 v1이라고 선언한 적이 없다.
    파일이 생기면 재실행 시 자동으로 잡힌다.
    """
    r = {"stage": "0. 코퍼스 버전"}
    candidates = [md_dir / "VERSION.txt", md_dir.parent / "VERSION.txt"]
    for c in candidates:
        if c.is_file():
            r["found_at"] = str(c)
            r["content"] = c.read_text(encoding="utf-8", errors="replace").strip()
            return r
    r["found_at"] = None
    r["content"] = None
    r["note"] = ("VERSION.txt 없음 — 버전 미지정. 팀원에게 요청 중. "
                 "임의로 v1이라 적지 않는다(1-19 재현 근거)")
    r["searched"] = [str(c) for c in candidates]
    return r

# ── 2-B단계: 변환본(md) 점검 ─────────────────────────────────

def measure_md(md_dir):
    """interim/md/ = 잠정 corpus_v1. 손실 판정이 아니라 규모만 잰다."""
    r = {"stage": "2-B. 변환본(md) 점검", "dir": str(md_dir)}
    if not md_dir.is_dir():
        r["error"] = "폴더 없음"
        return r
    files = sorted(p for p in md_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md")
    lens = {}
    for p in files:
        try:
            lens[p.stem] = len(p.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            lens[p.stem] = None
            r.setdefault("read_errors", []).append((p.name, str(e)))
    vals = [v for v in lens.values() if v is not None]
    r["count"] = len(files)
    r["char_length"] = {
        "min": min(vals) if vals else None,
        "median": int(pct(vals, 50)) if vals else None,
        "p95": int(pct(vals, 95)) if vals else None,
        "max": max(vals) if vals else None,
        "zero": sum(1 for v in vals if v == 0),
        "under_500": sum(1 for v in vals if 0 < v < 500),
    }
    r["_lengths"] = lens
    return r

# ── 3단계: 3자 매칭 ──────────────────────────────────────────

def measure_matching(files_dir, csv_result, md_result=None):
    r = {"stage": "3. raw ↔ CSV ↔ md 매칭"}
    if "error" in csv_result or not files_dir.is_dir():
        r["error"] = "선행 단계 실패"
        return r
    cols = csv_result.get("_cols", {})
    if not cols.get("파일명"):
        r["error"] = "CSV에 파일명 컬럼을 찾지 못함"
        return r

    disk = {norm(p.name): p.name for p in files_dir.iterdir() if p.is_file()}
    disk_stem = {norm(p.stem): p.name for p in files_dir.iterdir() if p.is_file()}

    both, csv_only = [], []
    stem_only = []  # 확장자 빼면 붙는 것 — '연결 불가'와 구분
    for row in csv_result["_rows"]:
        name = norm(row.get(cols["파일명"]))
        if not name:
            csv_only.append("(빈 값)")
        elif name in disk:
            both.append(disk[name])
        elif Path(name).stem and norm(Path(name).stem) in disk_stem:
            stem_only.append((name, disk_stem[norm(Path(name).stem)]))
        else:
            csv_only.append(name)

    matched = set(both) | {d for _, d in stem_only}
    file_only = sorted(v for k, v in disk.items() if v not in matched)

    r["exact_match"] = len(both)
    r["stem_match"] = {"count": len(stem_only), "samples": stem_only[:10]}
    r["csv_only"] = {"count": len(csv_only), "samples": csv_only[:10]}
    r["file_only"] = {"count": len(file_only), "samples": file_only[:10]}
    r["note"] = "csv_only/file_only는 '연결 불가'가 아니라 확인 대상. 실제 문자열을 눈으로 볼 것"

    # ── md 대조: parse_file.py가 f.stem으로 이름을 만든다
    if md_result and "error" not in md_result:
        md_len = md_result.get("_lengths", {})
        md_keys = {norm(k): k for k in md_len}
        raw_stems = {norm(Path(v).stem): v for v in disk.values()}

        r["md_vs_raw"] = {
            "md_count": len(md_len),
            "raw_count": len(raw_stems),
            "both": len(set(md_keys) & set(raw_stems)),
            "raw_only": sorted(raw_stems[k] for k in set(raw_stems) - set(md_keys))[:10],
            "md_only": sorted(md_keys[k] for k in set(md_keys) - set(raw_stems))[:10],
        }

        # 같은 문서의 md 글자수 vs CSV 글자수 — '다른가'만 본다. 손실 판정 아님.
        cols_txt = csv_result.get("_cols", {}).get("본문")
        pairs = []
        if cols_txt:
            for row in csv_result["_rows"]:
                fname = norm(row.get(csv_result["_cols"]["파일명"]))
                if not fname:
                    continue
                stem = norm(Path(fname).stem)
                if stem in md_keys and md_len[md_keys[stem]] is not None:
                    csv_n = len(row.get(cols_txt) or "")
                    md_n = md_len[md_keys[stem]]
                    pairs.append((fname, csv_n, md_n))
        if pairs:
            ratios = [md / csv for _, csv, md in pairs if csv > 0]
            r["md_vs_csv"] = {
                "compared": len(pairs),
                "ratio_md_over_csv": {
                    "min": round(min(ratios), 3) if ratios else None,
                    "median": round(pct(ratios, 50), 3) if ratios else None,
                    "max": round(max(ratios), 3) if ratios else None,
                },
                "near_identical_within_1pct": sum(1 for x in ratios if 0.99 <= x <= 1.01),
                "md_much_shorter_under_80pct": sum(1 for x in ratios if x < 0.8),
                "md_much_longer_over_120pct": sum(1 for x in ratios if x > 1.2),
                "extreme_samples": sorted(
                    ((n, c, m) for n, c, m in pairs if c > 0),
                    key=lambda t: t[2] / t[1])[:5],
                "note": "두 경로가 같은 것인지 다른 것인지만 본다. 손실률 측정은 1-13.",
            }
    return r

# ── 출력 ────────────────────────────────────────────────────

def render_md(f, c, m, dm, v):
    L = []
    A = L.append
    A("# [연쇄 체크리스트 1-9] 실측 결과")
    A(f"\n측정 시각: {datetime.now():%Y-%m-%d %H:%M}")
    A("\n> 측정만 함. 판정·해석은 확정 정리에서.")
    A("\n## 0. 측정 대상 코퍼스 버전\n")
    if v.get("content"):
        A(f"- VERSION.txt: `{v['found_at']}`")
        A("```")
        A(v["content"])
        A("```")
    else:
        A("- ⚠️ **VERSION.txt 없음 — 버전 미지정**")
        A(f"- 찾아본 곳: {v.get('searched')}")
        A(f"- {v.get('note')}")
    A(f"- 측정 대상 경로: `{dm.get('dir')}`")
    A("\n---\n\n## 1. 실물 확인")
    if "error" in f:
        A(f"\n⚠️ {f['error']}")
    else:
        A(f"\n- 파일 개수: **{f['total_count']}건**")
        A(f"- 확장자: {f['extensions']}")
        A(f"- 총 용량: **{f['total_mb']} MB**")
        s = f["size_bytes"]
        A(f"- 파일당 용량(byte): min {s['min']} / median {s['median']} / max {s['max']}")
        A(f"- 0바이트: {len(f['zero_byte'])}건 {f['zero_byte'][:5]}")
        A(f"- 10KB 미만: {len(f['under_10kb'])}건 {f['under_10kb'][:5]}")
        A(f"- 파일명 중복: {f['duplicate_names'] or '없음'}")

    A("\n---\n\n## 2. CSV 점검")
    if "error" in c:
        A(f"\n⚠️ {c['error']}")
    else:
        A(f"\n- 데이터 행 수: **{c['row_count']}행**")
        if c["unmapped"]:
            A(f"- ⚠️ 매핑 실패 컬럼: {c['unmapped']}")
        A("\n### 컬럼별 결측률 (%)\n")
        A("| 컬럼 | 결측률 |")
        A("|---|---|")
        for k, v in c["missing_rate"].items():
            A(f"| {k} | {v} |")
        if "notice_no" in c:
            n = c["notice_no"]
            A(f"\n### 공고번호\n")
            A(f"- 고유값 {n['unique']}개 / 빈 값 {n['blank']}건")
            A(f"- 중복: {n['duplicated'] or '없음'}")
        if "correction_pairs" in c:
            cp = c["correction_pairs"]
            A(f"\n### ⭐ 정정 쌍 후보 (같은 공고번호에 차수 여럿)\n")
            A(f"- **{cp['count']}건**")
            A(f"- 예시: {cp['samples']}")
            A(f"- 차수 값 분포: {c.get('round_values')}")
        if "text_length" in c:
            t = c["text_length"]
            A(f"\n### 본문 길이 (문자 수)\n")
            A(f"- min {t['min']} / median {t['median']} / p95 {t['p95']} / max {t['max']}")
            A(f"- 0자: {t['zero']}건 / 500자 미만: {t['under_500']}건")
        if "deadline" in c:
            d = c["deadline"]
            A(f"\n### 마감일 분포\n")
            A(f"- **결측(빈 값): {d['blank']}건 ({d['blank_rate']}%)** ← 채우지 않음. 이것도 측정 결과")
            A(f"- 파싱 성공 {d['parsed']}건 / 형식 불명 {d['unparsed_count']}건 {d['unparsed_samples'][:3]}")
            A(f"- 범위: {d['min']} ~ {d['max']}")
            A(f"- 월별: {d['by_month']}")
            A(f"- ⚠️ {d['note']}")
            if "reference_only_past_as_of_today" in d:
                ref = d["reference_only_past_as_of_today"]
                A(f"- (참고, 확정값 아님) {ref['as_of']} 기준 경과 {ref['count']}건")
        if "csv_filetype" in c:
            A(f"\n- CSV 파일형식 컬럼: {c['csv_filetype']}")

    A("\n---\n\n## 2-B. 변환본(md) 점검 — 잠정 corpus_v1")
    if "error" in dm:
        A(f"\n⚠️ {dm['error']}")
    else:
        A(f"\n- md 파일 개수: **{dm['count']}건**")
        t = dm["char_length"]
        A(f"- 글자 수: min {t['min']} / median {t['median']} / p95 {t['p95']} / max {t['max']}")
        A(f"- 0자: {t['zero']}건 / 500자 미만: {t['under_500']}건")
        if dm.get("read_errors"):
            A(f"- ⚠️ 읽기 실패: {dm['read_errors'][:5]}")

    A("\n---\n\n## 3. raw ↔ CSV ↔ md 매칭")
    if "error" in m:
        A(f"\n⚠️ {m['error']}")
    else:
        A(f"\n- 완전 일치: **{m['exact_match']}건**")
        A(f"- 확장자 제외 시 일치: {m['stem_match']['count']}건 {m['stem_match']['samples'][:3]}")
        A(f"- CSV에만 있음: {m['csv_only']['count']}건 {m['csv_only']['samples'][:5]}")
        A(f"- 파일에만 있음: {m['file_only']['count']}건 {m['file_only']['samples'][:5]}")
        A(f"- {m['note']}")
        if "md_vs_raw" in m:
            v = m["md_vs_raw"]
            A(f"\n### raw ↔ md\n")
            A(f"- 양쪽 존재 {v['both']}건 (raw {v['raw_count']} / md {v['md_count']})")
            A(f"- raw에만: {v['raw_only']}")
            A(f"- md에만: {v['md_only']}")
        if "md_vs_csv" in m:
            v = m["md_vs_csv"]
            A(f"\n### ⭐ md 글자수 ÷ CSV 글자수 — 두 경로가 같은가\n")
            A(f"- 비교 {v['compared']}건")
            A(f"- 비율: min {v['ratio_md_over_csv']['min']} / median {v['ratio_md_over_csv']['median']} / max {v['ratio_md_over_csv']['max']}")
            A(f"- 1% 이내로 거의 동일: **{v['near_identical_within_1pct']}건**")
            A(f"- md가 80% 미만으로 짧음: {v['md_much_shorter_under_80pct']}건")
            A(f"- md가 120% 초과로 김: {v['md_much_longer_over_120pct']}건")
            A(f"- 가장 짧은 쪽 예시 (파일, CSV자수, md자수): {v['extreme_samples']}")
            A(f"- ⚠️ {v['note']}")

    A("\n---\n\n## 이번에 측정하지 않은 것\n")
    A("| 항목 | 어디서 |")
    A("|---|---|")
    A("| 페이지 수 | 1-13 (HWP 도구 미정) |")
    A("| 토큰 수 | 1-15 (1-13-1 포맷 확정 후) |")
    A("| 파일이 열리는지 | 1-11 |")
    A("| 스캔본 비율 | 1-12 |")
    A("| CSV 값이 맞는지 | 1-18 (여기선 결측률만) |")
    A("| 변환 손실률 (원본 대조) | 1-13 |")
    A("| 마감 경과 건수 | 기준일 확정 후 |")
    return "\n".join(L)

def main():
    p = get_paths()
    p["out"].mkdir(parents=True, exist_ok=True)

    v = read_version(p["md"])
    f = measure_files(p["files"])
    c = measure_csv(p["csv"])
    d = measure_md(p["md"])
    m = measure_matching(p["files"], c, d)

    c.pop("_rows", None)
    c.pop("_cols", None)
    d.pop("_lengths", None)

    md = render_md(f, c, m, d, v)
    (p["out"] / "measurement_1-9.md").write_text(md, encoding="utf-8")
    (p["out"] / "measurement_1-9.json").write_text(
        json.dumps({"measured_at": datetime.now().isoformat(),
                    "corpus_version": v,
                    "files": f, "csv": c, "md": d, "matching": m},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(md)
    print(f"\n저장: {p['out']}/measurement_1-9.md, .json")

if __name__ == "__main__":
    main()
