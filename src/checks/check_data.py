"""
데이터 정상성 검사 — 체크리스트 1-11 (로드 점검)

측정만 한다. 판정·해석은 확정 정리에서.
CI(김하루 3-6-1)로 옮길 것을 전제로, 매번 돌 수 있는 검사만 넣는다.

실행:
    RAG_ROOT=/srv/rfp python3 src/checks/check_data.py

검사 항목
    1. md 로드      — UTF-8 디코딩 / 빈 파일        -> 정상 로드율 %
    2. 개수 정합    — raw 개수 == md 개수
    3. 파일명 정규화 — NFD 형태 / NFC 후 매칭 실패   (0이어야 정상)
    4. CSV 인코딩   — 판별 / 디코딩 실패 행
    5. 전처리본     — 자리만. 1-19-1 검사기와의 범위 정리 후 채운다
    6. OCR 검토 필요 — parse_file.py의 kordoc qualitySummary.needsOcr 신호
                      (로드 자체는 성공이라 게이트 위반으로는 안 세고 별도 지표로만 본다)
                      ⏸ 현재 비활성 — 이 신호를 만드는 kordoc_convert.mjs가
                        아직 도입되지 않았다 (Node 18 필요, 서버는 v12.22.9)
                        
종료 코드:
    0  게이트 위반 없음
    1  위반 있음 (1-19-1 — 종료코드 0일 때만 공식 공개)
"""
import csv
import io
import os
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

# --- 경로: 절대경로 하드코딩 금지 (규약 §2-3) ---
try:
    RAG_ROOT = Path(os.environ["RAG_ROOT"])
except KeyError:
    sys.exit("환경변수 RAG_ROOT 가 없습니다.  예:  export RAG_ROOT=/srv/rfp")

RAW_DIR   = RAG_ROOT / "shared_data" / "raw" / "files"
CSV_PATH  = RAG_ROOT / "shared_data" / "raw" / "data_list.csv"
MD_DIR    = RAG_ROOT / "shared_data" / "interim" / "md"
PROC_DIR  = RAG_ROOT / "shared_data" / "processed"
NEEDS_OCR_PATH = RAG_ROOT / "shared_data" / "interim" / "needs_ocr_files.txt"

RAW_SUFFIXES = (".hwp", ".hwpx", ".pdf")
SHORT_DOC_THRESHOLD = 500          # 이 미만이면 '짧은 문서'로 따로 센다
ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

failures = []                       # 게이트 위반 사유. 비어 있어야 정상


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_nfd(s: str) -> bool:
    """파일명이 NFD(자모 분리) 형태로 저장돼 있는가."""
    return unicodedata.normalize("NFC", s) != s


def head(items, n=10):
    """긴 목록은 앞부분만. 전체는 파일로 따로 남긴다."""
    items = list(items)
    if len(items) <= n:
        return items
    return items[:n] + [f"... 외 {len(items) - n}건"]


def git_version() -> str:
    """규약 §2-4 — 이 숫자를 어느 코드로 냈는지 남긴다."""
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo, capture_output=True, text=True, timeout=5,
        ).stdout.strip() != ""
    except Exception:
        return "미상 (git 정보 없음)"
    if not commit:
        return "미상 (리포 밖에서 실행됨)"
    return f"{commit} (dirty: {str(dirty).lower()})"


# ---------------------------------------------------------------- 1. md 로드
def check_md_load():
    print("## 1. md 로드 (interim/md)\n")

    if not MD_DIR.is_dir():
        print(f"- 경로 없음: {MD_DIR}")
        failures.append("md 폴더 없음")
        return {}

    md_files = sorted(p for p in MD_DIR.iterdir() if p.suffix.lower() == ".md")
    total = len(md_files)
    print(f"- 대상: **{total}건**")

    decode_fail, empty, short, lengths = [], [], [], []

    for p in md_files:
        raw_bytes = p.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            alt = None
            for enc in ENCODING_CANDIDATES[2:]:          # cp949, euc-kr
                try:
                    raw_bytes.decode(enc)
                    alt = enc
                    break
                except UnicodeDecodeError:
                    continue
            decode_fail.append((p.name, f"{e.reason} / 대체: {alt or '판별 실패'}"))
            continue

        n = len(text)
        lengths.append(n)
        if n == 0:
            empty.append(p.name)
        elif n < SHORT_DOC_THRESHOLD:
            short.append((p.name, n))

    ok = total - len(decode_fail) - len(empty)
    rate = (ok / total * 100) if total else 0.0

    print(f"- UTF-8 디코딩 성공: {total - len(decode_fail)}건 / 실패: {len(decode_fail)}건")
    if decode_fail:
        for name, why in head(decode_fail):
            print(f"    - {name} — {why}")
    print(f"- 0자: {len(empty)}건 {head(empty) if empty else ''}")
    print(f"- {SHORT_DOC_THRESHOLD}자 미만: {len(short)}건 {head(short) if short else ''}")

    if lengths:
        lengths.sort()
        print(f"- 글자 수: min {lengths[0]} / median {lengths[len(lengths)//2]} / max {lengths[-1]}")

    print(f"\n- ⭐ **md 정상 로드율: {rate:.1f}%**  ({ok}/{total})")
    print("  (분자 = 디코딩 성공 AND 0자 아님 · 분모 = md 파일 총 개수)")

    if decode_fail or empty:
        failures.append(f"md 로드 실패 {len(decode_fail) + len(empty)}건")

    print()
    return {p.stem: p for p in md_files}


# ------------------------------------------------------------ 2. 개수 정합
def check_count_match(md_map):
    print("## 2. 개수 정합 (raw ↔ md)\n")

    if not RAW_DIR.is_dir():
        print(f"- 경로 없음: {RAW_DIR}")
        failures.append("raw 폴더 없음")
        return {}

    raw_files = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in RAW_SUFFIXES
    )
    ext = Counter(p.suffix.lower() for p in raw_files)

    print(f"- raw: **{len(raw_files)}건** {dict(ext)}")
    print(f"- md : **{len(md_map)}건**")

    if len(raw_files) == len(md_map):
        print("- 판정: 일치 ✅")
    else:
        print(f"- 판정: **불일치 (차이 {abs(len(raw_files) - len(md_map))}건)** ⚠️")
        print("  ⚠️ 이 검사는 실패를 감지하지만 원인은 알려주지 않는다.")
        print("     원인 진단은 변환 스크립트의 실패 기록이 있어야 한다 → 1-19-1")
        failures.append("raw/md 개수 불일치")

    print()
    return {p.stem: p for p in raw_files}


# ------------------------------------------------- 3. 파일명 정규화 (별도 지표)
def check_filename_normalization(raw_map, md_map):
    """로드율 분자에 넣지 않는다. 파일 문제가 아니라 경로 문자열 문제라
    한 숫자에 섞으면 CI가 울렸을 때 어디를 볼지 모른다. (1-11 결정 (a))"""
    print("## 3. 파일명 정규화 — 별도 지표\n")

    raw_nfd = [s for s in raw_map if is_nfd(s)]
    md_nfd  = [s for s in md_map  if is_nfd(s)]

    print(f"- NFD 형태 파일명 — raw {len(raw_nfd)}건 / md {len(md_nfd)}건")
    if raw_nfd:
        print(f"    raw 예시: {head(raw_nfd, 5)}")
    if md_nfd:
        print(f"    md  예시: {head(md_nfd, 5)}")

    # 정규화 없이 맞춰볼 때
    naive_miss = [s for s in raw_map if s not in md_map]
    # NFC 정규화 후 맞춰볼 때
    md_nfc = {nfc(s) for s in md_map}
    nfc_miss = [s for s in raw_map if nfc(s) not in md_nfc]

    print(f"- 정규화 없이 매칭 실패: {len(naive_miss)}건 {head(naive_miss, 5) if naive_miss else ''}")
    print(f"- ⭐ **NFC 정규화 후 매칭 실패: {len(nfc_miss)}건**  ← 0이어야 정상")
    if nfc_miss:
        print(f"    {head(nfc_miss, 5)}")
        failures.append(f"NFC 후에도 매칭 실패 {len(nfc_miss)}건")

    if naive_miss and not nfc_miss:
        print(f"  → 정규화만으로 {len(naive_miss)}건이 해소된다. 1-17 NFC 규칙의 근거.")

    print()


# ---------------------------------------------------------- 4. CSV 인코딩
def check_csv_encoding():
    print("## 4. CSV 인코딩 (raw/data_list.csv)\n")

    if not CSV_PATH.is_file():
        print(f"- 파일 없음: {CSV_PATH}")
        failures.append("data_list.csv 없음")
        print()
        return

    blob = CSV_PATH.read_bytes()
    detected, fail_pos = None, None

    for enc in ENCODING_CANDIDATES:
        try:
            text = blob.decode(enc)
            detected = enc
            break
        except UnicodeDecodeError as e:
            if fail_pos is None:
                fail_pos = e.start

    if detected is None:
        print(f"- **판별 실패** — 후보 {ENCODING_CANDIDATES} 전부 디코딩 불가")
        print(f"    최초 실패 바이트 위치: {fail_pos}")
        failures.append("CSV 디코딩 실패")
        print()
        return

    print(f"- 판별 결과: **{detected}**")
    if detected in ("cp949", "euc-kr"):
        print("  ⚠️ UTF-8이 아니다. 읽는 코드마다 encoding 을 명시하지 않으면")
        print("     사람마다 다른 값을 보게 된다 → 1-17 / 1-18")

    # 행 단위 디코딩 실패 (전체가 성공했으면 0이지만, 개수와 컬럼은 확인해 둔다)
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
        rows = list(reader)
        print(f"- 컬럼 {len(header)}개 / 데이터 {len(rows)}행")
        ragged = [i + 2 for i, r in enumerate(rows) if len(r) != len(header)]
        print(f"- 컬럼 수가 헤더와 다른 행: {len(ragged)}건 {head(ragged, 5) if ragged else ''}")
        if ragged:
            failures.append(f"CSV 행 컬럼 수 불일치 {len(ragged)}건")
    except csv.Error as e:
        print(f"- CSV 파싱 오류: {e}")
        failures.append("CSV 파싱 오류")

    print("- 디코딩 실패 행: 0건 (파일 전체가 한 인코딩으로 디코딩됨)")
    print()


# ------------------------------------------------------------ 5. 전처리본
def check_processed():
    print("## 5. 전처리본 (processed) — 자리만\n")
    if PROC_DIR.is_dir():
        subs = sorted(p.name for p in PROC_DIR.iterdir() if p.is_dir())
        print(f"- 하위 폴더: {subs if subs else '없음'}")
    else:
        print(f"- 경로 없음: {PROC_DIR}")
    print("- ⏸ 전처리본이 코퍼스 정본이다. 1-17은 확정됐으나(rules_v2.yaml · 검증 12항목 통과),")
    print("  1-19-1 검사기가 이미 같은 항목을 보고 있을 수 있어 범위 정리 후 채운다.")
    print("  후보 — 이미지 링크 제거 확인 / NFC 적용 확인 / 문서 수 대조\n")


# ------------------------------------------------------- 6. OCR 검토 필요
def check_ocr_candidates(md_map):
    """로드율 분자에 넣지 않는다. md 자체는 정상 생성됐고, 내용 품질이
    낮을 수 있다는 신호일 뿐이라 게이트 위반과는 성격이 다르다. (3번과 동일한 이유)"""
    print("## 6. OCR 검토 필요 — 변환 품질 신호\n")

    if not NEEDS_OCR_PATH.is_file():
        print(f"- 목록 없음: {NEEDS_OCR_PATH}")
        print("  ⏸ 현재 parse_file.py는 이 신호를 만들지 않는다.")
        print("     kordoc_convert.mjs 도입 후 활성화된다 (Node 18 필요 — 서버는 v12).")
        print("     따라서 '목록 없음'을 'OCR 대상 0건'으로 읽으면 안 된다.\n")
        return

    names = [n.strip() for n in NEEDS_OCR_PATH.read_text(encoding="utf-8").splitlines() if n.strip()]
    stems = {Path(n).stem for n in names}
    stale = stems - set(md_map)

    print(f"- 목록: **{len(names)}건** ({NEEDS_OCR_PATH})")
    if names:
        print(f"    {head(names, 5)}")
    if stale:
        print(f"  ⚠️ md_map과 매칭 안 되는 항목 {len(stale)}건 — 목록이 오래됐을 수 있음 {head(sorted(stale), 5)}")

    print()


def main():
    print("# 데이터 정상성 검사 — 체크리스트 1-11\n")
    print(f"측정 시각: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"코드 버전: {git_version()}")
    print(f"RAG_ROOT: `{RAG_ROOT}`\n")
    print("> 측정만 함. 판정·해석은 확정 정리에서.\n")
    print("---\n")

    md_map  = check_md_load()
    raw_map = check_count_match(md_map)
    if raw_map and md_map:
        check_filename_normalization(raw_map, md_map)
    check_csv_encoding()
    check_processed()
    check_ocr_candidates(md_map)

    print("---\n")
    print("## 게이트 판정\n")
    if failures:
        print(f"- **위반 {len(failures)}건**")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("- 위반 없음 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())