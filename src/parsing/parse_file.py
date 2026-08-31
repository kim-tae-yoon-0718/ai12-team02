"""
원본 hwp/pdf 파일을 kordoc을 통해 .md로 변환

kordoc CLI 대신 src/parsing/kordoc_convert.mjs를 거친다.
CLI로는 qualitySummary(needsOcr 등)를 받을 수 없기 때문이다.

실행:
    RAG_ROOT=/srv/rfp python3 src/parsing/parse_file.py
    RAG_ROOT=/srv/rfp python3 src/parsing/parse_file.py --force   # 전량 재변환

산출물:
    <RAG_ROOT>/shared_data/interim/md/*.md
    <RAG_ROOT>/shared_data/interim/failed_files.txt
    <RAG_ROOT>/shared_data/interim/needs_ocr_files.txt

종료 코드:
    0  전부 성공
    1  변환 실패가 있음 (1-19-1 — 종료코드 0일 때만 공식 공개)
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# --- 경로: 절대경로 하드코딩 금지 (규약 §2-3) ---
try:
    RAG_ROOT = Path(os.environ["RAG_ROOT"])
except KeyError:
    sys.exit("환경변수 RAG_ROOT 가 없습니다.  예:  export RAG_ROOT=/srv/rfp")

DATA_DIR   = RAG_ROOT / "shared_data" / "raw" / "files"        # 원본 RFP 폴더
META_PATH  = RAG_ROOT / "shared_data" / "raw" / "data_list.csv"  # 메타데이터
OUT_DIR    = RAG_ROOT / "shared_data" / "interim" / "md"       # 중간 산출물
REPORT_DIR = OUT_DIR.parent                                    # 실패·OCR 목록

# kordoc 라이브러리(node_modules)가 설치된 공용 폴더. 여기를 cwd로 실행한다.
KORDOC = RAG_ROOT / "shared_data" / "tools" / "kordoc-run"
# 변환 스크립트는 리포 안에 둔다. 개인 홈에 있으면 다른 사람이 재현할 수 없다.
CONVERT_SCRIPT = Path(__file__).resolve().parent / "kordoc_convert.mjs"

RAW_SUFFIXES = (".hwp", ".hwpx", ".pdf")
BATCH = 10


def git_version() -> str:
    """규약 §2-4 — 이 산출물을 어느 코드로 냈는지 남긴다."""
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


def write_report(path: Path, names: list) -> None:
    """실패 0건이어도 파일을 갱신한다.

    갱신하지 않으면 이전 실행의 목록이 남아, check_data.py가 이미 해소된
    문서를 계속 문제로 읽는다.
    """
    if names:
        path.write_text("\n".join(names) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="이미 변환된 문서도 다시 변환한다 "
                         "(1-17 — 규칙이 바뀌면 전 문서를 다시 처리)")
    args = ap.parse_args()

    for p in (DATA_DIR, KORDOC):
        if not p.is_dir():
            sys.exit(f"경로를 찾을 수 없습니다: {p}")
    if not CONVERT_SCRIPT.is_file():
        sys.exit(f"변환 스크립트를 찾을 수 없습니다: {CONVERT_SCRIPT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"실행 시각: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"코드 버전: {git_version()}")
    print(f"RAG_ROOT : {RAG_ROOT}")
    print(f"모드     : {'전량 재변환' if args.force else '증분 (미변환분만)'}\n")

    all_files = sorted(p for p in DATA_DIR.iterdir()
                       if p.suffix.lower() in RAW_SUFFIXES)
    targets = [f for f in all_files
               if args.force or not (OUT_DIR / f"{f.stem}.md").exists()]

    print(f"원본 RFP: {len(all_files)}건 / 변환 대상 {len(targets)}건")
    if not targets:
        write_report(REPORT_DIR / "failed_files.txt", [])
        return 0

    failed, needs_ocr = [], []

    for i in range(0, len(targets), BATCH):
        batch = targets[i:i + BATCH]
        r = subprocess.run(
            ["node", str(CONVERT_SCRIPT), *[str(f) for f in batch], "-d", str(OUT_DIR)],
            cwd=KORDOC, capture_output=True, text=True,
        )

        reports = {}
        for line in r.stdout.splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            reports[Path(entry["file"]).name] = entry

        for j, f in enumerate(batch):
            entry = reports.get(f.name)
            ok = (OUT_DIR / f"{f.stem}.md").exists()
            note = ""
            if not ok and entry:
                # 변환 스크립트가 남긴 실패 단계를 그대로 보여준다.
                note = f"  ({entry.get('stage', '?')}: {entry.get('thrown', '')[:120]})"
            print(f"[{i + j + 1} / {len(targets)}] {f.name} ... "
                  f"{'OK' if ok else 'FAIL'}{note}")
            if not ok:
                failed.append(f.name)
            elif entry and entry.get("needsOcr"):
                needs_ocr.append(f.name)

        if r.returncode != 0 and not reports:
            print(r.stderr.strip()[:500], file=sys.stderr)

    print(f"\n성공 {len(targets) - len(failed)}건 / 실패 {len(failed)}건 "
          f"/ OCR 검토 필요 {len(needs_ocr)}건")

    write_report(REPORT_DIR / "failed_files.txt", failed)
    write_report(REPORT_DIR / "needs_ocr_files.txt", needs_ocr)
    print(f"실패 목록        : {REPORT_DIR / 'failed_files.txt'}")
    print(f"OCR 검토 필요 목록: {REPORT_DIR / 'needs_ocr_files.txt'}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())