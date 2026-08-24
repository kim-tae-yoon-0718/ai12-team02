"""
원본 hwp/pdf 파일을 kordoc을 통해 .md로 변환
"""
import subprocess
import sys
from pathlib import Path

DATA_DIR  = "/srv/rfp/shared_data/raw/files/"            # 원본 RFP 폴더
META_PATH = "/srv/rfp/shared_data/raw/data_list.csv"     # 메타데이터
OUT_DIR   = "/srv/rfp/shared_data/interim/md"            # 중간 산출물 저장위치
KORDOC = Path.home() / "tools" / "kordoc-run"            # kordoc 설치된 위치
BATCH = 10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        f for f in sorted(DATA_DIR.iterdir())
        if f.suffix.lower() in (".hwp", ".hwpx", "pdf")
        and not (OUT_DIR / f"{f.stem}.md").exists()      # 이미 반환된 것 제외
    ]
    print(f"원본 RFP: {len(list(DATA_DIR.iterdir()))}건 / 변환 대상 {len(targets)}건")
    if not targets:
        return

    failed = []
    for i in range(0, len(targets), BATCH):
        batch = targets[i: BATCH + 1]
        r = subprocess.run(
            ["npx", "kordoc", *[str(f) for f in batch], "-d", f"{OUT_DIR}/"],
            cwd=KORDOC,
            capture_output=True,
            text=True
        )
        for f in batch:
            ok = (OUT_DIR / f"{f.stem}.md").exists()
            print(f"[{batch + i.index(f) + 1} / {len(targets)}] {f.name} ... {'OK' if ok else 'FAIL'}")
            if not ok:
                failed.append(f.name)
        if r.returncode != 0 and not any((OUT_DIR / f"{f.stem}.md").exists() for f in batch):
            print(r.stderr.strip()[:500], file=sys.stedrr)

    print(f"\n성공 {len(targets) - len(failed)}건 / 실패 {len(failed)}건")
    if failed:
        Path("failed_files.txt").write_text("\n".join(failed), encoding="utf-8")
        print("실패 목록: failed_files.txt")


if __name__ == "__main__":
    main()
