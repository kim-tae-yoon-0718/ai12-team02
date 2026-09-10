"""동결된 최종 패치의 SHA-256 이 VERSION.txt · provenance.json · APPLY_VERIFICATION.json ·
CANDIDATE_FINGERPRINTS.json 에 **같은 값**으로 적혀 있고 실제 파일 해시와 일치하는지 검사한다.
하나라도 다르면 종료코드 1 — 패치가 바뀐 뒤 옛 해시가 남는 일을 막는다."""
import hashlib, json, re, sys
from pathlib import Path
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
patch = OUT / "patch/worktree.patch"
actual = hashlib.sha256(patch.read_bytes()).hexdigest()
ver = OUT / "evalset/candidate/VERSION.txt"
m = re.search(r"^final_patch_sha256:\s*(\S+)", ver.read_text(encoding="utf-8"), re.M)
recorded = {
    "VERSION.txt": m.group(1) if m else None,
    "provenance.json": json.loads((OUT / "evalset/provenance.json").read_text(encoding="utf-8")).get("final_patch_sha256"),
    "APPLY_VERIFICATION.json": json.loads((OUT / "patch/APPLY_VERIFICATION.json").read_text(encoding="utf-8")).get("patch_sha256"),
    "CANDIDATE_FINGERPRINTS.json": json.loads((OUT / "extraction/CANDIDATE_FINGERPRINTS.json").read_text(encoding="utf-8")).get("final_patch", {}).get("sha256"),
}
# VERSION.txt 중복 키 검사
keys = [l.split(":")[0].strip() for l in ver.read_text(encoding="utf-8").splitlines() if re.match(r"^[a-z_]+:", l)]
dups = sorted({k for k in keys if keys.count(k) > 1})
ok = all(v == actual for v in recorded.values()) and not dups
print(json.dumps({"actual_patch_sha256": actual, "recorded": recorded, "duplicate_keys": dups, "all_match": ok}, ensure_ascii=False, indent=2))
sys.exit(0 if ok else 1)
