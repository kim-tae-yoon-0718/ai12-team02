"""기준 커밋에 적용 가능한 patch 생성 + 적용 검증.

★실제 작업 worktree 의 인덱스를 건드리지 않는다. 작업본을 임시 폴더로 복사한 뒤
  거기서 `git add -N`(intent-to-add)을 걸어 미추적 파일까지 patch 에 담는다.
  그 다음 기준 커밋만 체크아웃한 **깨끗한 폴더**에 실제로 적용해 보고,
  적용 결과가 작업본과 파일 단위로 같은지 해시로 확인한다.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", "node_modules"}
# 캐시는 재현 대상이 아니다 — 지우고 다시 만들면 되는 산출물이라 patch 에 담지 않는다.
EXCLUDE_PREFIXES = (".cache/", "data/outputs/")
# ★.gitignore 가 data/ 전체를 무시한다(.gitignore:234). 그런데 config/grader.yaml 이
#   가리키는 심판 프롬프트와 gold 자료가 그 안에 있다. 무시된 채로 두면 patch 로도
#   전달되지 않아 다른 사람이 같은 설정으로 실행할 수 없다 — 강제로 포함시킨다.
FORCE_INCLUDE = ("data",)   # data/outputs 는 EXCLUDE_PREFIXES 로 다시 빠진다


def sh(*args, cwd=None, check=True):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"실패: {' '.join(args)}\n{p.stdout}\n{p.stderr}")
    return p


def _tracked_files(root: Path) -> dict[str, str]:
    """작업본의 (추적+미추적) 파일 해시 — 캐시·빌드 산출물 제외."""
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in EXCLUDE_DIRS for part in p.parts) or rel.endswith(".pyc"):
            continue
        if rel.startswith(EXCLUDE_PREFIXES):
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def build(worktree: Path, base: str, out_patch: Path) -> dict:
    worktree = Path(worktree)
    with tempfile.TemporaryDirectory() as td:
        mirror = Path(td) / "mirror"
        # ① 작업본 복사 (실제 worktree 는 읽기만 한다)
        shutil.copytree(worktree, mirror, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        # 복사본을 독립 저장소로 만든다 — 원본 .git 링크를 끊는다
        gitfile = mirror / ".git"
        if gitfile.exists():
            (shutil.rmtree if gitfile.is_dir() else Path.unlink)(gitfile)
        _init_with_alternates(mirror, worktree)
        # 기준 커밋을 인덱스에 올린다(작업 파일은 그대로) — 임시 복사본이라 안전하다.
        sh("git", "reset", "-q", "--mixed", base, cwd=mirror)
        # ② 미추적 파일까지 patch 에 담기게 intent-to-add
        sh("git", "add", "-A", "-N", "--", ".", cwd=mirror)
        for forced in FORCE_INCLUDE:
            if (mirror / forced).exists():
                sh("git", "add", "-f", "-N", "--", forced, cwd=mirror)
        for skip in EXCLUDE_PREFIXES:
            if (mirror / skip).exists():
                sh("git", "rm", "-q", "--cached", "-r", "--ignore-unmatch", "--",
                   skip.rstrip("/"), cwd=mirror, check=False)
        diff = sh("git", "diff", "--binary", "--no-color", base, cwd=mirror).stdout
    out_patch.write_text(diff, encoding="utf-8")
    return {"patch_path": str(out_patch), "patch_bytes": len(diff.encode("utf-8")),
            "patch_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            "base": base}


def _init_with_alternates(target: Path, worktree: Path) -> None:
    """빈 저장소를 만들되 원본 오브젝트 저장소를 alternates 로 참조한다.

    ★fetch 로 임의 SHA 를 가져오는 건 서버 설정에 막힌다. alternates 를 걸면
      원본을 **읽기만** 하면서 기준 커밋을 그대로 쓸 수 있다.
    """
    sh("git", "init", "-q", cwd=target)
    objects = _repo_of(worktree) / "objects"
    alt = target / ".git/objects/info/alternates"
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_text(str(objects) + "\n", encoding="utf-8")


def _repo_of(worktree: Path) -> Path:
    common = sh("git", "-C", str(worktree), "rev-parse", "--git-common-dir").stdout.strip()
    p = Path(common)
    return p if p.is_absolute() else (worktree / p).resolve()


def verify(worktree: Path, base: str, patch: Path) -> dict:
    """기준 커밋만 있는 깨끗한 폴더에 patch 를 적용해 작업본과 같아지는지 확인."""
    worktree = Path(worktree)
    want = _tracked_files(worktree)
    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / "clean"
        clean.mkdir()
        _init_with_alternates(clean, worktree)
        sh("git", "reset", "-q", "--mixed", base, cwd=clean)
        sh("git", "checkout", "-q", base, "--", ".", cwd=clean)
        applied = sh("git", "apply", "--whitespace=nowarn", str(patch), cwd=clean, check=False)
        if applied.returncode != 0:
            return {"applies": False, "error": (applied.stdout + applied.stderr)[:2000]}
        got = _tracked_files(clean)
    only_want = sorted(set(want) - set(got))
    only_got = sorted(set(got) - set(want))
    differ = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    return {"applies": True, "n_files_expected": len(want), "n_files_after_apply": len(got),
            "missing_after_apply": only_want, "unexpected_after_apply": only_got,
            "content_mismatch": differ,
            "identical": not (only_want or only_got or differ)}


if __name__ == "__main__":
    wt, base, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    info = build(wt, base, out)
    info["verification"] = verify(wt, base, out)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    sys.exit(0 if info["verification"].get("identical") else 1)
