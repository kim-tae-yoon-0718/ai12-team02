"""
git 상태 자동 기록 — 팀 규약 2-4.
실행 진입점에서 호출해 결과 config.yaml에 git_commit·git_dirty를 남긴다.
git_dirty=true는 "이 실험은 재현 불가"라는 뜻 — 최종(발표용) 실험은
반드시 커밋 후 git_dirty=false 상태로 돌려야 한다.
"""
from __future__ import annotations
import subprocess


def get_git_info() -> dict[str, str | bool]:
    try:
        commit = subprocess.getoutput("git log -1 --format=%h").strip()
        dirty = subprocess.getoutput("git status --short").strip() != ""
    except Exception:
        commit, dirty = "unknown", True
    return {"git_commit": commit, "git_dirty": dirty}


def warn_if_dirty(purpose: str = "실험") -> None:
    info = get_git_info()
    if info["git_dirty"]:
        print(
            f"⚠️  git_dirty=true — 커밋 안 된 변경사항이 있습니다. "
            f"이 {purpose}은 재현 불가로 기록됩니다. "
            f"발표용 최종 실험이라면 지금 커밋부터 하세요."
        )
