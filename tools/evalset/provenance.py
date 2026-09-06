"""재현성 기록 — 다른 작업자가 같은 공식 자료로 같은 결과를 낼 수 있게 한다.

★기준 dev 커밋만 적으면 재현이 안 된다. 정답을 만든 코드가 아직 커밋되지 않았기
  때문이다. 그래서 **작업 패치 자체의 SHA-256** 과 정책 파일·채점기 코드·심판
  프롬프트의 해시를 함께 고정한다.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SRV = Path("/srv/rfp")
OFFICIAL_INPUTS = {
    "registry_v2": SRV / "shared_data/processed/document_registry_v2/document_registry_v2.json",
    "identity_v2": SRV / "shared_data/processed/document_registry_v2/document_identity_v2.csv",
    "extraction_table_v3": SRV / "shared_data/processed/rfp_extraction_table_v3/extraction_table_v3.json",
    "extraction_metadata": SRV / "shared_data/processed/rfp_extraction_table_v3/extraction_metadata.json",
    "chunks_v3": SRV / "shared_data/processed/chunks_v3/chunks.jsonl",
    "chunks_v3_version": SRV / "shared_data/processed/chunks_v3/VERSION.txt",
    "index_v2_tag": SRV / "shared_data/processed/index_v2/index_tag.json",
    "official_items_v1": SRV / "evalset/v1/items.jsonl",
}


def sha256_file(path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(worktree: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(worktree), *args],
                          capture_output=True, text=True).stdout


def working_patch(worktree: Path, extra_paths=()) -> str:
    """추적 변경(diff) + 미추적 파일 내용을 하나의 문자열로 모은다.

    ★`git diff` 만으로는 부족하다 — 정답 생성 코드가 통째로 미추적 파일인 경우가
      있고, 그러면 패치에 한 줄도 안 담긴다. 미추적 파일도 내용째 붙인다.
    """
    parts = [_git(worktree, "diff", "--no-color")]
    untracked = [l for l in _git(worktree, "ls-files", "--others",
                                 "--exclude-standard").splitlines() if l.strip()]
    for rel in sorted(untracked):
        p = worktree / rel
        if not p.is_file() or "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            body = f"<binary {sha256_file(p)}>"
        parts.append(f"\n--- /dev/null\n+++ b/{rel}\n{body}")
    for extra in extra_paths:
        p = Path(extra)
        if p.is_file():
            parts.append(f"\n--- /dev/null\n+++ b/{p.name}\n{p.read_text(encoding='utf-8')}")
    return "".join(parts)


def build(*, worktree, candidate_items, policy_files=(), scorer_files=(),
          judge_prompts=(), base_commit=None, generated_at=None) -> dict:
    worktree = Path(worktree)
    patch = working_patch(worktree)
    return {
        "generated": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_dev_commit": (base_commit or _git(worktree, "rev-parse", "HEAD").strip()),
        "base_dev_commit_subject": _git(worktree, "log", "-1", "--format=%s").strip(),
        "branch": _git(worktree, "branch", "--show-current").strip(),
        "commits_since_base": int(_git(worktree, "rev-list", "--count",
                                       "HEAD..HEAD").strip() or 0),
        # ★미커밋 생성 코드를 고정하는 값. 이게 없으면 재현이 불가능하다.
        "working_patch_sha256": sha256_text(patch),
        "working_patch_bytes": len(patch.encode("utf-8")),
        "policy_files": {str(Path(p).name): sha256_file(p) for p in policy_files},
        "scorer_files": {str(Path(p).relative_to(worktree)): sha256_file(p)
                         for p in scorer_files if Path(p).is_file()},
        "judge_prompts": {str(Path(p).name): sha256_file(p) for p in judge_prompts},
        "candidate_evalset_sha256": sha256_file(candidate_items),
        "official_inputs": {k: sha256_file(v) for k, v in OFFICIAL_INPUTS.items()},
    }


def write(path, **kw) -> dict:
    data = build(**kw)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
