"""최종셋 유출 검사 — 평가 문항 텍스트가 프롬프트·코드 등 추적 파일에 새어 들어갔는지.

임현진 `check_evalset.check_no_leakage` 와 같은 목적. few-shot 예시나 프롬프트에 평가
질문이 그대로 들어가면 부정행위(【25】). CI(3-6-1)에서 성능 평가보다 먼저 값싸게 잡는다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from ..models import EvaluationItem


def _tracked_files(repo_root: Path) -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        return [repo_root / p for p in out.stdout.splitlines()]
    except Exception:
        return []


# 평가셋이 놓일 법한 경로 — 유출 검사에서 당연히 제외(문항이 여기 있는 건 정상)
_EVALSET_DIRS = ("/tests/fixtures/", "/data/", "/examples/", "/evalset/")


def check_leakage(items: Iterable[EvaluationItem], repo_root: str | Path,
                  min_len: int = 12, exclude_paths: Iterable[str | Path] = ()) -> list[str]:
    """추적 파일에서 각 문항의 question 문자열이 그대로 나오는지 본다 —
    프롬프트·코드에 평가 질문이 새면 부정행위(【25】).

    min_len: 너무 짧은 질문("이거?" 등)은 우연히 겹칠 수 있어 건너뛴다.
    exclude_paths: 평가셋 파일 자체 등. 평가셋 경로·tests/fixtures·data 는 자동 제외.
    """
    repo_root = Path(repo_root)
    needles = {it.id: it.question for it in items
               if it.question and len(it.question) >= min_len}
    if not needles:
        return []
    skip = {str(Path(p).resolve()) for p in exclude_paths}

    problems: list[str] = []
    for path in _tracked_files(repo_root):
        rp = str(path.resolve())
        if rp in skip or any(d in rp.replace("\\", "/") for d in _EVALSET_DIRS):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for qid, q in needles.items():
            if q in text:
                rel = path.relative_to(repo_root) if repo_root in path.parents else path
                problems.append(f"[유출] {qid}: 문항 텍스트가 추적 파일 {rel} 에 있음 — {q!r}")
    return problems
