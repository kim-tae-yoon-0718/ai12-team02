"""
grader.versioning — 평가셋 버전의 출처(VERSION.txt).

v0.2(임현진)에서 문항 레벨 `schema_version` 필드가 폐지되고 **파일 단위**
`VERSION.txt` 로 이동했다. 팀 확정 위치·포맷(2026-08-30):

    $RAG_ROOT/evalset/v1/VERSION.txt

        evalset: v1
        corpus: [대기]
        created: 2026-08-30

`key: value` 여러 줄 포맷이다(구 단일 문자열 포맷도 계속 읽는다).

★경로 규칙(팀 실험 인프라 규약 §2-3): 절대 경로 하드코딩 금지 — `RAG_ROOT`
환경변수 + 코드 조립. `RAG_ROOT` 가 없으면 저장소 루트의 `VERSION.txt` 로 폴백해
개발/테스트에서 배관이 죽지 않게 한다. 아무 데도 없으면 "UNKNOWN".
"""

from __future__ import annotations

import os
from pathlib import Path

_FILENAME = "VERSION.txt"
_EVALSET_SUBPATH = ("evalset", "v1")
_FALLBACK = "UNKNOWN"


def _candidate_paths(path: str | Path | None) -> list[Path]:
    if path is not None:
        return [Path(path)]
    out: list[Path] = []
    rag_root = os.environ.get("RAG_ROOT")
    if rag_root:
        out.append(Path(rag_root, *_EVALSET_SUBPATH, _FILENAME))
    # 개발/테스트 폴백 — 저장소 루트 (src/grader/../..)
    out.append(Path(__file__).resolve().parent.parent.parent / _FILENAME)
    return out


def read_versions(path: str | Path | None = None) -> dict[str, str]:
    """VERSION.txt 를 파싱해 {key: value} 로 돌려준다.

    `key: value` 줄들을 읽는다. 구 포맷(버전 문자열 한 줄)은 {"schema_version": "<줄>"}
    로 담는다. 파일이 없으면 빈 dict.
    """
    for p in _candidate_paths(path):
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8-sig").strip()
        if not raw:
            return {}
        out: dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
            elif "schema_version" not in out:
                out["schema_version"] = line
        return out
    return {}


def read_schema_version(path: str | Path | None = None) -> str:
    """평가셋 버전 문자열. VERSION.txt 의 `evalset:` → `schema_version:` 순으로 찾고,
    없으면 "UNKNOWN"."""
    v = read_versions(path)
    return v.get("evalset") or v.get("schema_version") or _FALLBACK
