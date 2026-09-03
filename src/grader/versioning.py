"""
grader.versioning — 평가 자산 버전의 출처(VERSION.txt).

v0.2(임현진)에서 문항 레벨 `schema_version` 필드가 폐지되고 **파일 단위** VERSION.txt 로
이동했다. 자산마다 VERSION.txt 가 있고 포맷이 조금씩 다르다:

    $RAG_ROOT/evalset/v1/VERSION.txt                     evalset: v1 / corpus: [대기] / created: ...
    $RAG_ROOT/shared_data/processed/corpus_v2/VERSION.txt  corpus version : v2 / preprocess version : v2
    $RAG_ROOT/shared_data/processed/chunks_v1/VERSION.txt  chunking version : v1 / corpus version : v2 / table version : v2

`key : value` 여러 줄 포맷(콜론 앞뒤 공백 허용). 키의 끝 " version" 은 떼어 6-자산 이름
(corpus/preprocess/table/evalset)으로 정규화한다. 구 단일 문자열 포맷도 계속 읽는다.

★경로 규칙(팀 규약 §2-3): 절대 경로 하드코딩 금지 — `RAG_ROOT` 환경변수 + 코드 조립.
`RAG_ROOT` 가 없으면 저장소 루트 VERSION.txt 로 폴백. 아무 데도 없으면 "UNKNOWN".
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

_FALLBACK = "UNKNOWN"
_SENTINELS = {"", "unknown", "[대기]", "todo", "null", "none", "false", "true"}
# VERSION.txt 에서 뽑을 키만 화이트리스트 (그 외 자유 텍스트 줄은 무시).
# 끝 " version" 은 파싱에서 이미 제거됨. chunking 은 6-자산이 아니라 참고 정보로만 담는다.
_WANT = {"evalset", "corpus", "preprocess", "table", "index", "scorer", "chunking"}
_VER_RE = re.compile(r"^v?\d+(?:\.\d+)*$", re.I)


def _norm_key(k: str) -> str:
    k = k.strip().lower()
    if k.endswith(" version"):
        k = k[: -len(" version")].strip()
    return k


def _parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = _norm_key(k), v.strip()
            if k in _WANT and v.lower() not in _SENTINELS:
                out.setdefault(k, v)
        elif "schema_version" not in out and _VER_RE.match(line):
            out["schema_version"] = line
    return out


def _version_files(path: str | Path | None) -> list[Path]:
    if path is not None:
        return [Path(path)]
    out: list[Path] = []
    root = os.environ.get("RAG_ROOT")
    if root:
        root = Path(root)
        for pat in ("evalset/*/VERSION.txt",
                    "shared_data/processed/chunks_*/VERSION.txt",
                    "shared_data/processed/corpus_*/VERSION.txt"):
            # 높은 버전 폴더 우선 (chunks_v2 > chunks_v1)
            out += [Path(p) for p in sorted(glob.glob(str(root / pat)), reverse=True)]
    out.append(Path(__file__).resolve().parent.parent.parent / "VERSION.txt")  # dev 폴백
    return out


def _latest_table_version() -> str | None:
    """rfp_extraction_table_v3 처럼 버전이 올라간다 — 가장 높은 v 를 추출 테이블 실사용값으로."""
    root = os.environ.get("RAG_ROOT")
    if not root:
        return None
    dirs = sorted(glob.glob(str(Path(root) / "shared_data/processed/rfp_extraction_table_v*")))
    return ("v" + dirs[-1].rsplit("_v", 1)[-1]) if dirs else None


def read_versions(path: str | Path | None = None) -> dict[str, str]:
    """VERSION.txt 여러 개를 읽어 6-자산 이름 기준 {key: value} 로 합친다(먼저 나온 값 우선)."""
    merged: dict[str, str] = {}
    for p in _version_files(path):
        if not p.exists():
            continue
        for k, v in _parse(p.read_text(encoding="utf-8-sig").strip()).items():
            merged.setdefault(k, v)
    # 추출 테이블은 별도 디렉토리 버전이 실사용값 — chunks VERSION.txt 의 'table version' 보다 우선.
    if path is None:
        t = _latest_table_version()
        if t:
            merged["table"] = t
    return merged


def read_schema_version(path: str | Path | None = None) -> str:
    """평가셋 버전 문자열 — VERSION.txt 의 `evalset:` → `schema_version:` 순, 없으면 "UNKNOWN"."""
    v = read_versions(path)
    return v.get("evalset") or v.get("schema_version") or _FALLBACK
