"""
grader.versioning — 평가셋 schema_version의 새 출처.

v0.1에서는 EvaluationItem.schema_version(문항마다 반복)이 출처였다.
v0.2(2026-08-27, 임현진)에서 이 필드가 폐지되고 저장소 루트의 VERSION.txt
하나로 옮겨졌다 — 문항마다 같은 문자열을 반복해서 들고 다닐 이유가 없었기 때문.

[가정 — 팀 확인 필요] VERSION.txt의 정확한 경로/포맷은 이번 마이그레이션
요청(필드 제거) 범위에 명시되지 않아 임시로 다음과 같이 잡았다:
  - 위치: 프로젝트 루트(pyproject.toml과 같은 위치)의 VERSION.txt
  - 포맷: 공백 없이 버전 문자열 한 줄(예: "v0.2")
파일이 없으면 예외를 던지지 않고 "UNKNOWN"으로 안전하게 폴백한다 — 이 저장소가
아직 VERSION.txt를 받지 못한 개발 단계에서 전체 배관이 죽지 않게 하기 위함이다
(check_data_sanity의 "버전 꼬리표 없음" 폴백과 같은 원칙).

configs/grader.yaml 최상단의 schema_version 키(GraderConfig.schema_version)는
이번에 다루는 "문항 레벨" schema_version과는 별개의, 현재 코드 어디에서도
소비되지 않는 값이다 — 이번 리팩터에서는 건드리지 않았다. VERSION.txt와
합칠지는 팀 확인이 필요하다.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_FILENAME = "VERSION.txt"
_FALLBACK = "UNKNOWN"


def read_schema_version(root: str | Path | None = None) -> str:
    """root/VERSION.txt를 읽어 공백을 제거한 문자열로 반환한다.

    root를 생략하면 이 파일 기준 프로젝트 루트(src/grader/../..)를 쓴다.
    파일이 없거나 비어 있으면 "UNKNOWN"을 반환한다.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent.parent
    path = base / _DEFAULT_FILENAME
    if not path.exists():
        return _FALLBACK
    content = path.read_text(encoding="utf-8").strip()
    return content or _FALLBACK
