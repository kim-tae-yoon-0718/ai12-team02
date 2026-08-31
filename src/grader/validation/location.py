"""location 좌표 검증 — document / section / ref_no (2-9 확정 단위).

채점기가 좌표를 `ref_no` 단위까지 비교하므로(retrieval.DEFAULT_PRECISION), 평가셋의
`location.ref_no` 가 비어 있거나 자리표시자면 좌표 채점(3-4-3)이 성립하지 않는다.
"""

from __future__ import annotations

from typing import Iterable

from ..models import EvaluationItem, _PART_OF_RE
from ..normalize import match_location  # noqa: F401  (편의 재노출)

_PLACEHOLDERS = {"", "<문서id>", "<문서ID>", "TODO", "[대기]", "?", "-", "n/a", "na"}


def _is_placeholder(v: str | None) -> bool:
    return v is None or str(v).strip().lower() in {p.lower() for p in _PLACEHOLDERS}


def check_locations(items: Iterable[EvaluationItem], *, require_ref_no: bool = True) -> list[str]:
    """location 이 있는 문항의 좌표 3필드가 실제 값인지 확인한다.

    ★팀 확정 좌표 형식(2026-08-31): `ref_no = "표 7"`, part/of 미포함. 옛 형식
      `표 7 (2/3)` 가 남아 있으면 채점은 무해하지만(매칭 시 자동으로 뗌) 평가셋을
      정리하도록 경고한다.
    """
    problems: list[str] = []
    for it in items:
        loc = it.location
        if loc is None:
            continue
        if _is_placeholder(loc.document):
            problems.append(f"[좌표] {it.id}: location.document 가 비었거나 자리표시자 ({loc.document!r})")
        if _is_placeholder(loc.section):
            problems.append(f"[좌표] {it.id}: location.section 가 비었거나 자리표시자 ({loc.section!r})")
        if require_ref_no and _is_placeholder(loc.ref_no):
            problems.append(
                f"[좌표] {it.id}: location.ref_no 가 비었거나 자리표시자 ({loc.ref_no!r}) — "
                f"ref_no 단위 좌표 채점(3-4-3) 불가")
        elif _PART_OF_RE.search(str(loc.ref_no or "")):
            problems.append(
                f"[좌표] {it.id}: location.ref_no 에 조각 표기 (N/M) 가 있음 ({loc.ref_no!r}) — "
                f"팀 확정 형식은 part/of 제외 (예: '표 7'). 채점은 자동으로 뗀다")
    return problems
