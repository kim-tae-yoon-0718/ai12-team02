"""provenance 검증 — 6-자산(base.yaml §① 6칸)이 결과에 온전히 기록됐는지.

report.json 의 `manifest.provenance` 나 per_item 의 `provenance` dict 를 받아,
6개 필드가 정확한 이름으로 모두 존재하고 값이 None 이 아닌지 확인한다.
"""

from __future__ import annotations

from ..models import PROVENANCE_FIELDS


def check_provenance(provenance: dict | None, *, where: str = "manifest") -> list[str]:
    problems: list[str] = []
    if not isinstance(provenance, dict):
        return [f"[provenance] {where}: provenance 블록이 없다(dict 아님)"]
    keys = set(provenance)
    missing = [k for k in PROVENANCE_FIELDS if k not in keys]
    if missing:
        problems.append(f"[provenance] {where}: 6-자산 필드 누락 {missing}")
    extra = keys - set(PROVENANCE_FIELDS)
    if extra:
        problems.append(f"[provenance] {where}: 알 수 없는 provenance 필드 {sorted(extra)}")
    for k in PROVENANCE_FIELDS:
        if k in provenance and provenance[k] is None:
            problems.append(f"[provenance] {where}: {k} 값이 None (미상이면 'UNKNOWN' 문자열)")
    return problems


def check_reproducible(manifest: dict) -> list[str]:
    """재현 가능한 실행인지 — git_dirty 는 false 여야 하고 6-자산에 UNKNOWN 이 없어야 한다.
    (최종 실험 게이트용. 개발 실행에서는 경고 참고만.)"""
    problems: list[str] = []
    if manifest.get("git_dirty") is True:
        problems.append("[재현] git_dirty=true — 커밋되지 않은 변경이 있어 재현 불가(규약 §2-4)")
    prov = manifest.get("provenance") or {}
    unknown = [k for k in PROVENANCE_FIELDS if prov.get(k) in (None, "UNKNOWN")]
    if unknown:
        problems.append(f"[재현] provenance 미상 축 {unknown} — 최종 실험은 6-자산이 모두 확정돼야 한다")
    return problems
