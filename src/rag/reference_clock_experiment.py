"""실사용 대비 실험 모듈 — reference_datetime을 실제 현재 시각(now)으로 얻는 경로.

⚠️ 공식 경로(identity_metadata.reference_datetime_from_config, 4-10-2 확정/1-9④)는
   건드리지 않는다. 그 함수는 여전히 "external"/None만 받고 그 외는 거부한다 —
   평가셋은 지금과 똑같이 재현 가능한 고정 시각만 쓴다.

이 모듈은 그 함수와 별개의 실험 진입점이다. 공식 설정 키(reference_datetime_source)를
공유하지 않고 독자적인 experimental_reference_datetime_source를 본다 — 그래야 이 모듈이
실수로라도 평가 경로에 섞여 들어가지 않는다. 지금은 어떤 공식 스크립트에서도
자동으로 불려지지 않으며, 나중에 실사용(라이브) 진입점이 생기면 거기서 명시적으로
가져다 쓴다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from identity_metadata import reference_datetime_from_config


def now_reference_datetime() -> datetime:
    """실제 현재 시각. 평가셋 재현성이 필요 없는 실사용 경로에서만 쓴다."""
    return datetime.now()


def resolve_reference_datetime(cfg: dict[str, Any]) -> datetime:
    """실험용 reference_datetime 결정.

    experimental_reference_datetime_source가 "now"면 실제 현재 시각을 쓰고,
    그 외에는 공식 reference_datetime_from_config에 그대로 위임한다(기존 계약 그대로
    "external"/None만 허용, 나머지는 거기서 에러).
    """
    if cfg.get("experimental_reference_datetime_source") == "now":
        return now_reference_datetime()
    return reference_datetime_from_config(cfg)
