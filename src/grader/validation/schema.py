"""평가셋·응답 JSONL 로딩 + pydantic 스키마 검증."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from checks.check_evalset import load_jsonl  # 로더 단일 출처 (JSONL + concatenated JSON 허용)

from ..models import EvaluationItem, ModelResponse

__all__ = ["load_jsonl", "validate_evaluation_set", "validate_model_responses", "index_by_id"]


def validate_evaluation_set(path: str | Path, *, strict_meta: bool = False) -> list[EvaluationItem]:
    """평가셋 JSONL 로드 + 스키마 검증.

    strict_meta=True 이면 `_` 로 시작하는 주석 키(practice 세트의 `_source_note` 등)를
    허용하지 않고 실패시킨다 — 최종셋 CI 검사용(임현진 2026-08-30). practice 세트나
    개발 실행에서는 False 로 두어 주석 키를 조용히 무시한다(models 의 before-validator).
    """
    result: list[EvaluationItem] = []
    for idx, record in enumerate(load_jsonl(path), start=1):
        if strict_meta and isinstance(record, dict):
            meta = sorted(k for k in record if str(k).startswith("_"))
            if meta:
                raise ValueError(
                    f"{path} line#{idx}: 최종셋에는 주석 키({meta}) 를 둘 수 없다 "
                    f"(practice 세트 전용). --strict 실행"
                )
        try:
            result.append(EvaluationItem.model_validate(record))
        except ValidationError as exc:
            raise ValueError(f"{path} line#{idx} 평가셋 스키마 오류:\n{exc}") from exc
    return result


def validate_model_responses(path: str | Path) -> list[ModelResponse]:
    result: list[ModelResponse] = []
    for idx, record in enumerate(load_jsonl(path), start=1):
        try:
            result.append(ModelResponse.model_validate(record))
        except ValidationError as exc:
            raise ValueError(f"{path} line#{idx} 모델 응답 스키마 오류:\n{exc}") from exc
    return result


def index_by_id(items: Iterable) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for item in items:
        if item.id in indexed:
            raise ValueError(f"중복 id 발견: {item.id}")
        indexed[item.id] = item
    return indexed
