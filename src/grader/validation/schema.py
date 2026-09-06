"""평가셋·응답 JSONL 로딩 + pydantic 스키마 검증."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from checks.check_evalset import load_jsonl  # 로더 단일 출처 (JSONL + concatenated JSON 허용)

from ..models import EvaluationItem, ModelResponse

__all__ = ["load_jsonl", "validate_evaluation_set", "validate_model_responses",
           "index_by_id", "check_response_contract"]


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


def check_response_contract(items, responses, *, require_exact: bool = True) -> dict:
    """평가셋 문항 ↔ 모델 응답의 1:1 대응을 검사한다(4-5 입력 완전성).

    ★조용히 넘어가면 안 되는 이유: 응답이 빠진 문항을 그냥 건너뛰면 채점 분모가
      줄어들어 "답을 안 낸 문항"이 점수에 아무 영향을 못 준다. 응답을 절반만
      제출해도 남은 절반의 정확도가 그대로 최종 점수가 되는 구멍이 생긴다.

    require_exact=False 면 추가 응답(평가셋에 없는 id)은 경고로만 남긴다 —
    CI/개발 모드는 층화 부분집합만 채점하므로 응답이 더 많은 게 정상이다.
    누락과 중복은 어느 모드에서도 실패다.
    """
    item_ids = [it.id for it in items]
    seen: dict[str, int] = {}
    for r in responses:
        seen[r.id] = seen.get(r.id, 0) + 1
    duplicate = sorted(k for k, n in seen.items() if n > 1)
    missing = [i for i in item_ids if i not in seen]
    extra = sorted(set(seen) - set(item_ids))
    problems: list[str] = []
    if missing:
        problems.append(f"응답 누락 {len(missing)}건: {missing[:10]}")
    if duplicate:
        problems.append(f"응답 id 중복 {len(duplicate)}건: {duplicate[:10]}")
    if extra and require_exact:
        problems.append(f"평가셋에 없는 응답 id {len(extra)}건: {extra[:10]}")
    return {
        "ok": not problems,
        "n_items": len(item_ids), "n_responses": len(responses),
        "missing": missing, "duplicate": duplicate, "extra": extra,
        "require_exact": require_exact,
        "problems": problems,
    }
