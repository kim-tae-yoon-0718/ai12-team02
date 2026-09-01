"""grader.validation — grader 가 책임지는 검증만 남긴다.

  schema.py      JSONL 로딩 + pydantic EvaluationItem/ModelResponse 파싱 (채점 입력 계약)
  provenance.py  6-자산 provenance 기록 완전성 + 재현성 (채점 결과 검증)
  assets.py      CI 1층 데이터 정상성 (박예진 산출물 — data_manifest)

★ 평가셋 계약 검사(스키마·조합·중복·할당량·참조무결성·좌표·유출, 2-17)는
  `checks.check_evalset` 로 단일화됐다(임현진 소관). runner 2층은 거기를 호출한다.
  아래 check_evalset_integrity() 는 그 얇은 어댑터다(EvaluationItem/ dict 둘 다 받음).
"""

from __future__ import annotations

from typing import Iterable

from ..models import EvaluationItem
from .assets import check_data_sanity, check_data_warnings
from .provenance import check_provenance, check_reproducible
from .schema import (
    index_by_id,
    load_jsonl,
    validate_evaluation_set,
    validate_model_responses,
)

__all__ = [
    "load_jsonl", "validate_evaluation_set", "validate_model_responses", "index_by_id",
    "check_provenance", "check_reproducible",
    "check_data_sanity", "check_data_warnings",
    "check_evalset_integrity",
]

_COMPUTED = {"document_unspecified", "time_dependent", "conversational"}


def _as_record(item: EvaluationItem | dict) -> dict:
    if isinstance(item, EvaluationItem):
        return item.model_dump(mode="json", exclude_none=True, exclude=_COMPUTED)
    return item


def check_evalset_integrity(
    items: Iterable[EvaluationItem | dict],
    corpus_doc_ids: set[str] | None = None,
    quota: dict[str, int] | None = None,
    quota_tolerance: float = 0.2,  # (미사용 — checks.check_evalset 는 정확 대조/경고만)
    retrieval_excluded_ids: set[str] | None = None,
    leak_check_root: str | None = None,
    leak_exclude: Iterable[str] = (),
) -> list[str]:
    """CI 2층 — 평가셋 계약 검사(2-17). 실제 규칙은 checks.check_evalset 단일 출처.

    ★ 값이 싸고 모델 성능과 무관하다. 성능 평가보다 먼저 돌린다.
    """
    from checks.check_evalset import run_all

    records = [_as_record(it) for it in items]
    return run_all(
        records,
        corpus_doc_ids=corpus_doc_ids,
        excluded_doc_ids=retrieval_excluded_ids,
        quota=quota,
        leak_repo_root=leak_check_root,
        leak_exclude=list(leak_exclude),
    )
