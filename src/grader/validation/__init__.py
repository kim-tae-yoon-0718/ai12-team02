"""grader.validation — 평가셋·데이터 자체 검증 (모델 성능과 무관, 성능 평가보다 먼저 돈다).

  schema.py      JSONL 로딩 + pydantic 스키마 검증
  answer.py      task_type ↔ answer_type 허용 조합 + 정답 형태 정합성
  location.py    document/section/ref_no 좌표 필드 검증
  provenance.py  6-자산 provenance 기록 완전성 + 재현성
  assets.py      코퍼스 문서 참조 무결성 + 할당량 + CI 1층 데이터 정상성
"""

from __future__ import annotations

from typing import Iterable

from ..models import EvaluationItem
from .answer import check_answer_types
from .assets import (
    check_data_sanity,
    check_data_warnings,
    check_quota,
    check_references,
)
from .leakage import check_leakage
from .location import check_locations
from .provenance import check_provenance, check_reproducible
from .schema import (
    index_by_id,
    load_jsonl,
    validate_evaluation_set,
    validate_model_responses,
)

__all__ = [
    "load_jsonl", "validate_evaluation_set", "validate_model_responses", "index_by_id",
    "check_answer_types", "check_locations", "check_provenance", "check_reproducible",
    "check_references", "check_quota", "check_data_sanity", "check_data_warnings",
    "check_leakage", "check_evalset_integrity",
]


def check_evalset_integrity(
    items: Iterable[EvaluationItem],
    corpus_doc_ids: set[str] | None = None,
    quota: dict[str, int] | None = None,
    quota_tolerance: float = 0.2,
    retrieval_excluded_ids: set[str] | None = None,
    leak_check_root: str | None = None,
    leak_exclude: Iterable[str] = (),
) -> list[str]:
    """CI 2층 — 평가셋이 매번 통과해야 하는 계약 검사(2-17)를 한 번에 돌린다.

    ★ 이 검사들은 모델 성능과 무관하고 값이 싸다. 성능 평가보다 먼저 돌린다.
    leak_check_root: 주면 문항 텍스트가 추적 파일에 유출됐는지도 본다(【25】, 최종셋용).
    """
    items = list(items)
    problems: list[str] = []

    # (1) 중복 id
    seen: dict[str, int] = {}
    for it in items:
        seen[it.id] = seen.get(it.id, 0) + 1
    problems += [f"[중복] id 중복 {k} x{v}" for k, v in seen.items() if v > 1]

    # (2) answer_type 정합성  (3) 참조 무결성 + 중복제외  (4) 할당량  (5) 유출
    problems += check_answer_types(items)
    problems += check_references(items, corpus_doc_ids, retrieval_excluded_ids)
    problems += check_quota(items, quota, quota_tolerance)
    if leak_check_root:
        problems += check_leakage(items, leak_check_root, exclude_paths=leak_exclude)
    return problems
