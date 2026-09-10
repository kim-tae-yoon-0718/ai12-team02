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
    check_response_contract,
)

__all__ = [
    "load_jsonl", "validate_evaluation_set", "validate_model_responses",
    "check_response_contract", "index_by_id",
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
    retrieval_excluded_ids: set[str] | None = None,
    leak_check_root: str | None = None,
    leak_exclude: Iterable[str] = (),
    final_set: bool = False,
) -> list[str]:
    """CI 2층 어댑터 — 실제 규칙은 checks.check_evalset 단일 출처.

    runner 2층·cli validate 는 run_all() 을 직접 부른다. 이 함수는 EvaluationItem 을
    받는 in-memory 호출(테스트 등)용 — corpus_doc_ids 는 set 으로 받는다.
    """
    from checks.check_evalset import (
        _locations_of, check_dup_ids, check_excluded_as_gold, check_leak,
        check_quota, check_schema, scan_tracked_files,
    )

    records = [_as_record(it) for it in items]
    problems: list[str] = []
    for i, r in enumerate(records, start=1):
        problems += check_schema(r, i)
    problems += check_dup_ids(records)
    problems += check_quota(records)
    problems += check_excluded_as_gold(records, retrieval_excluded_ids)
    if corpus_doc_ids is not None:
        for r in records:
            for loc in _locations_of(r):
                d = loc.get("document")
                if d and d not in corpus_doc_ids:
                    problems.append(f"C4: unknown document: {d} (item {r.get('id')})")
    if final_set:
        problems += check_leak(records, None)
    if leak_check_root:
        problems += scan_tracked_files(records, leak_check_root, exclude_paths=list(leak_exclude))
    return problems
