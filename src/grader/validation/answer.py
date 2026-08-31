"""answer_type 검증 — task_type ↔ answer_type 허용 조합 + 정답 형태 정합성 (2-17)."""

from __future__ import annotations

from typing import Iterable

from ..models import ALLOWED_ANSWER_TYPES, EvaluationItem, as_id_list


def check_answer_types(items: Iterable[EvaluationItem]) -> list[str]:
    """스키마 v0.2(임현진). 타입은 pydantic 이 이미 검사했다. 여기서는 의미 정합성만:

      - task_type 에 허용되지 않는 answer_type
      - list 인데 정답이 배열이 아님
      - summary 인데 answer_raw 에 체크포인트 배열 없음
      - unanswerable 인데 answer_raw 에 사유 문자열 없음(null 금지)
      - unspecified_type 인데 intermediate_answer 없음(3-2-2 분리 채점 불가)
    """
    problems: list[str] = []
    for it in items:
        allowed = ALLOWED_ANSWER_TYPES.get(it.task_type, set())
        if it.answer_type not in allowed:
            problems.append(
                f"[정합성] {it.id}: task_type={it.task_type} 에 허용되지 않는 "
                f"answer_type={it.answer_type} (허용: {sorted(allowed)})")
        if it.answer_type == "list" and not isinstance(it.answer_normalized or it.answer_raw, list):
            problems.append(f"[정합성] {it.id}: answer_type=list 인데 정답이 배열이 아님(3-4-4 항목 단위 채점 불가)")
        if it.answer_type == "summary" and not it.checkpoints:
            problems.append(f"[정합성] {it.id}: answer_type=summary 인데 answer_raw 에 체크포인트 배열 없음(2-8-2)")
        if it.answer_type == "unanswerable" and not (isinstance(it.answer_raw, str) and it.answer_raw):
            problems.append(f"[정합성] {it.id}: answer_type=unanswerable 인데 answer_raw 에 사유 문자열 없음(2-3, null 금지)")
        if it.unspecified_type is not None and not as_id_list(it.intermediate_answer):
            problems.append(f"[정합성] {it.id}: unspecified_type 인데 intermediate_answer 없음(3-2-2 분리 채점 불가)")
    return problems
