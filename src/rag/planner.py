"""Optional Stage1 question planner.

The planner is deliberately opt-in. Invalid or incomplete plans are rejected so
that the baseline rule-based path remains the fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_DOC_ID_RE = re.compile(r"^RFP-\d{6}$")
_ALLOWED_TASKS = {"select", "extract", "compare", "qa", "no_search_needed"}
_ALLOWED_ANSWER_MODES = {"value", "list", "explanation", "comparison", "clarify"}
_ALLOWED_FIELDS = {
    "사업 개요", "사업분야", "공고일", "사업기간", "예산",
    "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건", "평가 배점",
    "제출 방식", "필수 제출 서류", "과업 범위",
}
_ALLOWED_OPERATORS = {
    ">=", "<=", ">", "<", "==", "status_value_present", "status_field_absent",
    "status_external_reference", "status_not_disclosed", "status_conflict",
    "status_unresolved", "no_restriction", "has_restriction", "consortium_required",
    "consortium_not_required", "consortium_allowed", "consortium_forbidden",
}


@dataclass(frozen=True)
class ExecutionPlan:
    task_type: str
    document_ids: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    conditions: tuple[dict[str, Any], ...] = ()
    answer_mode: str = "explanation"


def _as_string_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"plan.{name} must be a string list")
    return tuple(x.strip() for x in value if x.strip())


def parse_plan(raw: str) -> ExecutionPlan:
    """Parse and validate a planner JSON response; never accept partial JSON."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("planner output must be an object")

    task_type = data.get("task_type")
    if task_type not in _ALLOWED_TASKS:
        raise ValueError(f"unsupported task_type: {task_type!r}")
    answer_mode = data.get("answer_mode", "explanation")
    if answer_mode not in _ALLOWED_ANSWER_MODES:
        raise ValueError(f"unsupported answer_mode: {answer_mode!r}")

    document_ids = _as_string_list(data.get("document_ids"), "document_ids")
    if any(not _DOC_ID_RE.fullmatch(doc_id) for doc_id in document_ids):
        raise ValueError("document_ids contains an invalid RFP id")
    fields = _as_string_list(data.get("fields"), "fields")
    if any(field not in _ALLOWED_FIELDS for field in fields):
        invalid = [field for field in fields if field not in _ALLOWED_FIELDS]
        raise ValueError(f"unsupported plan field: {invalid!r}")
    conditions = data.get("conditions") or []
    if not isinstance(conditions, list) or not all(isinstance(c, dict) for c in conditions):
        raise ValueError("plan.conditions must be an object list")
    for condition in conditions:
        if not isinstance(condition.get("field"), str) or not isinstance(condition.get("operator"), str):
            raise ValueError("each condition needs field and operator")
        if condition["field"] not in _ALLOWED_FIELDS:
            raise ValueError(f"unsupported condition field: {condition['field']!r}")
        if condition["operator"] not in _ALLOWED_OPERATORS:
            raise ValueError(f"unsupported condition operator: {condition['operator']!r}")
        if "value" in condition and not isinstance(condition["value"], (str, int, float, type(None))):
            raise ValueError("condition.value has an unsupported type")
        if "negated" in condition and not isinstance(condition["negated"], bool):
            raise ValueError("condition.negated must be boolean")

    if task_type == "select" and not conditions:
        raise ValueError("select plan needs conditions")
    if task_type in {"extract", "compare"} and not fields:
        raise ValueError(f"{task_type} plan needs fields")
    if task_type == "compare" and len(document_ids) < 2:
        raise ValueError("compare plan needs two document ids")

    return ExecutionPlan(task_type, document_ids, fields,
                         tuple(dict(c) for c in conditions), answer_mode)
