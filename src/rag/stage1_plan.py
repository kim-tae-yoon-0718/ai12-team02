"""Stage 1 one-shot plan contract.

The language model converts one user question into one bounded action.  This
module validates types and allow-lists only; it never re-interprets the user
question.  Stage 1 intentionally supports a single AND query.  Unsupported
OR/NOT requests must be returned as ``clarify`` instead of being weakened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from table_query import ConditionQuery, OFFICIAL_FIELDS, validate_condition

PLAN_VERSION = "stage1.one_shot.v1"
ACTIONS = (
    "table_select", "table_lookup", "table_compare", "vector_search", "clarify",
)
TASK_TYPES = ("select", "extract", "qa", "compare")
DOCUMENT_SCOPES = ("all_population", "specific_documents", "session_document")
ANSWER_MODES = (
    "document_set", "value", "list", "summary", "comparison", "unanswerable",
    "clarification",
)
DEADLINE_FIELD = "입찰 참여 마감일"
REQUESTABLE_FIELDS = (*OFFICIAL_FIELDS, DEADLINE_FIELD)


class Stage1PlanError(ValueError):
    """The model output cannot be executed without guessing."""


@dataclass
class Stage1Condition:
    field: str
    operator: str
    value: Any = None
    negated: bool = False
    question_span: str = ""

    def to_query(self) -> ConditionQuery:
        query = ConditionQuery(
            field=self.field,
            operator=self.operator,
            value=self.value,
            raw_text=self.question_span,
            from_request=True,
            negated=self.negated,
        )
        validate_condition(query)
        return query


@dataclass
class Stage1Plan:
    action: str
    task_type: str
    document_scope: str
    document_ids: list[str] = field(default_factory=list)
    requested_fields: list[str] = field(default_factory=list)
    condition_logic: str = "AND"
    conditions: list[Stage1Condition] = field(default_factory=list)
    search_query: str = ""
    answer_mode: str = "summary"
    clarification: str | None = None
    unsupported_parts: list[str] = field(default_factory=list)
    decision_note: str = ""
    schema_version: str = PLAN_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_json_schema(valid_document_ids: list[str] | None = None) -> dict[str, Any]:
    """Strict JSON Schema accepted by OpenAI Structured Outputs."""
    condition = {
        "type": "object",
        "additionalProperties": False,
        "required": ["field", "operator", "value", "negated", "question_span"],
        "properties": {
            "field": {"type": "string", "enum": list(OFFICIAL_FIELDS)},
            "operator": {
                "type": "string",
                "enum": [
                    ">=", "<=", ">", "<", "==",
                    "status_value_present", "status_field_absent",
                    "status_external_reference", "status_not_disclosed",
                    "status_conflict", "status_unresolved",
                    "no_restriction", "has_restriction",
                    "consortium_required", "consortium_not_required",
                    "consortium_allowed", "consortium_forbidden",
                ],
            },
            "value": {"type": ["string", "number", "null"]},
            "negated": {"type": "boolean"},
            "question_span": {"type": "string"},
        },
    }
    document_id_item = ({"type": "string", "enum": list(valid_document_ids)}
                        if valid_document_ids else
                        {"type": "string", "pattern": "^RFP-[0-9]{6}$"})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action", "task_type", "document_scope", "document_ids",
            "requested_fields", "condition_logic", "conditions", "search_query",
            "answer_mode", "clarification", "unsupported_parts", "decision_note",
        ],
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "task_type": {"type": "string", "enum": list(TASK_TYPES)},
            "document_scope": {"type": "string", "enum": list(DOCUMENT_SCOPES)},
            "document_ids": {
                "type": "array",
                "items": document_id_item,
                "description": "table_select는 반드시 빈 배열. 특정 문서 행동만 ID를 넣는다.",
            },
            "requested_fields": {
                "type": "array", "items": {"type": "string", "enum": list(REQUESTABLE_FIELDS)},
                "description": "table_lookup/table_compare만 채운다. 나머지 행동은 빈 배열.",
            },
            "condition_logic": {"type": "string", "enum": ["AND"]},
            "conditions": {"type": "array", "items": condition},
            "search_query": {
                "type": "string",
                "description": "vector_search만 채운다. 나머지 행동은 빈 문자열.",
            },
            "answer_mode": {"type": "string", "enum": list(ANSWER_MODES)},
            "clarification": {"type": ["string", "null"]},
            "unsupported_parts": {"type": "array", "items": {"type": "string"}},
            "decision_note": {"type": "string"},
        },
    }


def parse_plan(raw: Any) -> Stage1Plan:
    if not isinstance(raw, dict):
        raise Stage1PlanError("계획이 JSON object가 아닙니다.")
    if raw.get("action") not in ACTIONS:
        raise Stage1PlanError(f"허용되지 않은 action: {raw.get('action')!r}")
    if raw.get("task_type") not in TASK_TYPES:
        raise Stage1PlanError(f"허용되지 않은 task_type: {raw.get('task_type')!r}")
    if raw.get("document_scope") not in DOCUMENT_SCOPES:
        raise Stage1PlanError("허용되지 않은 document_scope입니다.")
    if raw.get("answer_mode") not in ANSWER_MODES:
        raise Stage1PlanError("허용되지 않은 answer_mode입니다.")
    if raw.get("condition_logic") != "AND":
        raise Stage1PlanError("Stage 1은 AND만 실행합니다. OR/NOT은 되물어야 합니다.")

    def string_list(name: str) -> list[str]:
        value = raw.get(name)
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise Stage1PlanError(f"{name}은 문자열 배열이어야 합니다.")
        return [v.strip() for v in value if v.strip()]

    conditions_raw = raw.get("conditions")
    if not isinstance(conditions_raw, list):
        raise Stage1PlanError("conditions는 배열이어야 합니다.")
    conditions: list[Stage1Condition] = []
    for item in conditions_raw:
        if not isinstance(item, dict):
            raise Stage1PlanError("condition 항목이 object가 아닙니다.")
        cond = Stage1Condition(
            field=str(item.get("field") or "").strip(),
            operator=str(item.get("operator") or "").strip(),
            value=item.get("value"),
            negated=bool(item.get("negated")),
            question_span=str(item.get("question_span") or "").strip(),
        )
        try:
            cond.to_query()
        except Exception as exc:  # table_query owns the operator/field contract
            raise Stage1PlanError(str(exc)) from exc
        conditions.append(cond)

    requested_fields = string_list("requested_fields")
    unknown_fields = sorted(set(requested_fields) - set(REQUESTABLE_FIELDS))
    if unknown_fields:
        raise Stage1PlanError(f"허용되지 않은 필드: {unknown_fields}")

    return Stage1Plan(
        action=raw["action"],
        task_type=raw["task_type"],
        document_scope=raw["document_scope"],
        document_ids=string_list("document_ids"),
        requested_fields=requested_fields,
        condition_logic="AND",
        conditions=conditions,
        search_query=str(raw.get("search_query") or "").strip(),
        answer_mode=raw["answer_mode"],
        clarification=(str(raw["clarification"]).strip()
                       if raw.get("clarification") is not None else None),
        unsupported_parts=string_list("unsupported_parts"),
        decision_note=str(raw.get("decision_note") or "").strip(),
    )


def validate_plan(
    plan: Stage1Plan,
    valid_document_ids: set[str],
    session_document_id: str | None = None,
) -> Stage1Plan:
    """Validate only the structured contract. Never inspect the question text."""
    if plan.document_scope == "session_document" and not plan.document_ids:
        if not session_document_id:
            raise Stage1PlanError("세션 문서가 지정되지 않았습니다.")
        plan.document_ids = [session_document_id]

    unknown = sorted(set(plan.document_ids) - valid_document_ids)
    if unknown:
        raise Stage1PlanError(f"공식 목록에 없는 문서 ID: {unknown}")
    if len(plan.document_ids) != len(set(plan.document_ids)):
        raise Stage1PlanError("document_ids에 중복이 있습니다.")
    unknown_fields = sorted(set(plan.requested_fields) - set(REQUESTABLE_FIELDS))
    if unknown_fields:
        raise Stage1PlanError(f"허용되지 않은 필드: {unknown_fields}")
    if plan.unsupported_parts and plan.action != "clarify":
        raise Stage1PlanError("해석하지 못한 조건을 남긴 채 일부만 실행할 수 없습니다.")

    if plan.action == "table_select":
        if plan.task_type != "select" or plan.document_scope != "all_population":
            raise Stage1PlanError("table_select는 전체 문서 선별 계획이어야 합니다.")
        # Some models repeat the condition fields in requested_fields.  It is
        # redundant but not a lost request: the selection answer already emits
        # every condition field.  New fields outside the conditions still fail.
        condition_fields = {condition.field for condition in plan.conditions}
        redundant_fields_only = set(plan.requested_fields) <= condition_fields
        if not plan.conditions or plan.document_ids or not redundant_fields_only:
            raise Stage1PlanError("table_select의 조건·범위 형식이 올바르지 않습니다.")
    elif plan.action == "table_lookup":
        if len(plan.document_ids) != 1 or not plan.requested_fields or plan.conditions:
            raise Stage1PlanError("table_lookup은 문서 1건과 조회 필드가 필요합니다.")
    elif plan.action == "table_compare":
        if len(plan.document_ids) < 2 or not plan.requested_fields or plan.conditions:
            raise Stage1PlanError("table_compare는 문서 2건 이상과 비교 필드가 필요합니다.")
    elif plan.action == "vector_search":
        if len(plan.document_ids) > 1 or not plan.search_query or plan.conditions:
            raise Stage1PlanError("vector_search의 문서 범위나 검색 질문이 올바르지 않습니다.")
        if plan.document_scope in {"specific_documents", "session_document"} and not plan.document_ids:
            raise Stage1PlanError("특정 문서 검색에는 확정 문서 ID가 필요합니다.")
    elif plan.action == "clarify":
        if not plan.clarification:
            raise Stage1PlanError("clarify 계획에는 사용자에게 보낼 질문이 필요합니다.")
    return plan
