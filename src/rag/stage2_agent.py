"""Bounded StructGPT-style controller for the Stage 2 experiment.

The model may inspect tool observations and choose another tool, but it never
returns executable code.  This module owns the strict response schema and the
closed action contract; the pipeline owns actual tool execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from generation_client import model_rejects_sampling_params
from pricing import Usage
from reflection_memory import ReflectionMemory, ReflectionMemoryError
from stage1_plan import (
    ANSWER_MODES,
    DOCUMENT_SCOPES,
    TASK_TYPES,
    Stage1Plan,
    Stage1PlanError,
    parse_plan,
    plan_json_schema,
    validate_plan,
)
from stage1_planner import document_catalog
from table_query import NUMERIC_OPERATORS

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


STAGE2_VERSION = "stage2.dual_interpretation.v1"
RESULT_ASSESSMENTS = ("initial", "sufficient", "insufficient", "plan_wrong")
VERIFICATION_OUTCOMES = ("approve", "correct", "clarify")
ADJUDICATION_CHOICES = ("candidate_a", "candidate_b", "clarify")
TOOL_ACTIONS = (
    "table_select", "table_lookup", "table_compare", "vector_search",
)
DECISION_ACTIONS = (*TOOL_ACTIONS, "finalize", "clarify")


class Stage2AgentError(RuntimeError):
    pass


class Stage2ResponseTruncated(Stage2AgentError):
    """The provider stopped before a complete strict response was produced."""


@dataclass
class Stage2Decision:
    action: str
    task_type: str
    document_scope: str
    result_assessment: str = "initial"
    document_ids: list[str] = field(default_factory=list)
    requested_fields: list[str] = field(default_factory=list)
    condition_logic: str = "AND"
    conditions: list[dict[str, Any]] = field(default_factory=list)
    search_query: str = ""
    answer_mode: str = "summary"
    clarification: str | None = None
    unsupported_parts: list[str] = field(default_factory=list)
    decision_note: str = ""
    final_answer: str = ""
    final_items: list[str] = field(default_factory=list)
    used_observation_ids: list[str] = field(default_factory=list)
    used_evidence_ids: list[str] = field(default_factory=list)
    tool_plan: Stage1Plan | None = None
    schema_version: str = STAGE2_VERSION

    def as_dict(self) -> dict[str, Any]:
        out = {
            "action": self.action,
            "task_type": self.task_type,
            "document_scope": self.document_scope,
            "result_assessment": self.result_assessment,
            "document_ids": list(self.document_ids),
            "requested_fields": list(self.requested_fields),
            "condition_logic": self.condition_logic,
            "conditions": list(self.conditions),
            "search_query": self.search_query,
            "answer_mode": self.answer_mode,
            "clarification": self.clarification,
            "unsupported_parts": list(self.unsupported_parts),
            "decision_note": self.decision_note,
            "final_answer": self.final_answer,
            "final_items": list(self.final_items),
            "used_observation_ids": list(self.used_observation_ids),
            "used_evidence_ids": list(self.used_evidence_ids),
            "schema_version": self.schema_version,
        }
        return out


@dataclass
class Stage2PlanVerification:
    outcome: str
    verified_plan: Stage2Decision
    checked_requirements: list[str] = field(default_factory=list)
    issue_types: list[str] = field(default_factory=list)
    verification_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "checked_requirements": list(self.checked_requirements),
            "issue_types": list(self.issue_types),
            "verification_note": self.verification_note,
            "verified_plan": self.verified_plan.as_dict(),
        }


@dataclass
class Stage2Adjudication:
    """Choice between two independently produced, already validated plans."""

    selected_candidate: str
    checked_requirements: list[str] = field(default_factory=list)
    decision_note: str = ""
    clarification: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate,
            "checked_requirements": list(self.checked_requirements),
            "decision_note": self.decision_note,
            "clarification": self.clarification,
        }


def decision_json_schema(
    valid_document_ids: list[str],
    observation_ids: list[str],
    evidence_ids: list[str],
    locked_task_type: str | None = None,
    locked_answer_mode: str | None = None,
    allowed_actions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    schema = plan_json_schema(valid_document_ids)
    action_values = allowed_actions or DECISION_ACTIONS
    unknown_actions = sorted(set(action_values) - set(DECISION_ACTIONS))
    if unknown_actions:
        raise Stage2AgentError(f"허용할 수 없는 action 목록: {unknown_actions}")
    schema["properties"]["action"]["enum"] = list(action_values)
    if locked_task_type is not None:
        if locked_task_type not in TASK_TYPES:
            raise Stage2AgentError(f"고정할 수 없는 task_type: {locked_task_type!r}")
        schema["properties"]["task_type"]["enum"] = [locked_task_type]
    if locked_answer_mode is not None:
        if locked_answer_mode not in ANSWER_MODES:
            raise Stage2AgentError(f"고정할 수 없는 answer_mode: {locked_answer_mode!r}")
        modes = [locked_answer_mode]
        if locked_answer_mode != "unanswerable":
            modes.append("unanswerable")
        schema["properties"]["answer_mode"]["enum"] = modes
    schema["properties"].update({
        "result_assessment": {
            "type": "string",
            "enum": ([value for value in RESULT_ASSESSMENTS if value != "initial"]
                     if observation_ids else ["initial"]),
            "description": (
                "도구 결과가 없으면 initial. 결과가 질문에 맞고 충분하면 sufficient, "
                "맞지만 부족하면 insufficient, 최초 질문 해석이나 도구 선택이 틀렸으면 plan_wrong."
            ),
        },
        "final_answer": {"type": "string"},
        "final_items": {
            "type": "array",
            "items": {
                "type": "string",
                "description": (
                    "하나의 독립된 조건·배점·서류·과업만 담는 원자적 항목. "
                    "여러 사실이나 여러 줄 목록을 한 문자열에 합치지 않는다."
                ),
            },
        },
        "used_observation_ids": {
            "type": "array",
            "items": ({"type": "string", "enum": observation_ids}
                      if observation_ids else {"type": "string"}),
            **({} if observation_ids else {"maxItems": 0}),
        },
        "used_evidence_ids": {
            "type": "array",
            "items": ({"type": "string", "enum": evidence_ids}
                      if evidence_ids else {"type": "string"}),
            **({} if evidence_ids else {"maxItems": 0}),
        },
    })
    schema["required"].extend([
        "result_assessment", "final_answer", "final_items",
        "used_observation_ids", "used_evidence_ids",
    ])
    return schema


def verification_json_schema(valid_document_ids: list[str]) -> dict[str, Any]:
    """Strict schema for the independent, pre-execution plan audit."""
    plan_schema = decision_json_schema(
        valid_document_ids,
        observation_ids=[],
        evidence_ids=[],
        allowed_actions=DECISION_ACTIONS,
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "outcome", "checked_requirements", "issue_types",
            "verification_note", "verified_plan",
        ],
        "properties": {
            "outcome": {"type": "string", "enum": list(VERIFICATION_OUTCOMES)},
            "checked_requirements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "issue_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "document_scope", "document_identity", "task_type",
                        "tool_choice", "field_mapping", "condition_semantics",
                        "answer_shape", "unsupported_request", "none",
                    ],
                },
            },
            "verification_note": {"type": "string"},
            "verified_plan": plan_schema,
        },
    }


def _decision_signature(decision: Stage2Decision) -> dict[str, Any]:
    """Return a canonical, language-blind execution signature.

    The comparison deliberately ignores notes and question spans.  It sorts
    set-like fields and conditions, but never tries to interpret Korean text.
    """
    conditions = []
    for item in decision.conditions:
        conditions.append({
            "field": item.get("field"),
            "operator": item.get("operator"),
            "value": item.get("value"),
            "negated": bool(item.get("negated")),
        })
    conditions.sort(key=lambda item: json.dumps(
        item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    requested_fields = ([] if decision.action == "table_select"
                        else sorted(decision.requested_fields))
    return {
        "action": decision.action,
        "task_type": decision.task_type,
        "document_scope": decision.document_scope,
        "document_ids": sorted(decision.document_ids),
        "requested_fields": requested_fields,
        "condition_logic": decision.condition_logic,
        "conditions": conditions,
        "search_query": decision.search_query,
        "answer_mode": decision.answer_mode,
        "clarification": decision.clarification,
        "unsupported_parts": sorted(decision.unsupported_parts),
    }


def decisions_structurally_equal(
    candidate_a: Stage2Decision, candidate_b: Stage2Decision,
) -> bool:
    """Compare only validated structured fields, never the user sentence."""
    return _decision_signature(candidate_a) == _decision_signature(candidate_b)


def adjudication_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "selected_candidate", "checked_requirements", "decision_note",
            "clarification",
        ],
        "properties": {
            "selected_candidate": {
                "type": "string", "enum": list(ADJUDICATION_CHOICES),
            },
            "checked_requirements": {
                "type": "array", "items": {"type": "string"},
            },
            "decision_note": {"type": "string"},
            "clarification": {"type": ["string", "null"]},
        },
    }


def parse_adjudication(raw: Any) -> Stage2Adjudication:
    if not isinstance(raw, dict):
        raise Stage2AgentError("계획 선택 결과가 JSON object가 아닙니다.")
    selected = raw.get("selected_candidate")
    if selected not in ADJUDICATION_CHOICES:
        raise Stage2AgentError("허용되지 않은 계획 선택 결과입니다.")
    checked = _string_list(raw, "checked_requirements")
    note = str(raw.get("decision_note") or "").strip()
    clarification = (str(raw["clarification"]).strip()
                     if raw.get("clarification") is not None else None)
    if selected == "clarify" and not clarification:
        raise Stage2AgentError("계획을 고를 수 없으면 구체적인 되묻기가 필요합니다.")
    # The adjudicator sometimes copies the clarification carried by a selected
    # terminal candidate.  That field is audit-only when A/B is selected; the
    # pipeline executes the selected candidate itself and never substitutes
    # this text.  Accepting it therefore changes no semantics.
    return Stage2Adjudication(
        selected_candidate=selected,
        checked_requirements=checked,
        decision_note=note,
        clarification=clarification,
    )


def parse_verification(
    raw: Any,
    proposed_plan: Stage2Decision,
    valid_document_ids: set[str],
    session_document_id: str | None = None,
) -> Stage2PlanVerification:
    if not isinstance(raw, dict):
        raise Stage2AgentError("계획 검토 결과가 JSON object가 아닙니다.")
    outcome = raw.get("outcome")
    if outcome not in VERIFICATION_OUTCOMES:
        raise Stage2AgentError("허용되지 않은 계획 검토 결과입니다.")
    checked = _string_list(raw, "checked_requirements")
    issues = _string_list(raw, "issue_types")
    allowed_issues = {
        "document_scope", "document_identity", "task_type", "tool_choice",
        "field_mapping", "condition_semantics", "answer_shape",
        "unsupported_request", "none",
    }
    if not set(issues) <= allowed_issues:
        raise Stage2AgentError("계획 검토의 issue_types가 허용 목록 밖입니다.")
    if "none" in issues and len(issues) != 1:
        raise Stage2AgentError("issue_types의 none은 다른 문제와 함께 쓸 수 없습니다.")
    verified = parse_decision(
        raw.get("verified_plan"),
        valid_document_ids=valid_document_ids,
        observation_ids=set(),
        evidence_ids=set(),
        session_document_id=session_document_id,
    )
    proposed_signature = _decision_signature(proposed_plan)
    verified_signature = _decision_signature(verified)
    if outcome == "approve":
        if issues != ["none"]:
            raise Stage2AgentError("승인된 계획의 issue_types는 none이어야 합니다.")
        if verified_signature != proposed_signature:
            raise Stage2AgentError("approve는 최초 계획을 바꿀 수 없습니다.")
    elif outcome == "correct":
        if not issues or issues == ["none"]:
            raise Stage2AgentError("수정된 계획에는 실제 문제 유형이 필요합니다.")
        if verified_signature == proposed_signature:
            raise Stage2AgentError("correct는 최초 계획을 실제로 수정해야 합니다.")
        if verified.action == "clarify":
            raise Stage2AgentError("되묻기는 outcome=clarify로 표시해야 합니다.")
    else:
        if verified.action != "clarify":
            raise Stage2AgentError("outcome=clarify의 verified_plan은 clarify여야 합니다.")
        if not issues or issues == ["none"]:
            raise Stage2AgentError("되묻기에는 확인된 문제 유형이 필요합니다.")
    return Stage2PlanVerification(
        outcome=outcome,
        verified_plan=verified,
        checked_requirements=checked,
        issue_types=issues,
        verification_note=str(raw.get("verification_note") or "").strip(),
    )


def _string_list(raw: dict[str, Any], name: str) -> list[str]:
    value = raw.get(name)
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise Stage2AgentError(f"{name}은 문자열 배열이어야 합니다.")
    stripped = [v.strip() for v in value if v.strip()]
    if len(stripped) != len(set(stripped)):
        raise Stage2AgentError(f"{name}에 중복이 있습니다.")
    return stripped


def _deduplicated_string_list(raw: dict[str, Any], name: str) -> list[str]:
    """Canonicalize a set-like string slot without interpreting its text."""
    value = raw.get(name)
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise Stage2AgentError(f"{name}은 문자열 배열이어야 합니다.")
    return list(dict.fromkeys(v.strip() for v in value if v.strip()))


def parse_decision(
    raw: Any,
    valid_document_ids: set[str],
    observation_ids: set[str],
    evidence_ids: set[str],
    session_document_id: str | None = None,
) -> Stage2Decision:
    if not isinstance(raw, dict):
        raise Stage2AgentError("결정이 JSON object가 아닙니다.")
    action = raw.get("action")
    if action not in DECISION_ACTIONS:
        raise Stage2AgentError(f"허용되지 않은 action: {action!r}")
    if raw.get("task_type") not in TASK_TYPES:
        raise Stage2AgentError("허용되지 않은 task_type입니다.")
    if raw.get("document_scope") not in DOCUMENT_SCOPES:
        raise Stage2AgentError("허용되지 않은 document_scope입니다.")
    if raw.get("answer_mode") not in ANSWER_MODES:
        raise Stage2AgentError("허용되지 않은 answer_mode입니다.")
    assessment = raw.get("result_assessment")
    if assessment not in RESULT_ASSESSMENTS:
        raise Stage2AgentError("허용되지 않은 result_assessment입니다.")
    has_observations = bool(observation_ids)
    if has_observations and assessment == "initial":
        raise Stage2AgentError("도구 결과를 받은 뒤에는 결과를 평가해야 합니다.")
    if not has_observations and assessment != "initial":
        raise Stage2AgentError("도구 결과가 없을 때는 result_assessment=initial이어야 합니다.")
    if has_observations and action in TOOL_ACTIONS and assessment not in {
        "sufficient", "insufficient", "plan_wrong",
    }:
        raise Stage2AgentError("추가 행동의 결과 판정이 올바르지 않습니다.")
    if (has_observations and action == "finalize"
            and assessment == "insufficient"
            and raw.get("answer_mode") != "unanswerable"):
        raise Stage2AgentError("정보가 부족하면 다른 도구를 쓰거나 기권해야 합니다.")
    if has_observations and action == "clarify" and assessment == "sufficient":
        raise Stage2AgentError("충분한 결과라고 판정하면서 되물을 수 없습니다.")

    used_observations = _string_list(raw, "used_observation_ids")
    used_evidence = _string_list(raw, "used_evidence_ids")
    unknown_observations = sorted(set(used_observations) - observation_ids)
    unknown_evidence = sorted(set(used_evidence) - evidence_ids)
    if unknown_observations:
        raise Stage2AgentError(f"존재하지 않는 도구 결과: {unknown_observations}")
    if unknown_evidence:
        raise Stage2AgentError(f"존재하지 않는 근거: {unknown_evidence}")

    decision = Stage2Decision(
        action=action,
        task_type=raw["task_type"],
        document_scope=raw["document_scope"],
        result_assessment=assessment,
        document_ids=_string_list(raw, "document_ids"),
        requested_fields=_deduplicated_string_list(raw, "requested_fields"),
        condition_logic=str(raw.get("condition_logic") or ""),
        conditions=list(raw.get("conditions") or []),
        search_query=str(raw.get("search_query") or "").strip(),
        answer_mode=raw["answer_mode"],
        clarification=(str(raw["clarification"]).strip()
                       if raw.get("clarification") is not None else None),
        unsupported_parts=_string_list(raw, "unsupported_parts"),
        decision_note=str(raw.get("decision_note") or "").strip(),
        final_answer=str(raw.get("final_answer") or "").strip(),
        final_items=_string_list(raw, "final_items"),
        used_observation_ids=used_observations,
        used_evidence_ids=used_evidence,
    )

    if action in TOOL_ACTIONS:
        # The tool input is validated independently below.  Strict JSON makes
        # all final-output fields present on every turn, and a model may fill
        # them early.  They are retained for audit but never executed or used
        # as evidence until a later explicit ``finalize`` action.
        try:
            tool_raw = {key: raw[key] for key in (
                "action", "task_type", "document_scope", "document_ids",
                "requested_fields", "condition_logic", "conditions", "search_query",
                "answer_mode", "clarification", "unsupported_parts", "decision_note",
            )}
            tool_raw["requested_fields"] = list(dict.fromkeys(
                str(value).strip() for value in tool_raw["requested_fields"]
                if str(value).strip()))
            for item in tool_raw["conditions"]:
                if not isinstance(item, dict):
                    continue
                if (item.get("operator") not in NUMERIC_OPERATORS
                        and item.get("value") not in (None, "")):
                    raise Stage2AgentError(
                        "상태·의미 연산자는 값을 검색하지 않습니다. value는 null이어야 합니다.")
            if action == "table_select":
                # Strict outputs use one broad object for every action. Models
                # can place display columns or prior result IDs in slots that
                # table_select never executes. Keep the raw decision for audit,
                # but derive the executable input only from its conditions.
                condition_fields = [
                    str(item.get("field") or "").strip()
                    for item in tool_raw["conditions"] if isinstance(item, dict)
                ]
                tool_raw["document_ids"] = []
                tool_raw["requested_fields"] = list(dict.fromkeys(condition_fields))
                tool_raw["search_query"] = ""
            plan = parse_plan(tool_raw)
            decision.tool_plan = validate_plan(
                plan, valid_document_ids, session_document_id=session_document_id)
        except (KeyError, Stage1PlanError) as exc:
            raise Stage2AgentError(str(exc)) from exc
        return decision

    # Strict Structured Outputs requires every property on every turn.  Models
    # sometimes repeat the last tool's harmless inputs when finalizing.  Those
    # values are never executed on a terminal action, so accepting and ignoring
    # them is safer than discarding an otherwise grounded answer.  We still
    # retain them in the trace for audit; unsupported_parts remains a hard stop.
    if decision.unsupported_parts and action != "clarify":
        raise Stage2AgentError("해석하지 못한 부분이 있으면 되물어야 합니다.")

    if action == "clarify":
        if not decision.clarification:
            raise Stage2AgentError("clarify에는 질문이 필요합니다.")
        # Other strict-schema slots are inert on clarify. They remain in the
        # trace but are never exposed as an answer or citation.
        return decision

    if not used_observations and decision.answer_mode != "unanswerable":
        raise Stage2AgentError("finalize에는 사용한 도구 결과가 필요합니다.")
    evidence_observations = {value.split(":E", 1)[0] for value in used_evidence}
    if not evidence_observations <= set(used_observations):
        raise Stage2AgentError("사용하지 않은 도구 결과의 근거를 인용했습니다.")
    if decision.answer_mode == "list":
        if not decision.final_items:
            if decision.final_answer:
                # Both slots are already strict strings supplied by the model.
                # Moving one complete answer into a one-item list is structural
                # canonicalization; no question text is inspected here.
                decision.final_items = [decision.final_answer]
            else:
                raise Stage2AgentError("목록 답변은 final_items가 필요합니다.")
    elif decision.answer_mode in {"value", "summary"}:
        if not decision.final_answer:
            raise Stage2AgentError("값·요약 답변은 final_answer가 필요합니다.")
    elif decision.answer_mode == "unanswerable":
        if not decision.final_answer:
            # This is a fixed transport fallback, not an interpretation of the
            # question.  The abstention state is already explicit in the plan.
            decision.final_answer = "제공된 근거로 답을 확정할 수 없습니다."
    elif decision.answer_mode in {"document_set", "comparison"}:
        # The deterministic structured tool result is returned unchanged.
        # Any repeated text in the strict output slots is ignored.
        pass
    else:
        raise Stage2AgentError(f"finalize에서 쓸 수 없는 answer_mode: {decision.answer_mode}")
    if decision.answer_mode not in {"unanswerable", "document_set"} and not used_evidence:
        raise Stage2AgentError("근거가 있는 최종 답변에는 used_evidence_ids가 필요합니다.")
    return decision


def _prompt_path(name: str) -> Path:
    here = Path(__file__).resolve().parent
    for root in [here, *here.parents]:
        candidate = root / "prompts" / name
        if candidate.exists():
            return candidate
    raise Stage2AgentError(f"Stage 2 프롬프트 파일을 찾지 못했습니다: {name}")


class Stage2Agent:
    def __init__(self, cfg: dict[str, Any], identity, eligible_ids: set[str] | None):
        if cfg.get("generation_provider") != "openai":
            raise Stage2AgentError("Stage 2 제어기는 OpenAI 경로만 지원합니다.")
        if not cfg.get("data_egress_confirmed", False):
            raise Stage2AgentError("data_egress_confirmed=false입니다.")
        if OpenAI is None:
            raise Stage2AgentError("openai 패키지가 없습니다.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise Stage2AgentError("OPENAI_API_KEY 환경변수가 없습니다.")
        model = cfg.get("stage2_agent_model")
        prompt_name = cfg.get("stage2_agent_prompt")
        if not model or not prompt_name:
            raise Stage2AgentError("stage2_agent_model과 stage2_agent_prompt를 명시해야 합니다.")
        self.cfg = cfg
        self.model = str(model)
        self.prompt_file = str(prompt_name)
        self.system_prompt = _prompt_path(self.prompt_file).read_text(encoding="utf-8")
        self.verifier_enabled = bool(cfg.get("stage2_plan_verifier_enabled", False))
        self.verifier_model: str | None = None
        self.verifier_prompt_file: str | None = None
        self.verifier_system_prompt: str | None = None
        if self.verifier_enabled:
            verifier_model = cfg.get("stage2_plan_verifier_model")
            verifier_prompt = cfg.get("stage2_plan_verifier_prompt")
            if not verifier_model or not verifier_prompt:
                raise Stage2AgentError(
                    "계획 검토를 켜려면 verifier 모델과 프롬프트를 명시해야 합니다.")
            self.verifier_model = str(verifier_model)
            self.verifier_prompt_file = str(verifier_prompt)
            self.verifier_system_prompt = _prompt_path(
                self.verifier_prompt_file).read_text(encoding="utf-8")
        self.dual_enabled = bool(cfg.get("stage2_dual_interpretation_enabled", False))
        self.independent_model: str | None = None
        self.independent_prompt_file: str | None = None
        self.independent_system_prompt: str | None = None
        self.adjudicator_model: str | None = None
        self.adjudicator_prompt_file: str | None = None
        self.adjudicator_system_prompt: str | None = None
        if self.dual_enabled:
            independent_model = cfg.get("stage2_independent_model")
            independent_prompt = cfg.get("stage2_independent_prompt")
            adjudicator_model = cfg.get("stage2_adjudicator_model")
            adjudicator_prompt = cfg.get("stage2_adjudicator_prompt")
            if not all((independent_model, independent_prompt,
                        adjudicator_model, adjudicator_prompt)):
                raise Stage2AgentError(
                    "독립 이중 해석에는 두 번째 해석기와 선택기 설정이 모두 필요합니다.")
            self.independent_model = str(independent_model)
            self.independent_prompt_file = str(independent_prompt)
            self.independent_system_prompt = _prompt_path(
                self.independent_prompt_file).read_text(encoding="utf-8")
            self.adjudicator_model = str(adjudicator_model)
            self.adjudicator_prompt_file = str(adjudicator_prompt)
            self.adjudicator_system_prompt = _prompt_path(
                self.adjudicator_prompt_file).read_text(encoding="utf-8")
        self.catalog_lines = document_catalog(identity, eligible_ids)
        if not self.catalog_lines:
            raise Stage2AgentError("질문 해석에 제공할 공식 문서 목록이 비어 있습니다.")
        self.valid_document_ids = [line.split(" | ", 1)[0] for line in self.catalog_lines]
        self.client = OpenAI(max_retries=int(cfg.get("api_max_retries", 0) or 0))
        self.reflection_memory_enabled = bool(
            cfg.get("reflection_memory_enabled", False))
        self.reflection_memory: ReflectionMemory | None = None
        if self.reflection_memory_enabled:
            try:
                self.reflection_memory = ReflectionMemory(cfg, self.client)
            except ReflectionMemoryError as exc:
                raise Stage2AgentError(str(exc)) from exc
        self.current_memory_A: list[dict[str, Any]] = []
        self.usage = Usage()
        self.calls = 0
        self.verifier_calls = 0
        self.independent_calls = 0
        self.adjudicator_calls = 0
        self.raw_decisions: list[dict[str, Any]] = []
        self.raw_verifications: list[dict[str, Any]] = []
        self.raw_independent_decisions: list[dict[str, Any]] = []
        self.raw_adjudications: list[dict[str, Any]] = []

    def prepare_question(self, question: str) -> list[dict[str, Any]]:
        """Retrieve approved precedents once for the current question."""
        self.current_memory_A = []
        if self.reflection_memory is not None:
            try:
                self.current_memory_A = self.reflection_memory.retrieve(
                    question, self.usage)
            except ReflectionMemoryError as exc:
                raise Stage2AgentError(str(exc)) from exc
        return list(self.current_memory_A)

    def reset_usage(self) -> None:
        self.usage = Usage()
        self.calls = 0
        self.verifier_calls = 0
        self.independent_calls = 0
        self.adjudicator_calls = 0
        self.raw_decisions = []
        self.raw_verifications = []
        self.raw_independent_decisions = []
        self.raw_adjudications = []
        self.current_memory_A = []

    def _messages(
        self, question: str, observations: list[dict[str, Any]],
        session_document_id: str | None,
    ) -> list[dict[str, str]]:
        payload = {
            "session_document_id": session_document_id,
            "official_document_catalog": self.catalog_lines,
            "question": question,
            "tool_observations": observations,
            "verified_memory_A": self.current_memory_A,
        }
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _verifier_messages(
        self, question: str, proposed_plan: Stage2Decision,
        session_document_id: str | None,
    ) -> list[dict[str, str]]:
        payload = {
            "session_document_id": session_document_id,
            "official_document_catalog": self.catalog_lines,
            "question": question,
            "proposed_plan": proposed_plan.as_dict(),
        }
        return [
            {"role": "system", "content": self.verifier_system_prompt or ""},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _independent_messages(
        self, question: str, session_document_id: str | None,
    ) -> list[dict[str, str]]:
        # Deliberately excludes the first interpretation and all of its notes.
        payload = {
            "session_document_id": session_document_id,
            "official_document_catalog": self.catalog_lines,
            "question": question,
        }
        return [
            {"role": "system", "content": self.independent_system_prompt or ""},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _adjudicator_messages(
        self, question: str, candidate_a: Stage2Decision,
        candidate_b: Stage2Decision, observation_a: dict[str, Any],
        observation_b: dict[str, Any], session_document_id: str | None,
    ) -> list[dict[str, str]]:
        payload = {
            "session_document_id": session_document_id,
            "question": question,
            "candidate_a": candidate_a.as_dict(),
            "candidate_b": candidate_b.as_dict(),
            "candidate_a_result": observation_a,
            "candidate_b_result": observation_b,
            "verified_memory_A": self.current_memory_A,
        }
        return [
            {"role": "system", "content": self.adjudicator_system_prompt or ""},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _add_response_usage(self, response) -> None:
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            self.usage.add_generation(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=cached or 0,
            )

    def independent_decide(
        self, question: str, session_document_id: str | None = None,
    ) -> Stage2Decision:
        """Create candidate B without seeing candidate A or its reasoning."""
        if not self.dual_enabled:
            raise Stage2AgentError("독립 이중 해석이 활성화되지 않았습니다.")
        self.calls += 1
        self.independent_calls += 1
        kwargs: dict[str, Any] = {
            "model": self.independent_model,
            "messages": self._independent_messages(question, session_document_id),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "stage2_independent_initial_decision",
                    "strict": True,
                    "schema": decision_json_schema(
                        self.valid_document_ids, [], []),
                },
            },
        }
        limit = self.cfg.get("stage2_independent_max_completion_tokens")
        if limit is not None:
            kwargs["max_completion_tokens"] = int(limit)
        if not model_rejects_sampling_params(str(self.independent_model)):
            temperature = self.cfg.get("stage2_independent_temperature")
            if temperature is not None:
                kwargs["temperature"] = temperature
        reasoning_effort = self.cfg.get("stage2_independent_reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(reasoning_effort)
        verbosity = self.cfg.get("stage2_independent_verbosity")
        if verbosity is not None:
            kwargs["verbosity"] = str(verbosity)
        response = self.client.chat.completions.create(**kwargs)
        self._add_response_usage(response)
        if not response.choices:
            raise Stage2AgentError("독립 해석 응답에 후보가 없습니다.")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "stop") != "stop":
            if choice.finish_reason == "length":
                raise Stage2ResponseTruncated("독립 해석 응답 길이 제한")
            raise Stage2AgentError(
                f"독립 해석 응답이 정상 종료되지 않았습니다: {choice.finish_reason}")
        if getattr(choice.message, "refusal", None):
            raise Stage2AgentError("독립 해석 요청이 거부됐습니다.")
        try:
            raw = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise Stage2AgentError("독립 해석 응답이 JSON이 아닙니다.") from exc
        self.raw_independent_decisions.append(raw)
        return parse_decision(
            raw,
            valid_document_ids=set(self.valid_document_ids),
            observation_ids=set(), evidence_ids=set(),
            session_document_id=session_document_id,
        )

    def adjudicate_initial_plans(
        self, question: str, candidate_a: Stage2Decision,
        candidate_b: Stage2Decision, observation_a: dict[str, Any],
        observation_b: dict[str, Any], session_document_id: str | None = None,
    ) -> Stage2Adjudication:
        """Choose A/B from question plus real read-only results; never invent C."""
        if not self.dual_enabled:
            raise Stage2AgentError("독립 이중 해석이 활성화되지 않았습니다.")
        self.calls += 1
        self.adjudicator_calls += 1
        kwargs: dict[str, Any] = {
            "model": self.adjudicator_model,
            "messages": self._adjudicator_messages(
                question, candidate_a, candidate_b, observation_a, observation_b,
                session_document_id),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "stage2_execution_guided_adjudication",
                    "strict": True,
                    "schema": adjudication_json_schema(),
                },
            },
        }
        limit = self.cfg.get("stage2_adjudicator_max_completion_tokens")
        if limit is not None:
            kwargs["max_completion_tokens"] = int(limit)
        if not model_rejects_sampling_params(str(self.adjudicator_model)):
            temperature = self.cfg.get("stage2_adjudicator_temperature")
            if temperature is not None:
                kwargs["temperature"] = temperature
        reasoning_effort = self.cfg.get("stage2_adjudicator_reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(reasoning_effort)
        verbosity = self.cfg.get("stage2_adjudicator_verbosity")
        if verbosity is not None:
            kwargs["verbosity"] = str(verbosity)
        response = self.client.chat.completions.create(**kwargs)
        self._add_response_usage(response)
        if not response.choices:
            raise Stage2AgentError("계획 선택 응답에 후보가 없습니다.")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "stop") != "stop":
            if choice.finish_reason == "length":
                raise Stage2ResponseTruncated("계획 선택 응답 길이 제한")
            raise Stage2AgentError(
                f"계획 선택 응답이 정상 종료되지 않았습니다: {choice.finish_reason}")
        if getattr(choice.message, "refusal", None):
            raise Stage2AgentError("계획 선택 요청이 거부됐습니다.")
        try:
            raw = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise Stage2AgentError("계획 선택 응답이 JSON이 아닙니다.") from exc
        self.raw_adjudications.append(raw)
        return parse_adjudication(raw)

    def verify_initial_plan(
        self, question: str, proposed_plan: Stage2Decision,
        session_document_id: str | None = None,
    ) -> Stage2PlanVerification:
        """Independently approve, correct, or reject the first model plan."""
        if not self.verifier_enabled:
            raise Stage2AgentError("독립 계획 검토기가 활성화되지 않았습니다.")
        if proposed_plan.result_assessment != "initial":
            raise Stage2AgentError("최초 계획만 독립 검토할 수 있습니다.")
        self.calls += 1
        self.verifier_calls += 1
        kwargs: dict[str, Any] = {
            "model": self.verifier_model,
            "messages": self._verifier_messages(
                question, proposed_plan, session_document_id),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "stage2_initial_plan_verification",
                    "strict": True,
                    "schema": verification_json_schema(self.valid_document_ids),
                },
            },
        }
        limit = self.cfg.get("stage2_plan_verifier_max_completion_tokens")
        if limit is not None:
            kwargs["max_completion_tokens"] = int(limit)
        if not model_rejects_sampling_params(str(self.verifier_model)):
            temperature = self.cfg.get("stage2_plan_verifier_temperature")
            if temperature is not None:
                kwargs["temperature"] = temperature
        reasoning_effort = self.cfg.get("stage2_plan_verifier_reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(reasoning_effort)
        verbosity = self.cfg.get("stage2_plan_verifier_verbosity")
        if verbosity is not None:
            kwargs["verbosity"] = str(verbosity)
        response = self.client.chat.completions.create(**kwargs)
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            self.usage.add_generation(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=cached or 0,
            )
        if not response.choices:
            raise Stage2AgentError("계획 검토 응답에 후보가 없습니다.")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "stop") != "stop":
            raise Stage2AgentError(
                f"계획 검토 응답이 정상 종료되지 않았습니다: {choice.finish_reason}")
        if getattr(choice.message, "refusal", None):
            raise Stage2AgentError("계획 검토 요청이 거부됐습니다.")
        try:
            raw = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise Stage2AgentError("계획 검토 응답이 JSON이 아닙니다.") from exc
        self.raw_verifications.append(raw)
        return parse_verification(
            raw,
            proposed_plan=proposed_plan,
            valid_document_ids=set(self.valid_document_ids),
            session_document_id=session_document_id,
        )

    def decide(
        self, question: str, observations: list[dict[str, Any]],
        session_document_id: str | None = None,
        locked_task_type: str | None = None,
        locked_answer_mode: str | None = None,
        allow_plan_revision: bool = False,
    ) -> Stage2Decision:
        self.calls += 1
        observation_ids = [str(item["observation_id"]) for item in observations]
        evidence_ids = [str(ev["evidence_id"]) for item in observations
                        for ev in item.get("evidence", [])]
        allowed_actions: tuple[str, ...] | None = None
        max_tool_calls = int(self.cfg.get("stage2_max_tool_calls", 3))
        if len(observations) >= max_tool_calls:
            allowed_actions = ("finalize", "clarify")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(question, observations, session_document_id),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "stage2_structgpt_decision",
                    "strict": True,
                    "schema": decision_json_schema(
                        self.valid_document_ids, observation_ids, evidence_ids,
                        locked_task_type=(None if allow_plan_revision else locked_task_type),
                        locked_answer_mode=(None if allow_plan_revision else locked_answer_mode),
                        allowed_actions=allowed_actions),
                },
            },
        }
        limit = self.cfg.get("stage2_agent_max_completion_tokens")
        if limit is not None:
            kwargs["max_completion_tokens"] = int(limit)
        if not model_rejects_sampling_params(self.model):
            temperature = self.cfg.get("stage2_agent_temperature")
            if temperature is not None:
                kwargs["temperature"] = temperature
        reasoning_effort = self.cfg.get("stage2_agent_reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(reasoning_effort)
        verbosity = self.cfg.get("stage2_agent_verbosity")
        if verbosity is not None:
            kwargs["verbosity"] = str(verbosity)
        response = self.client.chat.completions.create(**kwargs)
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            self.usage.add_generation(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=cached or 0,
            )
        if not response.choices:
            raise Stage2AgentError("Stage 2 응답에 후보가 없습니다.")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "stop") != "stop":
            if choice.finish_reason == "length":
                raise Stage2ResponseTruncated("Stage 2 응답 길이 제한")
            raise Stage2AgentError(f"Stage 2 응답이 정상 종료되지 않았습니다: {choice.finish_reason}")
        if getattr(choice.message, "refusal", None):
            raise Stage2AgentError("Stage 2 요청이 거부됐습니다.")
        try:
            raw = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise Stage2AgentError("Stage 2 응답이 JSON이 아닙니다.") from exc
        self.raw_decisions.append(raw)
        return parse_decision(
            raw,
            valid_document_ids=set(self.valid_document_ids),
            observation_ids=set(observation_ids),
            evidence_ids=set(evidence_ids),
            session_document_id=session_document_id,
        )
