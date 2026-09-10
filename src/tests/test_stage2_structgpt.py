"""Stage 2: bounded result-aware tool use without code-side NLP."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def raw_decision(**overrides):
    data = {
        "action": "table_lookup",
        "task_type": "extract",
        "document_scope": "specific_documents",
        "result_assessment": "initial",
        "document_ids": ["RFP-000001"],
        "requested_fields": ["필수 제출 서류"],
        "condition_logic": "AND",
        "conditions": [],
        "search_query": "",
        "answer_mode": "list",
        "clarification": None,
        "unsupported_parts": [],
        "decision_note": "구조화 값을 먼저 확인",
        "final_answer": "",
        "final_items": [],
        "used_observation_ids": [],
        "used_evidence_ids": [],
    }
    data.update(overrides)
    return data


def raw_verification(plan, **overrides):
    data = {
        "outcome": "approve",
        "checked_requirements": ["질문의 문서 범위", "요청 항목", "답변 형식"],
        "issue_types": ["none"],
        "verification_note": "질문과 계획이 일치합니다.",
        "verified_plan": dict(plan),
    }
    data.update(overrides)
    return data


def parse(raw, observations=(), evidence=(), valid=("RFP-000001", "RFP-000002")):
    from stage2_agent import parse_decision

    return parse_decision(
        raw,
        valid_document_ids=set(valid),
        observation_ids=set(observations),
        evidence_ids=set(evidence),
    )


def tool_decision(**overrides):
    observations = (
        ("O1",) if overrides.get("result_assessment", "initial") != "initial" else ()
    )
    return parse(raw_decision(**overrides), observations=observations)


def final_decision(**overrides):
    base = raw_decision(
        action="finalize",
        result_assessment="sufficient",
        document_ids=[],
        requested_fields=[],
        final_items=["제출 서류 A", "제출 서류 B"],
        used_observation_ids=["O1"],
        used_evidence_ids=["O1:E1"],
    )
    base.update(overrides)
    return parse(base, observations=("O1", "O2"), evidence=("O1:E1", "O2:E1"))


class FakeAgent:
    def __init__(self, *decisions):
        from pricing import Usage

        self.decisions = list(decisions)
        self.calls = 0
        self.usage = Usage()

    def reset_usage(self):
        from pricing import Usage

        self.usage = Usage()

    def decide(
        self,
        question,
        observations,
        session_document_id=None,
        locked_task_type=None,
        locked_answer_mode=None,
        allow_plan_revision=False,
    ):
        self.calls += 1
        return self.decisions.pop(0)


class FakeVerifierAgent(FakeAgent):
    def __init__(self, verification, *decisions):
        super().__init__(*decisions)
        self.verification = verification
        self.verifier_calls = 0
        self.raw_verifications = []

    def verify_initial_plan(self, question, proposed_plan, session_document_id=None):
        self.calls += 1
        self.verifier_calls += 1
        return self.verification


class FakeDualAgent(FakeAgent):
    def __init__(self, candidate_b, adjudication, *decisions):
        super().__init__(*decisions)
        self.candidate_b = candidate_b
        self.adjudication = adjudication
        self.independent_calls = 0
        self.adjudicator_calls = 0
        self.raw_decisions = []
        self.raw_independent_decisions = []
        self.raw_adjudications = []

    def independent_decide(self, question, session_document_id=None):
        self.calls += 1
        self.independent_calls += 1
        if isinstance(self.candidate_b, Exception):
            raise self.candidate_b
        return self.candidate_b

    def adjudicate_initial_plans(
        self,
        question,
        candidate_a,
        candidate_b,
        observation_a,
        observation_b,
        session_document_id=None,
    ):
        self.calls += 1
        self.adjudicator_calls += 1
        return self.adjudication


def scope(*ids):
    values = set(ids)
    return SimpleNamespace(
        eligible_ids=values,
        document_count=len(values),
        excluded=[],
        summary=lambda: {"eligible": len(values), "excluded": 0},
    )


class TestDecisionContract:
    def test_schema_is_strict_and_closes_every_action(self):
        from stage2_agent import DECISION_ACTIONS, decision_json_schema

        schema = decision_json_schema(["RFP-000001"], [], [])
        assert schema["additionalProperties"] is False
        assert (
            schema["properties"]["conditions"]["items"]["additionalProperties"] is False
        )
        assert schema["properties"]["action"]["enum"] == list(DECISION_ACTIONS)
        assert "result_assessment" in schema["required"]
        assert schema["properties"]["result_assessment"]["enum"] == ["initial"]
        assert schema["properties"]["document_ids"]["items"]["enum"] == ["RFP-000001"]
        assert schema["properties"]["used_observation_ids"]["maxItems"] == 0
        assert schema["properties"]["used_evidence_ids"]["maxItems"] == 0

    def test_verifier_schema_is_strict_and_embeds_an_initial_plan(self):
        from stage2_agent import verification_json_schema

        schema = verification_json_schema(["RFP-000001"])
        assert schema["additionalProperties"] is False
        assert schema["properties"]["outcome"]["enum"] == [
            "approve",
            "correct",
            "clarify",
        ]
        nested = schema["properties"]["verified_plan"]
        assert nested["additionalProperties"] is False
        assert nested["properties"]["result_assessment"]["enum"] == ["initial"]

    def test_verifier_approve_cannot_change_the_proposed_plan(self):
        from stage2_agent import Stage2AgentError, parse_verification

        proposed = tool_decision()
        approved = parse_verification(
            raw_verification(raw_decision()),
            proposed,
            valid_document_ids={"RFP-000001"},
        )
        assert approved.outcome == "approve"
        changed = raw_decision(requested_fields=["예산"], answer_mode="value")
        with pytest.raises(Stage2AgentError, match="approve"):
            parse_verification(
                raw_verification(changed), proposed, valid_document_ids={"RFP-000001"}
            )

    def test_verifier_correction_must_be_a_real_valid_change(self):
        from stage2_agent import Stage2AgentError, parse_verification

        proposed = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "컨소시엄 요건",
                    "operator": "consortium_required",
                    "value": None,
                    "negated": False,
                    "question_span": "공동수급 요건이 적힌",
                }
            ],
        )
        corrected_raw = raw_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "컨소시엄 요건",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "공동수급 요건이 적힌",
                }
            ],
        )
        corrected = parse_verification(
            raw_verification(
                corrected_raw,
                outcome="correct",
                issue_types=["condition_semantics"],
                verification_note="기재 여부 질문을 필수 여부로 해석했습니다.",
            ),
            proposed,
            valid_document_ids={"RFP-000001"},
        )
        assert (
            corrected.verified_plan.tool_plan.conditions[0].operator
            == "status_value_present"
        )
        with pytest.raises(Stage2AgentError, match="실제로 수정"):
            parse_verification(
                raw_verification(
                    proposed.as_dict(),
                    outcome="correct",
                    issue_types=["condition_semantics"],
                ),
                proposed,
                valid_document_ids={"RFP-000001"},
            )

    def test_structural_comparison_ignores_order_notes_and_question_spans(self):
        from stage2_agent import decisions_structurally_equal

        first = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=["예산", "지역제한"],
            answer_mode="document_set",
            decision_note="첫 해석",
            conditions=[
                {
                    "field": "예산",
                    "operator": ">=",
                    "value": 500000000,
                    "negated": False,
                    "question_span": "5억원 이상",
                },
                {
                    "field": "지역제한",
                    "operator": "no_restriction",
                    "value": None,
                    "negated": False,
                    "question_span": "지역 제한 없는",
                },
            ],
        )
        second = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=["지역제한", "예산"],
            answer_mode="document_set",
            decision_note="다른 표현",
            conditions=[
                {
                    "field": "지역제한",
                    "operator": "no_restriction",
                    "value": None,
                    "negated": False,
                    "question_span": "지역제한이 없는",
                },
                {
                    "field": "예산",
                    "operator": ">=",
                    "value": 500000000,
                    "negated": False,
                    "question_span": "예산 조건",
                },
            ],
        )
        assert decisions_structurally_equal(first, second)
        second.conditions[0]["operator"] = "has_restriction"
        assert not decisions_structurally_equal(first, second)

    def test_adjudication_contract_never_creates_a_third_plan(self):
        from stage2_agent import adjudication_json_schema, parse_adjudication

        schema = adjudication_json_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"]["selected_candidate"]["enum"] == [
            "candidate_a",
            "candidate_b",
            "clarify",
        ]
        selected = parse_adjudication(
            {
                "selected_candidate": "candidate_b",
                "checked_requirements": ["조건 의미"],
                "decision_note": "B가 질문의 기재 상태를 보존함",
                "clarification": None,
            }
        )
        assert selected.selected_candidate == "candidate_b"
        copied_clarification = parse_adjudication(
            {
                "selected_candidate": "candidate_b",
                "checked_requirements": ["질문 범위"],
                "decision_note": "B의 되묻기 후보를 선택함",
                "clarification": "어느 문서를 말씀하시나요?",
            }
        )
        assert copied_clarification.selected_candidate == "candidate_b"
        with pytest.raises(Exception, match="되묻기"):
            parse_adjudication(
                {
                    "selected_candidate": "clarify",
                    "checked_requirements": [],
                    "decision_note": "모호함",
                    "clarification": None,
                }
            )

    def test_requested_fields_are_canonicalized_without_reading_text(self):
        decision = tool_decision(requested_fields=["필수 제출 서류", "필수 제출 서류"])
        assert decision.requested_fields == ["필수 제출 서류"]
        assert decision.tool_plan.requested_fields == ["필수 제출 서류"]

    def test_empty_unanswerable_text_uses_fixed_transport_message(self):
        decision = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                answer_mode="unanswerable",
                final_answer="",
                final_items=[],
                document_ids=[],
                requested_fields=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        assert decision.final_answer == "제공된 근거로 답을 확정할 수 없습니다."

    def test_clarify_after_insufficient_result_is_a_valid_terminal_choice(self):
        decision = parse(
            raw_decision(
                action="clarify",
                result_assessment="insufficient",
                answer_mode="list",
                clarification="원문을 더 찾아볼까요?",
                final_answer="",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        assert decision.action == "clarify"

    def test_list_text_in_wrong_strict_slot_is_canonicalized_to_one_item(self):
        decision = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                answer_mode="list",
                final_answer="원문에 해당 항목이 없습니다.",
                final_items=[],
                document_ids=[],
                requested_fields=[],
                used_observation_ids=["O1"],
                used_evidence_ids=["O1:E1"],
            ),
            observations=("O1",),
            evidence=("O1:E1",),
        )
        assert decision.final_items == ["원문에 해당 항목이 없습니다."]

    def test_followup_schema_locks_the_initial_task_type(self):
        from stage2_agent import decision_json_schema

        schema = decision_json_schema(
            ["RFP-000001"],
            ["O1"],
            ["O1:E1"],
            locked_task_type="extract",
            locked_answer_mode="list",
        )
        assert schema["properties"]["task_type"]["enum"] == ["extract"]
        assert schema["properties"]["answer_mode"]["enum"] == ["list", "unanswerable"]
        assert schema["properties"]["result_assessment"]["enum"] == [
            "sufficient",
            "insufficient",
            "plan_wrong",
        ]

    def test_schema_can_force_terminal_action_after_complete_tool_result(self):
        from stage2_agent import decision_json_schema

        schema = decision_json_schema(
            ["RFP-000001"],
            ["O1"],
            [],
            locked_task_type="select",
            locked_answer_mode="document_set",
            allowed_actions=("finalize", "clarify"),
        )
        assert schema["properties"]["action"]["enum"] == ["finalize", "clarify"]

    def test_result_assessment_matches_whether_an_observation_exists(self):
        from stage2_agent import Stage2AgentError

        with pytest.raises(Stage2AgentError, match="결과를 평가"):
            parse(raw_decision(result_assessment="initial"), observations=("O1",))
        with pytest.raises(Stage2AgentError, match="initial"):
            parse(raw_decision(result_assessment="insufficient"))

    def test_arbitrary_code_action_is_rejected(self):
        from stage2_agent import Stage2AgentError

        with pytest.raises(Stage2AgentError):
            parse(raw_decision(action="execute_python"))

    @pytest.mark.parametrize("field", ["사업명처럼 보이는 새 필드", "원문을 대충 읽기"])
    def test_unknown_table_field_is_rejected(self, field):
        from stage2_agent import Stage2AgentError

        with pytest.raises(Stage2AgentError):
            parse(raw_decision(requested_fields=[field]))

    def test_tool_call_ignores_early_final_text_and_keeps_auditable_raw_value(self):
        decision = parse(raw_decision(final_items=["아직 쓰지 않는 답"]))
        assert decision.action == "table_lookup"
        assert decision.tool_plan is not None
        assert decision.final_items == ["아직 쓰지 않는 답"]

    @pytest.mark.parametrize(
        "key,value",
        [
            ("used_observation_ids", ["O9"]),
            ("used_evidence_ids", ["O9:E1"]),
        ],
    )
    def test_finalize_rejects_unknown_result_or_evidence(self, key, value):
        from stage2_agent import Stage2AgentError

        raw = raw_decision(
            action="finalize",
            result_assessment="sufficient",
            document_ids=[],
            requested_fields=[],
            final_items=["A"],
            used_observation_ids=["O1"],
            used_evidence_ids=["O1:E1"],
        )
        raw[key] = value
        with pytest.raises(Stage2AgentError):
            parse(raw, observations=("O1",), evidence=("O1:E1",))

    def test_finalize_requires_an_observed_result(self):
        from stage2_agent import Stage2AgentError

        with pytest.raises(Stage2AgentError):
            parse(
                raw_decision(
                    action="finalize",
                    document_ids=[],
                    requested_fields=[],
                    final_items=["A"],
                )
            )

    def test_catalog_based_unanswerable_can_finish_without_a_tool(self):
        decision = parse(
            raw_decision(
                action="finalize",
                task_type="qa",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="unanswerable",
                final_answer="공식 문서 목록에서 찾을 수 없습니다.",
                final_items=[],
                used_observation_ids=[],
                used_evidence_ids=[],
            )
        )
        assert decision.answer_mode == "unanswerable"

    def test_insufficient_result_can_end_as_unanswerable(self):
        decision = parse(
            raw_decision(
                action="finalize",
                result_assessment="insufficient",
                document_ids=[],
                requested_fields=[],
                answer_mode="unanswerable",
                final_answer="근거가 부족해 답할 수 없습니다.",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        assert decision.answer_mode == "unanswerable"

    def test_list_and_value_use_different_output_slots(self):
        from stage2_agent import Stage2AgentError

        assert final_decision().final_items == ["제출 서류 A", "제출 서류 B"]
        value = final_decision(
            answer_mode="value", final_items=[], final_answer="10억원"
        )
        assert value.final_answer == "10억원"
        with pytest.raises(Stage2AgentError):
            final_decision(answer_mode="value", final_answer="", final_items=["10억원"])

    def test_unused_output_slot_is_ignored_instead_of_discarding_grounded_answer(self):
        list_answer = final_decision(final_answer="중복 요약")
        assert list_answer.final_items == ["제출 서류 A", "제출 서류 B"]
        value_answer = final_decision(
            answer_mode="value", final_answer="10억원", final_items=["중복 목록"]
        )
        assert value_answer.final_answer == "10억원"

    def test_final_answer_requires_declared_evidence(self):
        from stage2_agent import Stage2AgentError

        with pytest.raises(Stage2AgentError):
            final_decision(used_evidence_ids=[])

    def test_clarify_ignores_inert_strict_output_slots(self):
        decision = parse(
            raw_decision(
                action="clarify",
                document_ids=[],
                requested_fields=[],
                clarification="어느 문서인가요?",
                final_answer="사용되지 않는 값",
            )
        )
        assert decision.action == "clarify"
        assert decision.clarification == "어느 문서인가요?"

    def test_document_set_ignores_inert_final_text(self):
        decision = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_answer="다시 쓰지 않는 설명",
                final_items=["다시 쓰지 않는 목록"],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        assert decision.answer_mode == "document_set"

    def test_table_select_sanitizes_non_executable_slots(self):
        decision = parse(
            raw_decision(
                action="table_select",
                task_type="select",
                document_scope="all_population",
                document_ids=["RFP-000001"],
                requested_fields=["사업 개요", "공고일", "지역제한"],
                answer_mode="document_set",
                conditions=[
                    {
                        "field": "지역제한",
                        "operator": "has_restriction",
                        "value": None,
                        "negated": False,
                        "question_span": "지역 제한",
                    }
                ],
            )
        )
        assert decision.tool_plan.document_ids == []
        assert decision.tool_plan.requested_fields == ["지역제한"]

    def test_status_operator_cannot_silently_ignore_a_search_value(self):
        from stage2_agent import Stage2AgentError

        with pytest.raises(Stage2AgentError, match="value는 null"):
            parse(
                raw_decision(
                    action="table_select",
                    task_type="select",
                    document_scope="all_population",
                    document_ids=[],
                    requested_fields=["사업 개요"],
                    answer_mode="document_set",
                    conditions=[
                        {
                            "field": "사업 개요",
                            "operator": "status_value_present",
                            "value": "존재하지 않는 사업명",
                            "negated": False,
                            "question_span": "존재하지 않는 사업",
                        }
                    ],
                )
            )

    def test_terminal_decision_may_repeat_but_never_executes_tool_inputs(self):
        decision = final_decision(
            document_ids=["RFP-000001"],
            requested_fields=["필수 제출 서류"],
            search_query="이 값은 실행되지 않음",
        )
        assert decision.action == "finalize"
        assert decision.document_ids == ["RFP-000001"]


class TestAgentCall:
    def test_openai_call_receives_observations_and_strict_dynamic_schema(
        self,
        monkeypatch,
        base_cfg,
        identity_index,
    ):
        import stage2_agent as module

        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                raw = raw_decision(
                    action="finalize",
                    result_assessment="sufficient",
                    document_ids=[],
                    requested_fields=[],
                    final_items=["A"],
                    used_observation_ids=["O1"],
                    used_evidence_ids=["O1:E1"],
                )
                message = SimpleNamespace(
                    content=json.dumps(raw, ensure_ascii=False), refusal=None
                )
                choice = SimpleNamespace(message=message, finish_reason="stop")
                usage = SimpleNamespace(
                    prompt_tokens=20, completion_tokens=10, prompt_tokens_details=None
                )
                return SimpleNamespace(choices=[choice], usage=usage)

        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=Completions())

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(module, "OpenAI", Client)
        cfg = dict(
            base_cfg,
            data_egress_confirmed=True,
            stage2_agent_model="gpt-5-mini",
            stage2_agent_prompt="stage2_agent_v1.txt",
            stage2_agent_max_completion_tokens=512,
            stage2_agent_reasoning_effort="low",
            stage2_agent_verbosity="low",
            api_max_retries=0,
        )
        agent = module.Stage2Agent(cfg, identity_index, {"RFP-000001"})
        observations = [
            {
                "observation_id": "O1",
                "tool": "table_lookup",
                "answer_text": "A",
                "evidence": [{"evidence_id": "O1:E1", "citation": {}, "content": "A"}],
            }
        ]
        result = agent.decide(
            "질문", observations, locked_task_type="extract", locked_answer_mode="list"
        )
        assert result.action == "finalize"
        assert captured["response_format"]["json_schema"]["strict"] is True
        schema = captured["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["used_evidence_ids"]["items"]["enum"] == ["O1:E1"]
        assert schema["properties"]["task_type"]["enum"] == ["extract"]
        assert schema["properties"]["answer_mode"]["enum"] == ["list", "unanswerable"]
        payload = json.loads(captured["messages"][1]["content"])
        assert payload["tool_observations"] == observations
        assert "temperature" not in captured
        assert captured["reasoning_effort"] == "low"
        assert captured["verbosity"] == "low"

    def test_independent_verifier_uses_its_own_prompt_model_and_schema(
        self,
        monkeypatch,
        base_cfg,
        identity_index,
    ):
        import stage2_agent as module

        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                raw = raw_verification(raw_decision())
                message = SimpleNamespace(
                    content=json.dumps(raw, ensure_ascii=False), refusal=None
                )
                choice = SimpleNamespace(message=message, finish_reason="stop")
                usage = SimpleNamespace(
                    prompt_tokens=30, completion_tokens=12, prompt_tokens_details=None
                )
                return SimpleNamespace(choices=[choice], usage=usage)

        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=Completions())

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(module, "OpenAI", Client)
        cfg = dict(
            base_cfg,
            data_egress_confirmed=True,
            stage2_agent_model="gpt-5-mini",
            stage2_agent_prompt="stage2_agent_v1.txt",
            stage2_plan_verifier_enabled=True,
            stage2_plan_verifier_model="gpt-5-mini",
            stage2_plan_verifier_prompt="stage2_plan_verifier_v1.txt",
            stage2_plan_verifier_max_completion_tokens=1024,
            stage2_plan_verifier_reasoning_effort="medium",
            stage2_plan_verifier_verbosity="low",
            api_max_retries=0,
        )
        agent = module.Stage2Agent(cfg, identity_index, {"RFP-000001"})
        proposed = tool_decision()
        result = agent.verify_initial_plan("질문", proposed)
        assert result.outcome == "approve"
        assert agent.verifier_calls == 1
        assert captured["model"] == "gpt-5-mini"
        assert (
            captured["response_format"]["json_schema"]["name"]
            == "stage2_initial_plan_verification"
        )
        assert captured["reasoning_effort"] == "medium"
        payload = json.loads(captured["messages"][1]["content"])
        assert payload["proposed_plan"]["action"] == "table_lookup"
        assert "계획을 독립적으로 검토" in captured["messages"][0]["content"]


class TestBoundedLoop:
    def test_dual_interpreters_agree_then_execute_once(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module
        from stage2_agent import Stage2Adjudication

        condition = {
            "field": "예산",
            "operator": "status_value_present",
            "value": None,
            "negated": False,
            "question_span": "예산이 적힌",
        }
        candidate_a = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[condition],
            decision_note="A",
        )
        candidate_b = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=["예산"],
            answer_mode="document_set",
            conditions=[dict(condition, question_span="예산 항목이 있는")],
            decision_note="B",
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_answer="",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        agent = FakeDualAgent(
            candidate_b, Stage2Adjudication("candidate_a"), candidate_a, final
        )
        result = module.answer(
            "예산이 적힌 공고를 모두 찾아줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_dual_interpretation_enabled": True,
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert result.execution_plan["initial_structural_agreement"] is True
        assert result.execution_plan["tool_calls"] == 1
        assert agent.independent_calls == 1
        assert agent.adjudicator_calls == 0

    def test_dual_disagreement_executes_both_then_uses_selected_result(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module
        from stage2_agent import Stage2Adjudication

        wrong = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "지역제한",
                    "operator": "no_restriction",
                    "value": None,
                    "negated": False,
                    "question_span": "지역제한 항목이 적힌",
                }
            ],
        )
        correct = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "지역제한",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "지역제한 항목이 적힌",
                }
            ],
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_answer="",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        agent = FakeDualAgent(
            correct,
            Stage2Adjudication(
                selected_candidate="candidate_b",
                checked_requirements=["기재 상태"],
                decision_note="B가 기재 여부를 보존함",
            ),
            wrong,
            final,
        )
        result = module.answer(
            "지역제한 항목이 적힌 공고를 모두 찾아줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_dual_interpretation_enabled": True,
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert result.execution_plan["initial_structural_agreement"] is False
        assert (
            result.execution_plan["initial_adjudication"]["selected_candidate"]
            == "candidate_b"
        )
        assert result.execution_plan["tool_calls"] == 2
        assert result.execution_plan["selected_path_tool_calls"] == 1
        assert result.selected_document_ids == ["RFP-000002"]

    def test_one_structurally_invalid_candidate_does_not_discard_valid_one(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module
        from stage2_agent import Stage2AgentError

        candidate_a = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "예산",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "예산이 적힌",
                }
            ],
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_answer="",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        agent = FakeDualAgent(
            Stage2AgentError("구조 계약 위반"), None, candidate_a, final
        )
        result = module.answer(
            "예산이 적힌 공고를 모두 찾아줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_dual_interpretation_enabled": True,
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert (
            result.execution_plan["initial_adjudication"]["selection_method"]
            == "structural_validity"
        )
        assert "candidate_b" in result.execution_plan["invalid_initial_candidates"]
        assert result.execution_plan["tool_calls"] == 1

    def test_independent_verifier_corrects_initial_semantics_before_any_tool_runs(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module
        from stage2_agent import parse_verification

        wrong = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "지역제한",
                    "operator": "no_restriction",
                    "value": None,
                    "negated": False,
                    "question_span": "지역제한 항목이 적힌",
                }
            ],
        )
        corrected_raw = raw_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "지역제한",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "지역제한 항목이 적힌",
                }
            ],
        )
        verification = parse_verification(
            raw_verification(
                corrected_raw,
                outcome="correct",
                issue_types=["condition_semantics"],
                verification_note="기재 여부를 제한 없음으로 잘못 해석했습니다.",
            ),
            wrong,
            valid_document_ids={"RFP-000001", "RFP-000002", "RFP-000003"},
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_answer="",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        agent = FakeVerifierAgent(verification, wrong, final)
        result = module.answer(
            "지역제한 항목이 적힌 공고를 모두 찾아줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_plan_verifier_enabled": True,
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert result.selected_document_ids == ["RFP-000002"]
        assert result.execution_plan["tool_calls"] == 1
        assert (
            result.execution_plan["initial_plan_proposal"]["conditions"][0]["operator"]
            == "no_restriction"
        )
        assert (
            result.execution_plan["initial_plan_verification"]["outcome"] == "correct"
        )
        assert (
            result.execution_plan["decisions"][0]["conditions"][0]["operator"]
            == "status_value_present"
        )
        assert agent.verifier_calls == 1

    def test_table_selection_observation_is_compact(self):
        import answer_pipeline as module

        plan = SimpleNamespace(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=["예산"],
            condition_logic="AND",
            conditions=[],
            search_query="",
            answer_mode="document_set",
        )
        result = SimpleNamespace(
            selected_document_ids=[f"RFP-{index:06d}" for index in range(1, 99)],
            abstained=False,
        )
        observation = module._stage2_observation("O1", plan, result)
        assert observation["evidence"] == []
        assert observation["structured_result"] == {"document_count": 98}
        assert observation["tool_input"]["action"] == "table_select"
        assert len(json.dumps(observation, ensure_ascii=False)) < 3000

    def test_lookup_then_search_then_finalize_uses_only_declared_evidence(
        self,
        monkeypatch,
        table_rows,
        identity_index,
        small_store,
        fake_clients,
    ):
        import answer_pipeline as module

        embed, _, get_embed, _ = fake_clients
        lookup = tool_decision()
        search = tool_decision(
            action="vector_search",
            task_type="extract",
            requested_fields=[],
            search_query="필수 제출 서류 전체",
            answer_mode="list",
            result_assessment="insufficient",
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                document_ids=[],
                requested_fields=[],
                final_items=["원문에서 확인한 서류"],
                used_observation_ids=["O1", "O2"],
                used_evidence_ids=["O2:E1"],
            ),
            observations=("O1", "O2"),
            evidence=("O1:E1", "O2:E1"),
        )
        agent = FakeAgent(lookup, search, final)
        for name in ("route", "parse_selection", "resolve_document", "detect_fields"):
            monkeypatch.setattr(
                module,
                name,
                lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("code-side language parser called")
                ),
            )
        result = module.answer(
            "자유로운 자연어 질문",
            small_store,
            get_embed,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "top_k": 2,
                "stage2_max_tool_calls": 3,
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
                "chunking_version": "v3",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert result.text == "원문에서 확인한 서류"
        assert result.route == module.ROUTE_STRUCTURED_LLM
        assert result.execution_plan["tool_calls"] == 2
        assert [d["action"] for d in result.execution_plan["decisions"]] == [
            "table_lookup",
            "vector_search",
            "finalize",
        ]
        assert embed.calls == 1
        assert len(result.citations) == 1
        assert result.citations[0]["document"] == "RFP-000001"
        assert result.citations[0]["source"] == "chunks_v3"

    def test_table_selection_is_deterministic_and_not_rewritten(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        select = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "예산",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "예산이 적힌",
                }
            ],
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=("O1:E1",),
        )
        agent = FakeAgent(select, final)
        result = module.answer(
            "질문",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None and result.route == module.ROUTE_SELECT
        assert result.execution_plan["tool_calls"] == 1

    def test_existing_followup_turn_can_revise_a_wrong_initial_plan_once(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        wrong_select = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "사업 개요",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "사업이 있는지",
                }
            ],
        )
        correction = parse(
            raw_decision(
                action="finalize",
                result_assessment="plan_wrong",
                task_type="qa",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="unanswerable",
                final_answer="공식 문서 목록에서 찾을 수 없습니다.",
                final_items=[],
                used_observation_ids=[],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        agent = FakeAgent(wrong_select, correction)
        result = module.answer(
            "존재하지 않는 특정 사업이 있어?",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert result.task_type == "qa"
        assert result.abstained is True
        assert result.execution_plan["plan_revision_count"] == 1
        assert result.execution_plan["tool_calls"] == 1
        assert agent.calls == 2

    def test_sufficient_repeated_structured_action_does_not_execute_twice(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        condition = {
            "field": "예산",
            "operator": "status_value_present",
            "value": None,
            "negated": False,
            "question_span": "예산",
        }
        select = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[condition],
        )
        sufficient = tool_decision(
            action="table_select",
            result_assessment="sufficient",
            task_type="select",
            document_scope="all_population",
            document_ids=["RFP-000001"],
            requested_fields=["예산"],
            answer_mode="document_set",
            conditions=[condition],
            used_observation_ids=["O1"],
        )
        agent = FakeAgent(select, sufficient)
        result = module.answer(
            "예산이 적힌 공고",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        assert result.execution_plan["tool_calls"] == 1
        assert result.route_matched_rule == "stage2_structgpt:sufficient_result"

    def test_duplicate_tool_call_is_blocked_without_fallback(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        first = tool_decision()
        agent = FakeAgent(first, tool_decision(result_assessment="insufficient"))
        result = module.answer(
            "질문",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage == "stage2_structgpt"
        assert "반복" in result.error_detail
        assert result.route is None

    def test_task_type_drift_is_blocked(self, table_rows, identity_index):
        import answer_pipeline as module

        first = tool_decision()
        drift = tool_decision(
            action="vector_search",
            task_type="qa",
            requested_fields=[],
            search_query="근거",
            answer_mode="summary",
            result_assessment="insufficient",
        )
        agent = FakeAgent(first, drift)
        result = module.answer(
            "질문",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage == "stage2_structgpt"
        assert "task_type" in result.error_detail

    def test_tool_budget_is_enforced(
        self, table_rows, identity_index, small_store, fake_clients
    ):
        import answer_pipeline as module

        _, _, get_embed, _ = fake_clients
        lookup = tool_decision()
        search = tool_decision(
            action="vector_search",
            requested_fields=[],
            search_query="근거",
            result_assessment="insufficient",
        )
        agent = FakeAgent(lookup, search)
        result = module.answer(
            "질문",
            small_store,
            get_embed,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 1,
                "top_k": 2,
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
                "chunking_version": "v3",
            },
            identity=identity_index,
            registry_scope=scope("RFP-000001"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage == "stage2_structgpt"
        assert "최대 도구 호출" in result.error_detail

    def test_agent_failure_never_falls_back_to_old_router(
        self, monkeypatch, table_rows
    ):
        import answer_pipeline as module

        class Broken:
            def decide(self, *args, **kwargs):
                raise RuntimeError("broken agent")

        monkeypatch.setattr(
            module,
            "route",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("old router called")),
        )
        result = module.answer(
            "질문",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {"routing_method": "llm_structgpt"},
            get_stage2_agent=lambda: Broken(),
        )
        assert result.error_stage == "stage2_structgpt"
        assert result.route is None
        assert result.execution_plan["tool_calls"] == 0

    def test_baseline_and_stage1_routes_remain_separate(self, monkeypatch, table_rows):
        import answer_pipeline as module

        called = []
        monkeypatch.setattr(
            module,
            "route",
            lambda q, cfg: called.append(q)
            or SimpleNamespace(
                task_type="no_search_needed",
                matched_rule="x",
                is_fallback=False,
                no_search_kind=None,
            ),
        )
        result = module.answer(
            "안녕",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {"routing_method": "rule_based"},
        )
        assert called == ["안녕"] and result.error_stage is None


class RecordingAgent(FakeAgent):
    """FakeAgent + decide()에 실제로 전달된 session_document_id를 기록."""

    def __init__(self, *decisions):
        super().__init__(*decisions)
        self.seen_session_document_ids: list[str | None] = []

    def decide(
        self,
        question,
        observations,
        session_document_id=None,
        locked_task_type=None,
        locked_answer_mode=None,
        allow_plan_revision=False,
    ):
        self.seen_session_document_ids.append(session_document_id)
        return super().decide(
            question,
            observations,
            session_document_id=session_document_id,
            locked_task_type=locked_task_type,
            locked_answer_mode=locked_answer_mode,
            allow_plan_revision=allow_plan_revision,
        )


class TestMultiTurnSessionContinuity:
    """멀티턴 데모(연속 SessionState; 시나리오형 질문)와 CLI(매번 새 세션) 사이의 괴리 조정.

    CLI에서의 질문은 session_document_id가 항상 None임(멀티턴 불가)
    데모는 연속 세션 가능
    """

    def test_clarify_with_single_known_document_persists_active_document(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        clarify = parse(
            raw_decision(
                action="clarify",
                clarification="원문 그대로(value) 드릴까요, 요약(summary)으로 드릴까요?",
                document_ids=["RFP-000001"],
            )
        )
        agent = FakeAgent(clarify)
        agent.valid_document_ids = ["RFP-000001", "RFP-000002"]
        session = module.SessionState()
        result = module.answer(
            "건설통합시스템(CMS) 고도화 사업의 사업 목적이 뭐야?",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            session=session,
            registry_scope=scope("RFP-000001", "RFP-000002"),
            get_stage2_agent=lambda: agent,
        )
        assert result.abstained is True
        assert result.route == module.ROUTE_CLARIFY
        # 도구를 실행하지 않고 되묻기로 끝난 턴이지만, 모델이 이미 특정한
        # 문서 1건이 공식 목록에 있으므로 다음 턴을 위해 세션에 남는다.
        assert session.active_document_id == "RFP-000001"

    def test_clarify_does_not_trust_document_id_outside_official_catalog(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        clarify = parse(
            raw_decision(
                action="clarify",
                clarification="어느 문서를 말씀하시는 건가요?",
                document_ids=["RFP-000001"],
            ),
            valid=("RFP-000001",),
        )
        agent = FakeAgent(clarify)
        agent.valid_document_ids = ["RFP-000002"]  # 이번 실행의 공식 목록엔 없음
        session = module.SessionState()
        module.answer(
            "그 사업 요약해줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            session=session,
            registry_scope=scope("RFP-000002"),
            get_stage2_agent=lambda: agent,
        )
        assert session.active_document_id is None

    def test_broad_selection_question_ignores_stale_active_document(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        select = tool_decision(
            action="table_select",
            task_type="select",
            document_scope="all_population",
            document_ids=[],
            requested_fields=[],
            answer_mode="document_set",
            conditions=[
                {
                    "field": "예산",
                    "operator": "status_value_present",
                    "value": None,
                    "negated": False,
                    "question_span": "예산이 적힌",
                }
            ],
        )
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                task_type="select",
                document_scope="all_population",
                document_ids=[],
                requested_fields=[],
                answer_mode="document_set",
                final_answer="",
                final_items=[],
                used_observation_ids=["O1"],
                used_evidence_ids=[],
            ),
            observations=("O1",),
            evidence=(),
        )
        agent = RecordingAgent(select, final)
        # 직전 턴에서 특정 문서 하나가 활성 상태로 남아 있는, 데모의 연속 세션 상황.
        session = module.SessionState(active_document_id="RFP-000001")
        result = module.answer(
            "예산이 적힌 공고를 모두 찾아줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "deadline_filter_default": {"select": False},
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            session=session,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        # 광범위 선별형 질문에는 낡은 활성 문서를 모델에 넘기지 않는다.
        assert agent.seen_session_document_ids == [None, None]
        # 세션 자체는 지우지 않는다 — 이번 턴 모델 입력에서만 숨겼을 뿐이다.
        assert session.active_document_id == "RFP-000001"

    def test_document_specific_question_still_uses_active_document(
        self,
        table_rows,
        identity_index,
    ):
        import answer_pipeline as module

        lookup = tool_decision()  # table_lookup, RFP-000001, extract
        final = parse(
            raw_decision(
                action="finalize",
                result_assessment="sufficient",
                document_ids=[],
                requested_fields=[],
                final_items=["제출 서류 A"],
                used_observation_ids=["O1"],
                used_evidence_ids=["O1:E1"],
            ),
            observations=("O1",),
            evidence=("O1:E1",),
        )
        agent = RecordingAgent(lookup, final)
        session = module.SessionState(active_document_id="RFP-000001")
        result = module.answer(
            "그 사업 필수 제출 서류 알려줘",
            None,
            lambda: None,
            lambda: None,
            table_rows,
            {
                "routing_method": "llm_structgpt",
                "stage2_max_tool_calls": 3,
                "extraction_version": "v3",
                "schema_version": "x",
                "corpus": "v2",
            },
            identity=identity_index,
            session=session,
            registry_scope=scope("RFP-000001", "RFP-000002"),
            get_stage2_agent=lambda: agent,
        )
        assert result.error_stage is None
        # 문서 특정형 질문(선별형이 아님)이면 활성 문서를 그대로 모델에 넘긴다.
        assert agent.seen_session_document_ids == ["RFP-000001", "RFP-000001"]
