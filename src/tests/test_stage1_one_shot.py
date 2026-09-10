"""Stage 1: one interpretation call and one bounded tool execution."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def raw_plan(**overrides):
    data = {
        "action": "vector_search",
        "task_type": "qa",
        "document_scope": "specific_documents",
        "document_ids": ["RFP-000001"],
        "requested_fields": [],
        "condition_logic": "AND",
        "conditions": [],
        "search_query": "사업을 시작한 이유",
        "answer_mode": "summary",
        "clarification": None,
        "unsupported_parts": [],
        "decision_note": "원문 설명이 필요함",
    }
    data.update(overrides)
    return data


class FakePlanner:
    def __init__(self, plan):
        from pricing import Usage
        self.result = plan
        self.calls = 0
        self.usage = Usage()
        self.last_raw_plan = plan.as_dict()

    def reset_usage(self):
        from pricing import Usage
        self.usage = Usage()

    def plan(self, question, session_document_id=None):
        self.calls += 1
        return self.result


def scope(*ids):
    values = set(ids)
    return SimpleNamespace(
        eligible_ids=values, document_count=len(values), excluded=[],
        summary=lambda: {"eligible": len(values), "excluded": 0},
    )


class TestPlanContract:
    def test_schema_is_strict_at_every_object(self):
        from stage1_plan import plan_json_schema
        schema = plan_json_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"]["conditions"]["items"]["additionalProperties"] is False

    @pytest.mark.parametrize("action", [
        "table_select", "table_lookup", "table_compare", "vector_search", "clarify",
    ])
    def test_action_enum_is_closed(self, action):
        from stage1_plan import plan_json_schema
        assert action in plan_json_schema()["properties"]["action"]["enum"]

    def test_unknown_action_rejected(self):
        from stage1_plan import parse_plan, Stage1PlanError
        with pytest.raises(Stage1PlanError):
            parse_plan(raw_plan(action="execute_python"))

    def test_unknown_field_rejected(self):
        from stage1_plan import parse_plan, Stage1PlanError
        with pytest.raises(Stage1PlanError):
            parse_plan(raw_plan(
                action="table_lookup", task_type="extract",
                requested_fields=["모델이 만든 필드"], search_query=""))

    def test_operator_field_pair_rejected(self):
        from stage1_plan import parse_plan, Stage1PlanError
        with pytest.raises(Stage1PlanError):
            parse_plan(raw_plan(
                action="table_select", task_type="select",
                document_scope="all_population", document_ids=[], search_query="",
                conditions=[{"field": "사업기간", "operator": ">=", "value": 3,
                             "negated": False, "question_span": "3 이상"}]))

    def test_unsupported_parts_cannot_be_silently_dropped(self):
        from stage1_plan import parse_plan, validate_plan, Stage1PlanError
        plan = parse_plan(raw_plan(unsupported_parts=["두 조건 중 하나"]))
        with pytest.raises(Stage1PlanError):
            validate_plan(plan, {"RFP-000001"})

    def test_unknown_document_id_rejected(self):
        from stage1_plan import parse_plan, validate_plan, Stage1PlanError
        plan = parse_plan(raw_plan(document_ids=["RFP-999999"]))
        with pytest.raises(Stage1PlanError):
            validate_plan(plan, {"RFP-000001"})

    def test_document_id_shape_is_enforced_by_api_schema(self):
        from stage1_plan import plan_json_schema
        item = plan_json_schema()["properties"]["document_ids"]["items"]
        assert item["pattern"] == "^RFP-[0-9]{6}$"

    def test_runtime_schema_can_close_ids_to_catalog(self):
        from stage1_plan import plan_json_schema
        item = plan_json_schema(["RFP-000001", "RFP-000002"])["properties"][
            "document_ids"]["items"]
        assert item["enum"] == ["RFP-000001", "RFP-000002"]

    def test_session_document_is_filled_from_structured_state(self):
        from stage1_plan import parse_plan, validate_plan
        plan = parse_plan(raw_plan(document_scope="session_document", document_ids=[]))
        assert validate_plan(plan, {"RFP-000001"}, "RFP-000001").document_ids == ["RFP-000001"]

    def test_or_is_not_available_as_silent_and(self):
        from stage1_plan import plan_json_schema
        assert plan_json_schema()["properties"]["condition_logic"]["enum"] == ["AND"]

    def test_select_allows_only_redundant_condition_fields(self):
        from stage1_plan import parse_plan, validate_plan, Stage1PlanError
        base = raw_plan(
            action="table_select", task_type="select", document_scope="all_population",
            document_ids=[], requested_fields=["예산"], search_query="",
            answer_mode="document_set",
            conditions=[{"field": "예산", "operator": "status_value_present",
                         "value": None, "negated": False, "question_span": "예산 기재"}],
        )
        assert validate_plan(parse_plan(base), {"RFP-000001"}).action == "table_select"
        base["requested_fields"] = ["사업기간"]
        with pytest.raises(Stage1PlanError):
            validate_plan(parse_plan(base), {"RFP-000001"})

    @pytest.mark.parametrize("bad", [
        raw_plan(action="table_select", task_type="select", document_scope="all_population",
                 document_ids=[], conditions=[], search_query=""),
        raw_plan(action="table_lookup", task_type="extract", requested_fields=[]),
        raw_plan(action="table_compare", task_type="compare", document_ids=["RFP-000001"],
                 requested_fields=["예산"], search_query=""),
        raw_plan(action="vector_search", document_ids=["RFP-000001", "RFP-000002"]),
        raw_plan(action="clarify", clarification=None),
    ])
    def test_incomplete_action_is_rejected(self, bad):
        from stage1_plan import parse_plan, validate_plan, Stage1PlanError
        plan = parse_plan(bad)
        with pytest.raises(Stage1PlanError):
            validate_plan(plan, {"RFP-000001", "RFP-000002"})


class TestPlannerCall:
    def test_catalog_uses_only_eligible_documents(self, identity_index):
        from stage1_planner import document_catalog
        lines = document_catalog(identity_index, {"RFP-000002"})
        assert len(lines) == 1 and lines[0].startswith("RFP-000002 |")

    def test_request_uses_strict_json_schema_and_no_sampling_for_gpt5(
        self, monkeypatch, base_cfg, identity_index,
    ):
        import stage1_planner as module
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content=json.dumps(raw_plan(), ensure_ascii=False),
                                          refusal=None)
                choice = SimpleNamespace(message=message, finish_reason="stop")
                usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                        prompt_tokens_details=None)
                return SimpleNamespace(choices=[choice], usage=usage)

        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=Completions())

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(module, "OpenAI", Client)
        cfg = dict(base_cfg, stage1_planner_model="gpt-5-mini",
                   stage1_planner_prompt="stage1_plan_v1.txt",
                   stage1_planner_max_completion_tokens=512, api_max_retries=0)
        planner = module.Stage1Planner(cfg, identity_index, {"RFP-000001"})
        result = planner.plan("질문")
        assert result.action == "vector_search" and planner.calls == 1
        assert captured["response_format"]["json_schema"]["strict"] is True
        ids = captured["response_format"]["json_schema"]["schema"]["properties"][
            "document_ids"]["items"]["enum"]
        assert ids == ["RFP-000001"]
        assert "temperature" not in captured and captured["max_completion_tokens"] == 512


class TestOneShotExecution:
    def test_table_selection_bypasses_old_language_parsers(
        self, monkeypatch, table_rows, identity_index,
    ):
        import answer_pipeline as module
        from stage1_plan import parse_plan
        plan = parse_plan(raw_plan(
            action="table_select", task_type="select", document_scope="all_population",
            document_ids=[], requested_fields=[], search_query="", answer_mode="document_set",
            conditions=[{"field": "예산", "operator": "status_value_present", "value": None,
                         "negated": False, "question_span": "예산이 적혀 있는"}],
        ))
        planner = FakePlanner(plan)
        monkeypatch.setattr(module, "route", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("old router called")))
        monkeypatch.setattr(module, "parse_selection", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("old selection parser called")))
        result = module.answer(
            "자유로운 표현", None, lambda: None, lambda: None, table_rows,
            {"routing_method": "llm_one_shot", "deadline_filter_default": {"select": False},
             "extraction_version": "v3", "schema_version": "x", "corpus": "v2"},
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage1_planner=lambda: planner,
        )
        assert planner.calls == 1
        assert result.route == module.ROUTE_SELECT
        assert result.execution_plan["action"] == "table_select"

    def test_table_lookup_bypasses_field_and_document_parsers(
        self, monkeypatch, table_rows, identity_index,
    ):
        import answer_pipeline as module
        from stage1_plan import parse_plan
        plan = parse_plan(raw_plan(
            action="table_lookup", task_type="extract", document_scope="specific_documents",
            document_ids=["RFP-000001"], requested_fields=["예산"], search_query="",
            answer_mode="value",
        ))
        planner = FakePlanner(plan)
        for name in ("detect_fields", "resolve_document", "parse_selection"):
            monkeypatch.setattr(module, name, lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("old natural-language parser called")))
        result = module.answer(
            "자유로운 표현", None, lambda: None, lambda: None, table_rows,
            {"routing_method": "llm_one_shot", "extraction_version": "v3",
             "schema_version": "x", "corpus": "v2"}, identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage1_planner=lambda: planner,
        )
        assert planner.calls == 1 and result.selected_document_ids == ["RFP-000001"]
        assert result.route == module.ROUTE_EXTRACT_VALUE

    def test_vector_search_executes_once_without_replanning(
        self, monkeypatch, table_rows, identity_index, small_store, fake_clients,
    ):
        import answer_pipeline as module
        from stage1_plan import parse_plan
        embed, gen, get_embed, get_gen = fake_clients
        plan = parse_plan(raw_plan())
        planner = FakePlanner(plan)
        monkeypatch.setattr(module, "route", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("old router called")))
        monkeypatch.setattr(module, "resolve_document", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("old resolver called")))
        result = module.answer(
            "질문 원문", small_store, get_embed, get_gen, table_rows,
            {"routing_method": "llm_one_shot", "top_k": 2, "chunking_version": "v3"},
            identity=identity_index,
            registry_scope=scope("RFP-000001", "RFP-000002", "RFP-000003"),
            get_stage1_planner=lambda: planner,
        )
        assert planner.calls == 1 and embed.calls == 1 and gen.calls == 1
        assert result.route == module.ROUTE_SEARCH_LLM
        assert result.selected_document_ids == ["RFP-000001"]

    def test_baseline_routing_is_unchanged(self, monkeypatch, table_rows):
        import answer_pipeline as module
        called = []
        monkeypatch.setattr(module, "route", lambda q, cfg: called.append(q) or
                            SimpleNamespace(task_type="no_search_needed", matched_rule="x",
                                            is_fallback=False, no_search_kind=None))
        result = module.answer("안녕", None, lambda: None, lambda: None, table_rows,
                               {"routing_method": "rule_based"})
        assert called == ["안녕"] and result.error_stage is None

    def test_plan_failure_is_not_hidden_by_baseline_fallback(self, table_rows, identity_index):
        from answer_pipeline import answer

        class Broken:
            def plan(self, *args, **kwargs):
                raise RuntimeError("broken plan")

        result = answer(
            "질문", None, lambda: None, lambda: None, table_rows,
            {"routing_method": "llm_one_shot"}, identity=identity_index,
            registry_scope=scope("RFP-000001"), get_stage1_planner=lambda: Broken(),
        )
        assert result.error_stage == "stage1_one_shot"
        assert result.route is None
