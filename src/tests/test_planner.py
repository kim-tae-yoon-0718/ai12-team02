import json

import pytest

from planner import parse_plan
from answer_pipeline import (
    answer_select_by_table, answer_extract_by_table, answer_compare_by_table,
)


def test_parse_valid_select_plan():
    plan = parse_plan(json.dumps({
        "task_type": "select",
        "document_ids": [],
        "fields": [],
        "conditions": [{
            "field": "예산", "operator": ">=", "value": 500000000,
            "negated": False,
        }],
        "answer_mode": "list",
    }))
    assert plan.task_type == "select"
    assert plan.conditions[0]["field"] == "예산"


def test_parse_rejects_unknown_field():
    with pytest.raises(ValueError, match="unsupported condition field"):
        parse_plan(json.dumps({
            "task_type": "select",
            "conditions": [{"field": "발주기관", "operator": "=="}],
        }))


def test_parse_rejects_incomplete_extract_plan():
    with pytest.raises(ValueError, match="extract plan needs fields"):
        parse_plan(json.dumps({"task_type": "extract", "conditions": []}))


# ---------------------------------------------------------------------------
# 계획이 지정한 document_ids/fields가 실행 단계에서 실제로 지켜지는지
# (회귀: 코드리뷰 지적 — 계획을 검증만 하고 실행에는 충분히 쓰지 않던 문제)
# ---------------------------------------------------------------------------

class TestPlanIsActuallyEnforced:
    def test_select_scope_is_restricted_to_planned_document_ids(
            self, table_rows, base_cfg, identity_index):
        """조건만 넘기면 RFP-000001·RFP-000002가 둘 다 걸리는 조건인데,
        계획이 RFP-000001로 문서를 좁혔으면 RFP-000002는 나오면 안 된다."""
        without_plan = answer_select_by_table(
            "4,900만원 이상인 사업", table_rows, base_cfg, identity=identity_index)
        assert set(without_plan.selected_document_ids) >= {"RFP-000001", "RFP-000002"}

        with_plan = answer_select_by_table(
            "4,900만원 이상인 사업", table_rows, base_cfg, identity=identity_index,
            plan_document_ids=["RFP-000001"])
        assert with_plan.selected_document_ids == ["RFP-000001"]

    def test_extract_uses_planned_document_id_over_text_resolution(
            self, table_rows, base_cfg, identity_index, fake_clients):
        """질문 문장은 RFP-000001을 가리키지만, 계획이 RFP-000002를 특정했으면
        계획을 따라야 한다 — 문장을 다시 파싱해 다른 문서로 되돌리면 안 된다."""
        _e, _g, get_embed, get_gen = fake_clients
        result = answer_extract_by_table(
            "RFP-000001의 예산이 얼마야", table_rows, None,
            get_embed, get_gen, base_cfg,
            identity=identity_index, planned_fields=["예산"],
            planned_document_ids=["RFP-000002"])
        assert result.selected_document_ids == ["RFP-000002"]
        assert "49,500천원" in result.text

    def test_compare_uses_planned_document_ids_over_text_resolution(
            self, table_rows, base_cfg, identity_index):
        """질문에 문서를 언급하지 않아도 계획이 이미 비교 대상 두 건을
        정했으면 그대로 비교해야 한다."""
        result = answer_compare_by_table(
            "예산 비교해줘", table_rows, base_cfg, identity=identity_index,
            planned_document_ids=["RFP-000001", "RFP-000002"],
            planned_fields=["예산"])
        docs = {c["document"] for c in result.citations}
        assert docs == {"RFP-000001", "RFP-000002"}
