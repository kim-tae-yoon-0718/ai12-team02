import json

import pytest

from planner import parse_plan


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
