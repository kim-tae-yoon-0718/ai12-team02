import json
import argparse
from pathlib import Path 
from typing import Any
from collections import namedtuple


# status: "PASS" | "FAIL" | "SKIP"
CheckResult = namedtuple("CheckResult", ["name", "status", "message"])

# reference_time은 정해진 상수값
REFERENCE_TIME = "2024-06-01"
FIELD_SPEC = {
    "id": "string",
    "question": "string",
    "task_type": ("selection", "extraction", "qa"),
    "answer_type": ("document_set", "value", "list", "summary", "comparison", "unanswerable"),
    "answer_raw": "any",
    "document_id": "string",
    "unspecified_type": ("abbreviation", "org_only", "time_reference", "ambiguous_match"),
    "intermediate_answer": ["string", "array"],
    "scenario_type": ("workflow_chain", "anaphora", "condition_add", "selection_to_extraction"),
    "active_document_id": "string",
    "reference_time": REFERENCE_TIME,
    "field_tag": ("critical", "major", "minor"),
    "answer_source": ("table", "verified", "metadata"),
    'answer_normalized': "any",
    "location": {"document", "section", "ref_no"},
}
TASK_ANSWER_COMBOS ={
    "selection": ("document_set",),    # answer_type 값만
    "extraction": ("value", "list"),
    "qa": ("value", "summary", "comparison", "unanswerable"),
}
FIELD_DEPENDENCIES = {
    "unspecified_type": ["intermediate_answer"],
    "scenario_type": ["active_document_id"],
}
# 스키마v0.2에서 폐기된 필드
DEPRECATED_FIELDS = (
    "document_unspecified",
    "time_dependent",
    "conversational",
    "difficulty",
    "schema_version",
    "checkpoints",
    "unanswerable_reason",
)


def load_jsonl(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
