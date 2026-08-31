import json
from pathlib import Path 
from typing import List
from collections import namedtuple


# status: "PASS" | "FAIL" | "SKIP"
CheckResult = namedtuple("CheckResult", ["name", "status", "message"])

# reference_time은 정해진 상수값
REFERENCE_TIME = "2024-06-01"
TYPE_MAP = {"string": str, "array": list}
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


def load_jsonl(path: Path) -> List[dict]:
    """ 
    데이터를 jsonl형식으로 다운로드
    jsonl: 독립적으로 연결된 json (대괄호도 콤마도 없음) 
    {"id": "SEL-001", ...} {"id": "SEL-002", ...}
    """
    jsonl_data = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "r", encoding="utf-8") as json_file:
        errors = []
        for line_num, line in enumerate(json_file, start=1):
            line = line.strip()
            # 스펙 공백 건너뜀
            if not line:
                continue
            try:
                jsonl_data.append(json.loads(line))
            # 파싱 실패 시
            except json.JSONDecodeError as e:
                errors.append(f"line {line_num}: {e}")
        
        if errors:
            raise ValueError("\n".join(errors))
    return jsonl_data


def check_schema(item: dict, line_num: int) -> List[str]:
    """
    문항 한 줄(jsonl: item)을 받아 스키마가 결여된 사항을 리스트로 반환
    결여 사항이 없을 경우 -> 빈 리스트
    """
    errors = []
    # TODO 1: 공통 필수 5개 필드 존재 확인
    essential_fields = ["id", "question", "task_type", "answer_type", "answer_raw"]
    missing = [f for f in essential_fields if f not in item]
    if missing:
        errors.extend(f"line {line_num}: missing field '{f}'" for f in missing)

    # TODO 2: FIELD_SPEC에 정의된 각 필드의 타입/enum 검증
    for field_name, spec in FIELD_SPEC.items():
        if field_name not in item:
            continue

        value = item[field_name]

        if spec == "string":
            if not isinstance(value, str) or value.strip() == "":
                errors.append(f"line {line_num}: type error '{value}'")

        elif isinstance(spec, tuple):
            if value not in spec:
                errors.append(f"line {line_num}: invalid '{value}' for field '{field_name}' (allowed: {spec})")

        elif isinstance(spec, list):
            allowed_types = [TYPE_MAP[s] for s in spec]
            if not isinstance(value, tuple(allowed_types)):
                errors.append(f"line {line_num}: type error '{value}' for field '{field_name}'")

        elif field_name == "reference_time":
            if value != REFERENCE_TIME:
                errors.append(f"line {line_num}: wrong timeset '{value}' time must be {REFERENCE_TIME}")

        elif isinstance(spec, set):
            if not isinstance(value, dict):
                errors.append(f"line {line_num}: type error '{value}'")
            else: 
                if not spec.issubset(value.keys()):
                    missing_keys = spec - value.keys()
                    errors.append(f"line {line_num}: location missing key(s) '{missing_keys}'")

    # TODO 3: DEPRECATED_FIELDS에 있는 키가 item에 남아있는지 확인
    for field_name in item:
        if field_name in DEPRECATED_FIELDS:
            errors.append(f"line {line_num}: deprecated field '{field_name}'")

    # TODO 4: task_type x answer_type 조합 검증
    task_type = item.get("task_type")
    answer_type = item.get("answer_type")

    if task_type in TASK_ANSWER_COMBOS:
        if answer_type not in TASK_ANSWER_COMBOS[task_type]:
            errors.append(f"line {line_num}: '{task_type}' cannot have answer_type '{answer_type}'")

    # TODO 5: 문서 미특정 질문, 시나리오형 질문에서 필수 필드 조합 검증(FIELD_DEPENDENCIES)
    for trigger_field, required_field in FIELD_DEPENDENCIES.items():
        if trigger_field in item:
            missing_required = [r for r in required_field  if r not in item]
            if missing_required:
                errors.append(f"line {line_num}: '{trigger_field}' must activate by required_field '{missing_required}'")

    # TODO 6: task_type = "selection" x answer_source 조합 검증
    if item.get("task_type") == "selection":
        if "answer_source" not in item:
            errors.append(f"line {line_num}: task type 'selection' needs answer source")

    # TODO 7: None(null)값인 필드 검증
    for k, v in item.items():
        if v is None:
            errors.append(f"line {line_num}: invalid value(null)")

    return errors


# 최종 사용 방법
items = load_jsonl("evalset/practice_items.jsonl")
for i, item in enumerate(items, start=1):
    errs = check_schema(item, i)
    if errs:
        print(errs)