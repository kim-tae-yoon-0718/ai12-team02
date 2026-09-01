import json
from pathlib import Path 
from typing import List
import collections
from collections import namedtuple, Counter


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
    C1: 스키마 준수
    문항 한 줄(jsonl: item)을 받아 스키마가 결여된 사항을 리스트로 반환
    결여 사항이 없을 경우 -> 빈 리스트
    """
    errors = []
    # 1: 공통 필수 5개 필드 존재 확인
    essential_fields = ["id", "question", "task_type", "answer_type", "answer_raw"]
    missing = [f for f in essential_fields if f not in item]
    if missing:
        errors.extend(f"line {line_num}: missing field '{f}'" for f in missing)

    # 2: FIELD_SPEC에 정의된 각 필드의 타입/enum 검증
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

    # 3: DEPRECATED_FIELDS에 있는 키가 item에 남아있는지 확인
    for field_name in item:
        if field_name in DEPRECATED_FIELDS:
            errors.append(f"line {line_num}: deprecated field '{field_name}'")

    # 4: task_type x answer_type 조합 검증
    task_type = item.get("task_type")
    answer_type = item.get("answer_type")

    if task_type in TASK_ANSWER_COMBOS:
        if answer_type not in TASK_ANSWER_COMBOS[task_type]:
            errors.append(f"line {line_num}: '{task_type}' cannot have answer_type '{answer_type}'")

    # 5: 문서 미특정 질문, 시나리오형 질문에서 필수 필드 조합 검증(FIELD_DEPENDENCIES)
    for trigger_field, required_field in FIELD_DEPENDENCIES.items():
        if trigger_field in item:
            missing_required = [r for r in required_field  if r not in item]
            if missing_required:
                errors.append(f"line {line_num}: '{trigger_field}' must activate by required_field '{missing_required}'")

    # 6: task_type = "selection" x answer_source 조합 검증
    if item.get("task_type") == "selection":
        if "answer_source" not in item:
            errors.append(f"line {line_num}: task type 'selection' needs answer source")

    # 7: None(null)값인 필드 검증
    for k, v in item.items():
        if v is None:
            errors.append(f"line {line_num}: invalid value(null)")

    return errors


def check_dup_ids(items: list[dict]) -> List[str]:
    """
    C2: 중복 ID
    item_id 중복 검출
    error 메시지 반환 (빈 리스트 = 통과)
    """
    errors = []
    id_counts = collections.Counter(item["id"] for item in items)
    for id, c in id_counts.items():
        if c > 1:
            errors.append(f"C2: duplicate item_id: {id} ({c}회)")

    return errors


def check_quota(items: list[dict], strict: bool = False) -> List[str]:
    """
    C3: 할당량
    task_type별 문항 수 검사
    검사 완화(strict = False): 비율·형식만 경고 수준으로 확인 - 문항 50미만이어도 동작
    검사 강화(strict = True): 총 50문항 (선별25:추출15:QA10) 정확히 일치
    """
    errors = []
    EXPECTED = {"selection": 25, "extraction": 15, "qa": 10}
    tt_counts = collections.Counter(item["task_type"] for item in items)

    # 공통 검사: EXPECTED에 없는 task_type 검출
    for tt in tt_counts:
        if tt not in EXPECTED:
            errors.append(f"C3: unknown task_type: {tt}")

    # strict 적용: 개수 대조
    if strict:
        for tt, n in EXPECTED.items():
            got = tt_counts.get(tt, 0)
            if got != n:
                errors.append(f"C3: {tt} expected {n}, got {got}")

    return errors


def check_ref_intg(items: list[dict], doc_ids_path) -> List[str]:
    """
    C4: 참조 무결성
    location 및 정답 문서 ID 배열이 실재로 포함(corpus_doc_ids.json)되어 있는지 검사
    doc_ids_path 파일 없으면 SKIP
    """
    errors = []
    if not Path(doc_ids_path).exists():
        raise FileNotFoundError(doc_ids_path)
    
    data = json.loads(Path(doc_ids_path).read_text(encoding="utf-8"))
    valid_ids = set(data)

    for item in items:
        doc = item.get("location", {}).get("document")
        if doc is not None and doc not in valid_ids:
            errors.append(f"C4: 없는 문서: {doc} (item {item.get('item_id')})")
        for d in item.get("정답배열필드명", []):
            if d not in valid_ids:
                errors.append(f"C4: 없는 문서: {d} (item {item.get('item_id')})")
    
    return errors


def check_version(version_txt_path) -> List[str]:
    """
    C5: 코퍼스 버전 일치
    VERSION.txt의 corpus 값과 실제 코퍼스 버전 대조
    corpus: [대기] 동안은 SKIP
    """
    errors = []
    # TODO 1: VERSION.txt 파싱 (key: value 3줄, bare 표기)
    with open("VERSION.txt", "r", encoding="utf-8") as file:
        for line in file:
            clean_line = line.strip()
    # TODO 2: corpus 값이 "[대기]"면 print("C5: SKIP (...)") 후 반환
    # TODO 3: 활성화 시 대조 로직 — 지금은 pass로 두고 1-19 도착 후 채움
    return errors


def check_leak(items: list[dict], practice_path) -> List[str]:
    """
    C6: 최종셋 유출 방지
    items.jsonl과 practice_items.jsonl 간의 중복 검사
    """
    errors = []
    # TODO 1: 최종셋 item_id에 "PRAC-" 접두어가 섞여 있으면 에러
    # TODO 2: practice_path 로드 — practice 쪽 item_id와 최종셋 item_id 교집합 검사
    # TODO 3: 최종셋이 practice 전용 3개 문서(RFP-000038/000043/000001)를
    #         근거 문서로 쓰면 에러 (2-13: 최종 50문항 제작 제외 문서)
    return errors


def main():
    # TODO 1: argparse — 위치인자 items_path, 옵션 --strict, --practice, --doc-ids, --version-txt
    #   경로 기본값 하드코딩 금지(팀 규약) — 전부 인자나 환경변수($RAG_ROOT)로
    # TODO 2: load_jsonl() → C0/C1(기존 check_schema) → C2 → C3 → C4 → C5 → C6
    #   순서 근거: 값싼 검사부터 (2-17)
    # TODO 3: errors 전부 출력 후, 하나라도 있으면 sys.exit(1) — check.sh가 이 종료코드로 실험을 멈춤
    pass





if __name__ == "__main__":
    main()



# 최종 사용 방법
items = load_jsonl("evalset/practice_items.jsonl")
for i, item in enumerate(items, start=1):
    errs = check_schema(item, i)
    if errs:
        print(errs)