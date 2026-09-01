import json
import sys
import argparse
from pathlib import Path 
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


def load_jsonl(path: Path) -> list[dict]:
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


def check_schema(item: dict, line_num: int) -> list[str]:
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


def check_dup_ids(items: list[dict]) -> list[str]:
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


def check_quota(items: list[dict], strict: bool = False) -> list[str]:
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


def check_ref_intg(items: list[dict], doc_ids_path) -> list[str]:
    """
    C4: 참조 무결성
    location 및 정답 문서 ID 배열이 실재로 포함(corpus_doc_ids.json)되어 있는지 검사
    doc_ids_path 파일 없으면 SKIP
    """
    errors = []
    if not Path(doc_ids_path).exists():
        print("C4: SKIP (doc_ids not exist)")
        return errors
    
    data = json.loads(Path(doc_ids_path).read_text(encoding="utf-8"))
    valid_ids = set(data)

    for item in items:
        doc = item.get("location", {}).get("document")
        if doc is not None:
            for d in doc.split(", "):
                if d not in valid_ids:
                    errors.append(f"C4: unknown document: {d} (item {item.get('id')})")
    
    return errors


def check_version(evalset_version_path, corpus_version_path) -> list[str]:
    """
    C5: 코퍼스 버전 일치
    evalset VERSION.txt의 corpus 값과 실제 corpus_version 값 대조
    corpus: [대기] 동안은 SKIP
    """
    errors = []
    if not Path(evalset_version_path).exists():
        errors.append(f"C5 VERSION.txt not exist: {evalset_version_path}")
        return errors

    evalset_info = {}
    with open(evalset_version_path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:                        # 빈 줄 예방
                key, value = line.split(":", 1)    # ":" 기준 1번만
                evalset_info[key.strip()] = value.strip()

    corpus_expected = evalset_info.get("corpus")
    if corpus_expected == "[대기]":
        print("C5: SKIP (corpus version not exist)")
        return errors
    
    if not Path(corpus_version_path).exists():
        errors.append(f"C5: corpus VERSION.txt not exist: {corpus_version_path}")
        return errors

    corpus_actual = None
    with open(corpus_version_path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() == "corpus version":
                    corpus_actual = value.strip()

    if corpus_actual != corpus_expected:
        errors.append(
            f"C5: corpus version mismatch: evalset expects {corpus_expected}, corpus is {corpus_actual}"
        )

    return errors


def check_leak(items: list[dict], practice_path) -> list[str]:
    """
    C6: 최종셋 유출 방지
    items.jsonl과 practice_items.jsonl 간의 중복 검사
    """
    errors = []
    for item in items:
        if item["id"].startswith("PRAC-"):
            errors.append(f"C6: PRAC- item in final set: {item['id']}")

    if not Path(practice_path).exists():
        print("C6: SKIP (practice file not exist)")
        return errors

    practice_items = load_jsonl(practice_path)
    practice_ids = {p["id"] for p in practice_items}
    practice_docs = set()
    for p in practice_items:
        if "document_id" in p:
            practice_docs.add(p["document_id"])
        doc = p.get("location", {}).get("document")
        if doc:
            practice_docs.update(doc.split(", "))
    
    # practice와 최종셋 교집합 검사
    for item in items:
        if item["id"] in practice_ids:
            errors.append(f"C6: duplicate id with practice: {item['id']}")
        doc = item.get("location", {}).get("document")
        if doc:
            for d in doc.split(", "):
                if d in practice_docs:
                    errors.append(f"C6: practice document {d} used in final set (item {item['id']})")

    return errors


def run_check(items, args) -> list[CheckResult]:
    """
    각 check를 CheckResult로 묶어서 반환
    SKIP은 각 함수의 print가 대체, run_check에서는 PASS/FAIL만 분류
    """
    results =[]
    schema_errors = []
    for line_num, item in enumerate(items, start=1):
        schema_errors.extend(check_schema(item, line_num))
    results.append(CheckResult("C1 schema", "FAIL" if schema_errors else "PASS", schema_errors))

    checks = [
        ("C2 duplicate_ids", check_dup_ids(items)),
        ("C3 quota", check_quota(items, strict=args.strict)),
        ("C4 ref_integrity", check_ref_intg(items, args.doc_ids)),
        ("C5 version", check_version(args.evalset_version, args.corpus_version)),
        ("C6 leakage", check_leak(items, args.practice)),
    ]
    for name, errors in checks:
        results.append(CheckResult(name, "FAIL" if errors else "PASS", errors))

    return results


def main():
    parser = argparse.ArgumentParser(description="평가셋 검사(C0-C6")
    parser.add_argument("items_path", help="검사항 items.jsonl 경로")
    parser.add_argument("--strict", action="store_true", help="freeze용: 총량·비율 하드검사")
    parser.add_argument("--practice", default="/srv/rfp/evalset/practice_items.jsonl")
    parser.add_argument("--doc-ids", default="data/gold/corpus_doc_ids.json")
    parser.add_argument("--evalset-version", default="/srv/rfp/evalset/v1/VERSION.txt")
    parser.add_argument("--corpus-version", default="/srv/rfp/shared_data/processed/corpus_v2/VERSION.txt")
    args = parser.parse_args()

    # C0 load_jsonl 검사
    try:
        items = load_jsonl(args.items_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"C0 FAIL: {e}")
        sys.exit(1)
    print(f"C0 PASS: {len(items)} items loaded")

    results = run_check(items, args)

    failed = False
    for r in results:
        print(f"{r.name}: {r.status}")
        for msg in r.message:
            print(f" {msg}")
        if r.status == "FAIL":
            failed = True

    sys.exit(1 if failed else 0)






if __name__ == "__main__":
    main()