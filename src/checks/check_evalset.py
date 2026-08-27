"""
평가셋(evalset) 검사 3종 — ①스키마 ②참조 무결성 ③최종셋 유출
스키마: docs/schema_v0.1.md 기준
실행 코드: python3 check_evalset.py
실패 시 종료 코드 1 (check.sh 결합용)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

RAG_ROOT = Path(os.environ.get("RAG_ROOT", "/srv/rfp"))
EVALSET_PATH = RAG_ROOT / "evalset" / "v1" / "items.jsonl"
CORPUS_DIR = RAG_ROOT / "shared_data" / "processed" / "corpus_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "v0.1"
REFERENCE_TIME = "2024-06-01"
TASK_TYPES = {"selection", "extraction", "qa"}
ANSWER_TYPES = {"short_answer", "selection", "summary", "list", "comparison"}
UNSPECIFIED_TYPES = {"abbreviation", "org_only", "time_reference", "ambiguous_match"}
SCENARIO_TYPES = {"workflow_chain", "anaphora", "condition_add", "selection_to_extraction"}
FIELD_TAGS = {"critical", "major", "minor"}
ANSWER_SOURCES = {"table", "verified", "metadata"}
REQUIRED_FIELDS = (
    "id", "question", "task_type", "answer_type",
    "document_unspecified", "conversational", "time_dependent", "schema_version",
)


def load_items(path: Path) -> list[dict]:
    """jsonl 한 줄 = 문항 하나. 파싱 실패는 스키마 오류로 취급.(스키마 검사)"""
    items = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            items.append({"_line": i, "_parse_error": str(e)})
    return items


def check_schema(items: list[dict]) -> list[str]:
    """① 문항 JSON이 스키마 v0.1을 준수하는지."""
    errors = []
    seen_ids = set()
    for i, it in enumerate(items, start=1):
        tag = it.get("id", f"line{i}")

        if "_parse_error" in it:
            errors.append(f"[{tag}] JSON 파싱 실패: {it['_parse_error']}")
            continue

        # [id, question, task_type, answer_type, document_unspecified, conversational, time_dependent, schema_version]
        for f in REQUIRED_FIELDS:
            if f not in it:
                errors.append(f"[{tag}] 필수 필드 누락: {f}")
        if any(f not in it for f in REQUIRED_FIELDS):
            continue  # 필수 필드가 없으면 아래 검사는 의미 없음

        # 문항ID(=질문ID)의 중복 확인
        if it["id"] in seen_ids:
            errors.append(f"[{tag}] id 중복")
        seen_ids.add(it["id"])

        if it["schema_version"] != SCHEMA_VERSION:
            errors.append(f"[{tag}] schema_version이 {SCHEMA_VERSION}이 아님: {it['schema_version']}")

        # 질문 유형
        if it["task_type"] not in TASK_TYPES:
            errors.append(f"[{tag}] task_type 값 이상: {it['task_type']}")

        # 답변 유형
        if it["answer_type"] not in ANSWER_TYPES:
            errors.append(f"[{tag}] answer_type 값 이상: {it['answer_type']}")

        # 문서 미특정 여부(boolean)
        if it["document_unspecified"]:
            # 미특정 유형
            if it.get("unspecified_type") not in UNSPECIFIED_TYPES:
                errors.append(f"[{tag}] document_unspecified=true인데 unspecified_type 값 이상: {it.get('unspecified_type')}")
            # 중간 정답(=<중간 문서 ID들>)
            if "intermediate_answer" not in it:
                errors.append(f"[{tag}] document_unspecified=true인데 intermediate_answer 없음")
        elif it.get("unspecified_type") is not None:
            errors.append(f"[{tag}] document_unspecified=false인데 unspecified_type이 있음")

        # 후속질문여부(boolean)
        if it["conversational"]:
            # 후속질문에 의해 활성화 된 문서ID
            if "active_document_id" not in it:
                errors.append(f"[{tag}] conversational=true인데 active_document_id 없음")
            # 시나리오 유형
            if it.get("scenario_type") not in SCENARIO_TYPES:
                errors.append(f"[{tag}] conversational=true인데 scenario_type 값 이상: {it.get('scenario_type')}")
        elif it.get("scenario_type") is not None:
            errors.append(f"[{tag}] conversational=false인데 scenario_type이 있음")

        # 시간의존여부(boolean)
        if it["time_dependent"] and it.get("reference_time") != REFERENCE_TIME:
            errors.append(f"[{tag}] time_dependent=true인데 reference_time이 {REFERENCE_TIME}이 아님: {it.get('reference_time')}")

        # 항목 등급(심각도)
        field_tag = it.get("field_tag")
        if field_tag is not None and field_tag not in FIELD_TAGS:
            errors.append(f"[{tag}] field_tag 값 이상: {field_tag}")

        # 답변 유형 = 선별형일 떄, 정답 출처
        if it["answer_type"] == "selection":
            answer_source = it.get("answer_source")
            if answer_source is not None and answer_source not in ANSWER_SOURCES:
                errors.append(f"[{tag}] answer_source 값 이상: {answer_source}")

        # 근거 좌표(문서+페이지+섹션)
        location = it.get("location")
        if location is not None and not {"document", "section", "ref_no"} <= location.keys():
            errors.append(f"[{tag}] location에 document/section/ref_no 중 누락된 키 있음: {location}")

    return errors


def check_references(items: list[dict]) -> list[str]:
    """② document_id·location.document가 코퍼스에 실재하는 문서인지.
        답변에 근거가 되는 데이터(document_id, active_document_id, intermediate_answer)를 모아서 정리.(참조무결성 검사)"""
    if not CORPUS_DIR.exists():
        print(f"[참조무결성] {CORPUS_DIR} 없음 — 1-19-1 도착 전까지 비활성", file=sys.stderr)
        return []

    valid_ids = {p.stem for p in CORPUS_DIR.iterdir() if p.is_file()}
    errors = []
    for it in items:
        if "_parse_error" in it:
            continue
        tag = it.get("id", "?")
        refs = list(it.get("document_id") or [])
        refs += [it.get("active_document_id")]
        refs += list(it.get("intermediate_answer") or [])
        location = it.get("location")
        if location:
            refs.append(location.get("document"))
        for doc_id in refs:
            if doc_id is not None and doc_id not in valid_ids:
                errors.append(f"[{tag}] 코퍼스에 없는 document_id: {doc_id}")
    return errors


def check_no_leakage(items: list[dict]) -> list[str]:
    """③ 평가 문항이 프롬프트 파일 등 다른 개발 파일에 유출됐는지.(유출 검사)"""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    needles = [it["question"] for it in items if "_parse_error" not in it and it.get("question")]

    errors = []
    for rel_path in tracked:
        path = REPO_ROOT / rel_path
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for q in needles:
            if q in text:
                errors.append(f"[{rel_path}] 평가 문항 텍스트 유출: {q!r}")
    return errors


def main() -> None:
    if not EVALSET_PATH.exists():
        print(f"평가셋 파일 없음: {EVALSET_PATH}", file=sys.stderr)
        sys.exit(1)

    items = load_items(EVALSET_PATH)
    checks = [
        ("스키마", check_schema(items)),
        ("참조무결성", check_references(items)),
        ("최종셋유출", check_no_leakage(items)),
    ]

    ok = True
    for name, errors in checks:
        if errors:
            ok = False
            print(f"--- {name} 실패 ({len(errors)}건) ---")
            for e in errors:
                print(e)
        else:
            print(f"{name}: OK")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
