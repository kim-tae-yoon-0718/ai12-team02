"""평가셋 계약 검사 (2-17) — 임현진 소관, **팀 단일 출처**.

핵심 함수(check_schema / check_dup_ids / check_quota / check_ref_intg / check_version /
check_leak / run_check / main)는 origin/HJ `src/checks/check_evalset.py` 를 그대로 벤더링한다.
HJ 브랜치가 dev 에 머지되면 이 블록을 그 버전으로 갱신한다 — 구조를 맞춰 둬서 충돌 최소.

grader 쪽 부가분(명확히 분리):
  - load_jsonl : 멀티라인(pretty-print) JSON 도 로드 (임현진 practice_items.jsonl 형식)
  - check_excluded_as_gold : 1-9-1 수집중복 문서(RFP-000006/17)를 정답 근거로 쓰면 금지
  - scan_tracked_files : 【25】 문항 텍스트가 프롬프트·코드에 유출됐는지 (final 모드)
  - run_all : grader.runner 2층 / cli validate 가 부르는 오케스트레이터
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

# ══════════════════════════════════════════ 이하 origin/HJ 벤더링 (수정 최소) ══════

# status: "PASS" | "FAIL" | "SKIP"
CheckResult = namedtuple("CheckResult", ["name", "status", "message"])

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
    "answer_normalized": "any",
    "location": {"document", "section", "ref_no", "line"},  # metadata 문항은 line 면제(아래)
}
TASK_ANSWER_COMBOS = {
    "selection": ("document_set",),
    "extraction": ("value", "list"),
    "qa": ("value", "summary", "comparison", "unanswerable"),
}
FIELD_DEPENDENCIES = {
    "unspecified_type": ["intermediate_answer"],
    "scenario_type": ["active_document_id"],
}
DEPRECATED_FIELDS = (
    "document_unspecified",
    "time_dependent",
    "conversational",
    "difficulty",
    "schema_version",
    "checkpoints",
    "unanswerable_reason",
)


def check_schema(item: dict, line_num: int) -> list[str]:
    """C1: 스키마 준수. 결여·위반 사항 리스트(빈 리스트 = 통과)."""
    errors = []
    essential_fields = ["id", "question", "task_type", "answer_type", "answer_raw"]
    missing = [f for f in essential_fields if f not in item]
    if missing:
        errors.extend(f"line {line_num}: missing field '{f}'" for f in missing)

    for field_name, spec in FIELD_SPEC.items():
        if field_name not in item:
            continue
        value = item[field_name]

        if field_name == "reference_time":
            if value != REFERENCE_TIME:
                errors.append(f"line {line_num}: wrong timeset '{value}' time must be {REFERENCE_TIME}")
        elif spec == "string":
            if not isinstance(value, str) or value.strip() == "":
                errors.append(f"line {line_num}: type error '{value}'")
        elif spec == "any":
            pass
        elif isinstance(spec, tuple):
            if value not in spec:
                errors.append(f"line {line_num}: invalid '{value}' for field '{field_name}' (allowed: {spec})")
        elif isinstance(spec, list):
            allowed_types = tuple(TYPE_MAP[s] for s in spec)
            if not isinstance(value, allowed_types):
                errors.append(f"line {line_num}: type error '{value}' for field '{field_name}'")
        elif isinstance(spec, set):
            # 비교형(2-8-4): location 은 문서별 객체 배열. 단일 문항은 단일 객체.
            # metadata 문항(CSV 답변)은 본문 블록이 없어 line 을 면제한다(임현진 09-02).
            locations = value if isinstance(value, list) else [value]
            required_keys = (spec - {"line"}) if item.get("answer_source") == "metadata" else spec
            for loc in locations:
                if not isinstance(loc, dict):
                    errors.append(f"line {line_num}: type error '{loc}'")
                elif not required_keys.issubset(loc.keys()):
                    errors.append(f"line {line_num}: location missing key(s) '{required_keys - loc.keys()}'")

    for field_name in item:
        if field_name in DEPRECATED_FIELDS:
            errors.append(f"line {line_num}: deprecated field '{field_name}'")

    task_type = item.get("task_type")
    answer_type = item.get("answer_type")
    if task_type in TASK_ANSWER_COMBOS and answer_type not in TASK_ANSWER_COMBOS[task_type]:
        errors.append(f"line {line_num}: '{task_type}' cannot have answer_type '{answer_type}'")

    for trigger_field, required_field in FIELD_DEPENDENCIES.items():
        if trigger_field in item:
            missing_required = [r for r in required_field if r not in item]
            if missing_required:
                errors.append(
                    f"line {line_num}: '{trigger_field}' must activate by required_field '{missing_required}'")

    if item.get("task_type") == "selection" and "answer_source" not in item:
        errors.append(f"line {line_num}: task type 'selection' needs answer source")

    # 비교형(2-8-4) 정답 구조 — 팀장 2-6: 문서 ID 는 RFP-000000 형식만, 필드 키는 '항목' 확정
    if item.get("answer_type") == "comparison":
        rows = item.get("answer_normalized") or item.get("answer_raw")
        if not isinstance(rows, list):
            errors.append(f"line {line_num}: comparison answer must be a list of rows")
        else:
            for r_i, row in enumerate(rows):
                if not isinstance(row, dict):
                    errors.append(f"line {line_num}: comparison row {r_i} is not an object")
                    continue
                if not str(row.get("항목", "")).strip():
                    errors.append(f"line {line_num}: comparison row {r_i} missing '항목'")
                for k in row:
                    if k != "항목" and not re.match(r"RFP-\d{6}$", str(k).strip()):
                        errors.append(
                            f"line {line_num}: comparison row {r_i} unknown key '{k}' (RFP-000000 형식 아님)")

    for k, v in item.items():
        if v is None:
            errors.append(f"line {line_num}: invalid value(null)")

    return errors


def check_dup_ids(items: list[dict]) -> list[str]:
    """C2: 중복 ID."""
    errors = []
    id_counts = collections.Counter(item["id"] for item in items)
    for id_, c in id_counts.items():
        if c > 1:
            errors.append(f"C2: duplicate item_id: {id_} ({c}회)")
    return errors


def check_quota(items: list[dict], strict: bool = False) -> list[str]:
    """C3: 할당량. strict=True 면 총 50문항 (선별25:추출15:QA10) 정확히 대조."""
    errors = []
    EXPECTED = {"selection": 25, "extraction": 15, "qa": 10}
    tt_counts = collections.Counter(item["task_type"] for item in items)
    for tt in tt_counts:
        if tt not in EXPECTED:
            errors.append(f"C3: unknown task_type: {tt}")
    if strict:
        for tt, n in EXPECTED.items():
            got = tt_counts.get(tt, 0)
            if got != n:
                errors.append(f"C3: {tt} expected {n}, got {got}")
    return errors


def _locations_of(item: dict) -> list[dict]:
    loc = item.get("location")
    return loc if isinstance(loc, list) else [loc] if loc else []


def check_ref_intg(items: list[dict], doc_ids_path) -> list[str]:
    """C4: 참조 무결성 — location.document 가 corpus_doc_ids.json 에 실재하는지. 파일 없으면 SKIP."""
    errors = []
    if not doc_ids_path or not Path(doc_ids_path).exists():
        print("C4: SKIP (doc_ids not exist)")
        return errors
    valid_ids = set(json.loads(Path(doc_ids_path).read_text(encoding="utf-8")))
    for item in items:
        for loc in _locations_of(item):
            d = loc.get("document")
            if d is not None and d not in valid_ids:
                errors.append(f"C4: unknown document: {d} (item {item.get('id')})")
    return errors


def _read_version_key(path, key: str) -> str | None:
    """VERSION.txt 에서 `<key> : <value>` 한 줄을 읽는다."""
    if not path or not Path(path).exists():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() == key:
                return v.strip()
    return None


def check_version(evalset_version_path, corpus_version_path, chunking_version_path=None) -> list[str]:
    """C5: evalset VERSION.txt 의 corpus / chunking 값을 실제 코퍼스·청크 VERSION.txt 와 대조.
    각 값이 `[대기]` 이거나 파일이 없으면 SKIP (임현진 09-02: chunking 추가)."""
    errors = []
    if not evalset_version_path or not Path(evalset_version_path).exists():
        print("C5: SKIP (evalset VERSION.txt not exist)")
        return errors
    evalset_info = {}
    for line in Path(evalset_version_path).read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            evalset_info[key.strip()] = value.strip()

    for label, expected_key, actual_path, actual_key in (
        ("corpus", "corpus", corpus_version_path, "corpus version"),
        ("chunking", "chunking", chunking_version_path, "chunking version"),
    ):
        expected = evalset_info.get(expected_key)
        if expected in (None, "[대기]", "TBD", ""):
            print(f"C5: SKIP ({label} version pending)")
            continue
        if not actual_path or not Path(actual_path).exists():
            errors.append(f"C5: {label} VERSION.txt not exist: {actual_path}")
            continue
        actual = _read_version_key(actual_path, actual_key)
        if actual != expected:
            errors.append(
                f"C5: {label} version mismatch: evalset expects {expected}, {label} is {actual}")
    return errors


def check_leak(items: list[dict], practice_path) -> list[str]:
    """C6: 최종셋 유출 — PRAC- 접두어 / practice 세트와 id·문서 중복. practice 파일 없으면 접두어만."""
    errors = []
    for item in items:
        if str(item["id"]).startswith("PRAC-"):
            errors.append(f"C6: PRAC- item in final set: {item['id']}")

    if not practice_path or not Path(practice_path).exists():
        print("C6: SKIP practice cross-check (practice file not exist)")
        return errors

    practice_items = load_jsonl(practice_path)
    practice_ids = {p["id"] for p in practice_items}
    practice_docs = set()
    for p in practice_items:
        if "document_id" in p:
            practice_docs.add(p["document_id"])
        for loc in _locations_of(p):
            if loc.get("document"):
                practice_docs.add(loc["document"])

    for item in items:
        if item["id"] in practice_ids:
            errors.append(f"C6: duplicate id with practice: {item['id']}")
        for loc in _locations_of(item):
            d = loc.get("document")
            if d in practice_docs:
                errors.append(f"C6: practice document {d} used in final set (item {item['id']})")
    return errors


def run_check(items, args) -> list[CheckResult]:
    """origin/HJ 오케스트레이터 (argparse args 로 구동). CLI 전용."""
    results = []
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


# ══════════════════════════════════════════ grader 쪽 부가분 ══════════════════════

def load_jsonl(path) -> list[dict]:
    """평가셋 로드. 진짜 JSONL(줄마다 독립 JSON)이 정석이지만, 임현진
    `evalset/practice_items.jsonl` 처럼 **여러 줄에 걸친 pretty-print JSON 을 이어 붙인 파일**도
    받아 준다(확장자는 .jsonl 이나 내용은 concatenated JSON).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        return [json.loads(ln) for ln in lines]
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    records: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: JSON 파싱 실패 (offset {i}): {e}") from e
        records.append(obj)
        i = end
    return records


_DOC_ID_RE = re.compile(r"RFP-\d{4,}")
DEFAULT_EXCLUDED_DOC_IDS = ("RFP-000006", "RFP-000017")


def _gold_doc_ids(item: dict) -> list[str]:
    """문항이 정답 근거로 가리키는 문서 ID 전부 (location 배열·document_id·intermediate_answer·
    선별형 answer_raw 배열). 좌표는 이제 배열이므로 콤마 이어붙임 문자열은 안 나온다."""
    out: list[str] = []
    if item.get("answer_type") == "document_set" and isinstance(item.get("answer_raw"), list):
        out += [str(d) for d in item["answer_raw"]]
    if item.get("answer_type") == "comparison":  # 비교형 행의 RFP-* 키
        rows = item.get("answer_normalized") or item.get("answer_raw")
        for row in (rows if isinstance(rows, list) else []):
            if isinstance(row, dict):
                out += [k for k in row if k != "항목" and _DOC_ID_RE.match(str(k))]
    for key in ("document_id", "intermediate_answer", "active_document_id"):
        v = item.get(key)
        if isinstance(v, list):
            out += [str(d) for d in v]
        elif isinstance(v, str) and v:
            out += _DOC_ID_RE.findall(v) or [v]
    for loc in _locations_of(item):
        d = loc.get("document")
        if isinstance(d, str) and d:
            out += _DOC_ID_RE.findall(d) or [d]
    return out


def check_excluded_as_gold(items: list[dict], excluded_ids=None) -> list[str]:
    """1-9-1: 수집 중복 문서(RFP-000006/17 등, 검색 대상 아님)를 정답 근거로 쓰면 금지.
    corpus_doc_ids 에는 들어 있어서 C4 로는 안 잡힌다."""
    excluded = set(excluded_ids or DEFAULT_EXCLUDED_DOC_IDS)
    errors = []
    for item in items:
        hit = sorted(set(_gold_doc_ids(item)) & excluded)
        if hit:
            errors.append(f"1-9-1: {item.get('id')}: 정답 근거가 검색 대상 아닌 문서 {hit}")
    return errors


_EVALSET_DIRS = ("/tests/fixtures/", "/data/", "/examples/", "/evalset/")


def scan_tracked_files(items: list[dict], repo_root, min_len: int = 12, exclude_paths=()) -> list[str]:
    """【25】: 문항 텍스트가 추적 파일(프롬프트·코드)에 그대로 있는지 — few-shot 유출 = 부정행위.
    final / --leak-check 모드에서만."""
    import subprocess
    root = Path(repo_root)
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=10)
        files = [root / ln for ln in out.stdout.splitlines()] if out.returncode == 0 else []
    except Exception:
        files = []
    skip = {str(Path(p).resolve()) for p in exclude_paths}
    needles = {i.get("id"): i.get("question") for i in items
               if i.get("question") and len(i["question"]) >= min_len}
    errors = []
    for path in files:
        rp = str(path.resolve()).replace("\\", "/")
        if rp in skip or any(d in rp for d in _EVALSET_DIRS):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for qid, q in needles.items():
            if q in text:
                rel = path.relative_to(root) if root in path.parents else path
                errors.append(f"【25】 {qid}: 문항 텍스트가 추적 파일 {rel} 에 있음")
    return errors


def run_all(
    records: list[dict],
    *,
    doc_ids_path=None,
    excluded_ids=None,
    practice_path=None,
    evalset_version_path=None,
    corpus_version_path=None,
    chunking_version_path=None,
    strict: bool = False,
    final_set: bool = False,
    leak_repo_root=None,
    leak_exclude=(),
) -> list[str]:
    """grader.runner 2층 / cli validate 오케스트레이터. 값싼 것부터.

    final_set=True 일 때만 C6(PRAC 접두어·practice 교집합)를 켠다 — practice 세트 자체를
    검사할 땐 전부 오탐이라 끈다.
    """
    problems: list[str] = []
    for i, item in enumerate(records, start=1):
        problems += check_schema(item, i)
    problems += check_dup_ids(records)
    problems += check_quota(records, strict=strict)
    if doc_ids_path:
        problems += check_ref_intg(records, doc_ids_path)
    problems += check_excluded_as_gold(records, excluded_ids)
    if evalset_version_path:
        problems += check_version(evalset_version_path, corpus_version_path, chunking_version_path)
    if final_set:
        problems += check_leak(records, practice_path)
    if leak_repo_root:
        problems += scan_tracked_files(records, leak_repo_root, exclude_paths=leak_exclude)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="평가셋 계약 검사 (C0-C6, 2-17)")
    parser.add_argument("items_path", help="검사할 items.jsonl 경로")
    parser.add_argument("--strict", action="store_true", help="freeze용: 총량·비율 하드검사")
    parser.add_argument("--final-set", action="store_true", help="검사 대상이 최종셋 — C6 유출 검사 켬")
    parser.add_argument("--practice", default="/srv/rfp/evalset/practice_items.jsonl")
    parser.add_argument("--doc-ids", default="data/gold/corpus_doc_ids.json")
    parser.add_argument("--evalset-version", default="/srv/rfp/evalset/v1/VERSION.txt")
    parser.add_argument("--corpus-version",
                        default="/srv/rfp/shared_data/processed/corpus_v2/VERSION.txt")
    parser.add_argument("--chunking-version",
                        default="/srv/rfp/shared_data/processed/chunks_v3/VERSION.txt")
    parser.add_argument("--leak-scan-root", default=None, help="【25】 추적 파일 유출 스캔 루트")
    args = parser.parse_args(argv)

    try:
        items = load_jsonl(args.items_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"C0 FAIL: {e}")
        return 1
    print(f"C0 PASS: {len(items)} items loaded")

    problems = run_all(
        items,
        doc_ids_path=args.doc_ids,
        practice_path=args.practice,
        evalset_version_path=args.evalset_version,
        corpus_version_path=args.corpus_version,
        chunking_version_path=args.chunking_version,
        strict=args.strict,
        final_set=args.final_set,
        leak_repo_root=args.leak_scan_root,
        leak_exclude=[args.items_path],
    )
    for p in problems:
        print(f"  {p}")
    print(f"\n{'FAIL' if problems else 'PASS'} — {len(items)} items, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
