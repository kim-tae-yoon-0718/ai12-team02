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
import csv
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

# ══════════════════════════════════════════ 이하 origin/HJ 벤더링 (수정 최소) ══════

# status: "PASS" | "FAIL" | "SKIP"
CheckResult = namedtuple("CheckResult", ["name", "status", "message"])

# origin/HJ b8a0656(dev 병합분) 반영: 기준 경로를 절대경로로 — cwd 가 저장소 루트가
# 아닌 곳에서 CLI 를 실행해도 --doc-ids 기본값이 깨지지 않는다.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_IDS = REPO_ROOT / "data" / "gold" / "corpus_doc_ids.json"

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



# ---------------------------------------------------------------------------
# 【강화】 최종 평가셋 계약 검사 (2026-09-04)
# ---------------------------------------------------------------------------
# 기존 검사는 문법 중심이라 "정답이 틀린 평가셋"도 통과시켰다. 아래는 공식 자료를
# 직접 열어 정답·근거가 실제로 존재하는지 확인한다.
# ⚠️ 최종 모드(final_mode=True)에서는 필요한 공식 자료가 없으면 SKIP 하지 않고 실패한다.

REFERENCE_TIME_EXPECTED = "2024-06-01"

OFFICIAL_REGISTRY = "/srv/rfp/shared_data/processed/document_registry_v2/document_registry_v2.json"
OFFICIAL_IDENTITY = "/srv/rfp/shared_data/processed/document_registry_v2/document_identity_v2.csv"
OFFICIAL_TABLE = "/srv/rfp/shared_data/processed/rfp_extraction_table_v4/extraction_table_v4.json"
OFFICIAL_CHUNKS = "/srv/rfp/shared_data/processed/chunks_v3/chunks.jsonl"


class AssetMissing(RuntimeError):
    """최종 모드에서 필요한 공식 자료가 없을 때 — 조용한 SKIP 금지."""


def _load_registry_scope(registry_path):
    """(전체 문서 ID, 검색 대상 문서 ID) — active=true AND retrieval_eligible=true."""
    path = Path(registry_path)
    if not path.exists():
        raise AssetMissing(f"등록부를 찾지 못했습니다: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    docs = doc["documents"] if isinstance(doc, dict) else doc
    every = {d["document_id"] for d in docs}
    eligible = {d["document_id"] for d in docs
                if d.get("retrieval_eligible") is True
                and str(d.get("active", "true")).lower() == "true"}
    return every, eligible


def _item_doc_ids(item: dict) -> set:
    """문항이 참조하는 모든 문서 ID(정답·문서지정·중간답·근거 좌표)."""
    found = set()
    for value in (item.get("answer_raw"), item.get("intermediate_answer")):
        if isinstance(value, list):
            found |= {v for v in value if isinstance(v, str) and _DOC_ID_RE.fullmatch(v)}
        elif isinstance(value, str):
            found |= set(_DOC_ID_RE.findall(value))
    for key in ("document_id", "active_document_id"):
        v = item.get(key)
        if isinstance(v, str) and _DOC_ID_RE.fullmatch(v):
            found.add(v)
    for loc in _locations_of(item):
        d = loc.get("document")
        if isinstance(d, str) and _DOC_ID_RE.fullmatch(d):
            found.add(d)
    return found


def check_selection_scope(items: list[dict], registry_path=OFFICIAL_REGISTRY) -> list[str]:
    """【S1】 선별형 정답 문서는 등록부의 검색 대상 안에만 있어야 한다."""
    every, eligible = _load_registry_scope(registry_path)
    errors = []
    for item in items:
        qid = item.get("id")
        ids = _item_doc_ids(item)
        unknown = sorted(ids - every)
        if unknown:
            errors.append(f"【S1】 {qid}: 등록부에 없는 문서 {unknown}")
        if item.get("task_type") == "selection":
            outside = sorted(ids - eligible)
            if outside:
                errors.append(f"【S1】 {qid}: 검색 대상({len(eligible)}문서) 밖 문서가 정답에 있음 {outside}")
            gold = item.get("answer_raw")
            if not isinstance(gold, list):
                errors.append(
                    f"【S1】 {qid}: 선별형 정답(answer_raw)은 문서 ID 리스트여야 함 "
                    f"— 현재 {type(gold).__name__}. 문자열이면 문서 1건이 조용히 통과한다.")
            aset = item.get("answer_set")
            if aset is not None and not isinstance(aset, list):
                errors.append(f"【S1】 {qid}: answer_set 도 리스트여야 함 — 현재 {type(aset).__name__}")
            if isinstance(gold, list) and len(gold) != len(set(gold)):
                dup = sorted({d for d in gold if gold.count(d) > 1})
                errors.append(f"【S1】 {qid}: 정답 문서 ID 중복 {dup}")
    return errors


def check_reference_time(items: list[dict], expected: str = REFERENCE_TIME_EXPECTED) -> list[str]:
    """【S2】 마감 필터가 걸리는 선별형은 기준 시각을 기록해야 한다."""
    errors = []
    for item in items:
        if item.get("task_type") != "selection":
            continue
        got = item.get("reference_time")
        if got is None:
            errors.append(f"【S2】 {item.get('id')}: 선별형인데 reference_time 이 없음")
        elif got != expected:
            errors.append(f"【S2】 {item.get('id')}: reference_time={got!r} (기대 {expected!r})")
    return errors


def check_answer_evidence_docs(items: list[dict]) -> list[str]:
    """【S3】 근거 좌표의 문서가 정답이 가리키는 문서와 어긋나면 안 된다."""
    errors = []
    for item in items:
        qid = item.get("id")
        answer_docs = set()
        if isinstance(item.get("answer_raw"), list):
            answer_docs |= {v for v in item["answer_raw"]
                            if isinstance(v, str) and _DOC_ID_RE.fullmatch(v)}
        for key in ("document_id", "active_document_id"):
            v = item.get(key)
            if isinstance(v, str) and _DOC_ID_RE.fullmatch(v):
                answer_docs.add(v)
        if isinstance(item.get("intermediate_answer"), list):
            answer_docs |= {v for v in item["intermediate_answer"]
                            if isinstance(v, str) and _DOC_ID_RE.fullmatch(v)}
        if not answer_docs:
            continue
        for loc in _locations_of(item):
            d = loc.get("document")
            if d and d not in answer_docs:
                errors.append(f"【S3】 {qid}: 근거 문서 {d} 가 정답 문서 {sorted(answer_docs)} 에 없음")
    return errors


def check_comparison_structure(items: list[dict]) -> list[str]:
    """【S4】 비교형 정답은 문서마다 값을 갖는 구조여야 한다."""
    errors = []
    for item in items:
        # ★평가셋의 실제 값은 "comparison" 이다. 예전 표기("comparison_table")로만
        #   걸러서 이 검사가 한 번도 실행되지 않았다(합성 결함 D11 이 통과해 발각).
        if item.get("answer_type") not in ("comparison", "comparison_table"):
            continue
        qid = item.get("id")
        ans = item.get("answer_raw")
        if not isinstance(ans, (dict, list, str)):
            errors.append(f"【S4】 {qid}: 비교형 answer_raw 자료형이 {type(ans).__name__}")
            continue
        docs = _item_doc_ids(item)
        if len(docs) < 2:
            errors.append(f"【S4】 {qid}: 비교형인데 문서가 {len(docs)}개")
    return errors


def _chunk_index(chunks_path):
    path = Path(chunks_path)
    if not path.exists():
        raise AssetMissing(f"chunks_v3 를 찾지 못했습니다: {path}")
    index = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            index[(c["document_id"], c.get("location_label"))] = c
    return index


def check_locations_exist(items: list[dict], chunks_path=OFFICIAL_CHUNKS,
                          identity_path=OFFICIAL_IDENTITY) -> list[str]:
    """【S5】 근거 좌표가 실제 공식 자료에 존재하고 line 이 범위 안이어야 한다."""
    import csv as _csv
    ipath = Path(identity_path)
    if not ipath.exists():
        raise AssetMissing(f"identity_v2 를 찾지 못했습니다: {ipath}")
    identity = {r["document_id"]: r for r in _csv.DictReader(ipath.open(encoding="utf-8-sig"))}
    chunks = _chunk_index(chunks_path)
    errors = []
    for item in items:
        qid = item.get("id")
        for loc in _locations_of(item):
            doc, ref, line = loc.get("document"), loc.get("ref_no"), loc.get("line")
            if str(ref).startswith("CSV:"):
                col = str(ref).split(":", 1)[1].strip()
                if doc not in identity or col not in identity[doc]:
                    errors.append(f"【S5】 {qid}: identity_v2 에 {doc}/{col} 없음")
                continue
            chunk = chunks.get((doc, ref))
            if chunk is None:
                errors.append(f"【S5】 {qid}: chunks_v3 에 좌표 없음 (document={doc}, ref_no={ref!r})")
                continue
            start, end = chunk.get("md_line_start"), chunk.get("md_line_end")
            if line is not None and start is not None and not (start <= line <= (end or start)):
                errors.append(f"【S5】 {qid}: line={line} 이 청크 범위 {start}-{end} 밖")
    return errors


def check_answer_source_present(items: list[dict]) -> list[str]:
    """【S6】 모든 문항이 정답 출처 라벨을 가져야 한다(출처 종류와 좌표 대응 확인용)."""
    return [f"【S6】 {i.get('id')}: answer_source 없음"
            for i in items if "answer_source" not in i]


# ─────────────────────────────────────── S7~S12 (2026-09-04 강화)

# 사실을 묻는 문항 — 근거 없이 정답만 있으면 안 된다. 기권형은 제외한다.
_FACTUAL_ANSWER_TYPES = ("value", "list", "summary", "comparison", "document_set")


def _evidence_of(item: dict) -> list:
    ev = item.get("evidence")
    if isinstance(ev, list) and ev:
        return ev
    loc = item.get("location")
    if isinstance(loc, dict):
        return [{"kind": "chunk", "document": loc.get("document"), "location": loc}]
    if isinstance(loc, list):
        return [{"kind": "chunk", "document": l.get("document"), "location": l}
                for l in loc if isinstance(l, dict)]
    return []


def check_factual_has_evidence(items: list[dict]) -> list[str]:
    """【S7】 사실 답변인데 근거가 하나도 없으면 실패.

    ★근거 없는 정답은 채점할 수 없다 — 출처 채점이 통째로 N/A 가 되어, 시스템이
      근거를 아무렇게나 붙이거나 안 붙여도 감점이 없어진다. 실제로 선별형 25문항
      전부가 근거 없이 통과하고 있었다.
    """
    errors = []
    for item in items:
        at = item.get("answer_type")
        if at not in _FACTUAL_ANSWER_TYPES:
            continue
        gold = item.get("answer_raw")
        if isinstance(gold, list) and not gold:
            continue          # 의도적인 빈 정답(0건)은 근거를 요구하지 않는다
        if not _evidence_of(item):
            errors.append(f"【S7】 {item.get('id')}: 사실 답변({at})인데 근거(location/evidence)가 없음")
    return errors


def check_selection_gold_recomputed(items: list[dict], registry_path=OFFICIAL_REGISTRY,
                                    identity_path=OFFICIAL_IDENTITY,
                                    table_path=OFFICIAL_TABLE) -> list[str]:
    """【S8】 선별형 정답을 독립 정책으로 다시 계산해 정확히 대조한다.

    ★등록부에 있는 문서 ID 만 썼는지(S1)로는 부족하다. 유효한 ID 로만 구성해도
      집합 자체가 틀릴 수 있고, 그건 S1 을 그대로 통과한다.
    """
    for path in (registry_path, identity_path, table_path):
        if not Path(path).exists():
            raise AssetMissing(f"공식 자료를 찾지 못했습니다: {path}")
    from checks.selection_policy import SELECTION_SPEC, compute_selection_gold, load_official
    official = load_official(registry_path, identity_path, table_path)
    errors = []
    for item in items:
        qid = item.get("id")
        if item.get("task_type") != "selection" or qid not in SELECTION_SPEC:
            continue
        expect = compute_selection_gold(qid, official)
        got = item.get("answer_raw")
        if not isinstance(got, list):
            errors.append(f"【S8】 {qid}: 선별형 정답이 리스트가 아님")
            continue
        if sorted(got) != expect["gold"]:
            missing = sorted(set(expect["gold"]) - set(got))
            extra = sorted(set(got) - set(expect["gold"]))
            errors.append(
                f"【S8】 {qid}: 독립 재계산과 정답 집합이 다름 "
                f"(조건: {expect['condition']}) 누락 {missing[:8]} / 초과 {extra[:8]}")
    return errors


def check_consortium_states(items: list[dict], registry_path=OFFICIAL_REGISTRY,
                            identity_path=OFFICIAL_IDENTITY,
                            table_path=OFFICIAL_TABLE) -> list[str]:
    """【S9】 '공동수급 금지' 문항에 특정 방식만 금지한 문서를 넣으면 실패.

    ★"공동이행방식은 허용하지 않음"은 다른 방식으로는 참여할 수 있다는 뜻이다.
      이걸 전면 금지로 세면 실제로 참여 가능한 공고가 '금지 공고' 정답에 섞인다.
    """
    for path in (registry_path, identity_path, table_path):
        if not Path(path).exists():
            raise AssetMissing(f"공식 자료를 찾지 못했습니다: {path}")
    from checks.selection_policy import (
        FORBIDDEN_ALL, METHOD_RESTRICTED, SELECTION_SPEC, _consortium_state, load_official,
    )
    official = load_official(registry_path, identity_path, table_path)
    errors = []
    for item in items:
        qid = item.get("id")
        if item.get("task_type") != "selection" or qid not in SELECTION_SPEC:
            continue
        if FORBIDDEN_ALL not in SELECTION_SPEC[qid][0]:
            continue
        for doc in (item.get("answer_raw") or []):
            st = _consortium_state(official, doc)
            if st == METHOD_RESTRICTED:
                errors.append(f"【S9】 {qid}: {doc} 는 특정 이행방식만 금지({st}) — "
                              f"공동수급 전면 금지 문항의 정답이 될 수 없음")
            elif st != FORBIDDEN_ALL:
                errors.append(f"【S9】 {qid}: {doc} 의 공동수급 상태는 {st} — 전면 금지 아님")
    return errors


# value 는 '하나의 짧은 값'이다. 이 길이를 넘으면 채점기의 의미 매칭 경로가 꺼져
# 축자 일치만 정답이 된다(normalize.match_short 의 60자 컷) — 유형 선택 오류다.
_VALUE_MAX_CHARS = 60


_CLARIFY_HINT = re.compile(r"말씀하시는\s*건가요|어떤\s*\S+(?:을|를)?\s*말씀|특정할\s*수\s*없|"
                           r"알려주시겠어요|어느\s*것을|무엇을\s*(?:말씀|찾)")


def _gold_is_clarification(item: dict) -> bool:
    gold = item.get("answer_raw")
    texts = [gold] if isinstance(gold, str) else [g for g in (gold or []) if isinstance(g, str)]
    return any(_CLARIFY_HINT.search(t) for t in texts)


def check_unspecified_type_consistency(items: list[dict]) -> list[str]:
    """【S14】 unspecified_type 은 '되묻기가 정답인 문항'에만 붙어야 한다.

    ★해소 가능한 문항에 붙어 있으면 채점기의 되묻기 수용 분기가 열려, 시스템이
      약어·기관명을 해소하지 못하고 되묻기만 해도 만점이 나온다.
    """
    errors = []
    for item in items:
        qid = item.get("id")
        has = item.get("unspecified_type") is not None
        clarify = _gold_is_clarification(item)
        if has and not clarify:
            errors.append(f"【S14】 {qid}: unspecified_type={item.get('unspecified_type')!r} 인데 "
                          f"정답이 되묻기가 아님 — 되묻기만 해도 만점이 되는 구멍")
        if clarify and not has:
            errors.append(f"【S14】 {qid}: 정답이 되묻는 문장인데 unspecified_type 이 없음 "
                          f"— 같은 취지로 되물어도 축자가 다르면 0점")
    return errors


def check_answer_type_shape(items: list[dict]) -> list[str]:
    """【S10】 답 유형과 실제 정답 구조가 어긋나면 실패."""
    errors = []
    for item in items:
        qid, at, gold = item.get("id"), item.get("answer_type"), item.get("answer_raw")
        # 되묻기가 정답인 문항은 문자열 길이 규칙 대상이 아니다 — 채점 경로가 다르다.
        if _gold_is_clarification(item) and item.get("unspecified_type") is not None:
            continue
        if at == "value":
            if isinstance(gold, list):
                errors.append(f"【S10】 {qid}: answer_type=value 인데 정답이 목록({len(gold)}개)")
                continue
            text = "" if gold is None else str(gold)
            if "\n" in text:
                errors.append(f"【S10】 {qid}: answer_type=value 인데 정답이 여러 줄 — "
                              f"독립 항목이면 list, 설명이면 summary")
            elif len(text) > _VALUE_MAX_CHARS:
                errors.append(
                    f"【S10】 {qid}: answer_type=value 인데 정답이 {len(text)}자 "
                    f"(>{_VALUE_MAX_CHARS}) — 채점기가 축자 일치만 인정하게 되어 "
                    f"뜻이 같은 답도 0점이 된다. list 또는 summary 로 바꿔야 함")
        elif at == "list":
            if not isinstance(gold, list):
                errors.append(f"【S10】 {qid}: answer_type=list 인데 정답이 리스트가 아님")
            elif any(isinstance(g, str) and len(g) > 120 for g in gold):
                long_ones = [i for i, g in enumerate(gold)
                             if isinstance(g, str) and len(g) > 120]
                errors.append(f"【S10】 {qid}: list 항목이 지나치게 긺(index {long_ones[:5]}) — "
                              f"항목 단위를 더 잘게 나눠야 축자 일치 외에 채점이 가능")
        elif at == "summary":
            if not isinstance(gold, list) or not gold:
                errors.append(f"【S10】 {qid}: summary 정답은 체크포인트 배열이어야 함")
            else:
                long_ones = [i for i, g in enumerate(gold)
                             if isinstance(g, str) and len(g) > 40]
                if long_ones:
                    errors.append(
                        f"【S10】 {qid}: 체크포인트가 원문 문장 수준으로 긺(index {long_ones[:5]}) — "
                        f"문자열 포함 매칭이라 부분점수가 나오지 않는다")
    return errors


_EVIDENCE_REQUIRED = {
    "chunk": ("document", "location"),
    "extraction_table": ("document", "field", "status"),
    "identity": ("document", "field"),
}


def check_evidence_shape(items: list[dict], table_path=OFFICIAL_TABLE,
                         identity_path=OFFICIAL_IDENTITY) -> list[str]:
    """【S11】 출처별 근거 형식 검증 — 추출표/청크/identity 각각의 필수 항목."""
    for path in (table_path, identity_path):
        if not Path(path).exists():
            raise AssetMissing(f"공식 자료를 찾지 못했습니다: {path}")
    table = {}
    for row in json.loads(Path(table_path).read_text(encoding="utf-8"))["rows"]:
        if str(row.get("active", "true")).lower() == "true":
            table.setdefault(row["document_id"], {})[row["field_name"]] = row
    with Path(identity_path).open(encoding="utf-8-sig") as f:
        ident_fields = set(next(csv.reader(f), []))
    errors = []
    for item in items:
        qid = item.get("id")
        for ev in (item.get("evidence") or []):
            if not isinstance(ev, dict):
                errors.append(f"【S11】 {qid}: 근거 원소가 객체가 아님")
                continue
            kind = ev.get("kind")
            if kind not in _EVIDENCE_REQUIRED:
                errors.append(f"【S11】 {qid}: 알 수 없는 근거 종류 {kind!r}")
                continue
            for key in _EVIDENCE_REQUIRED[kind]:
                if not ev.get(key):
                    errors.append(f"【S11】 {qid}: {kind} 근거에 {key} 없음")
            if kind == "extraction_table":
                row = (table.get(ev.get("document")) or {}).get(ev.get("field"))
                if row is None:
                    errors.append(f"【S11】 {qid}: 추출표에 {ev.get('document')}/"
                                  f"{ev.get('field')} 행이 없음")
                elif ev.get("status") != row.get("status"):
                    errors.append(
                        f"【S11】 {qid}: 근거 status={ev.get('status')} 인데 추출표는 "
                        f"{row.get('status')} — 근거가 실제 자료와 다름")
            elif kind == "identity" and ev.get("field") not in ident_fields:
                errors.append(f"【S11】 {qid}: identity_v2 에 없는 필드 {ev.get('field')!r}")
    return errors


def check_retrievable_evidence(items: list[dict], chunks_path=OFFICIAL_CHUNKS) -> list[str]:
    """【S12】 검색 불가능한 청크를 **유일한** 근거로 쓰면 실패.

    ★retrieval_eligible=false 인 청크(목차 등)는 검색으로 절대 못 가져온다.
      그걸 유일한 정답 근거로 두면 어떤 시스템도 그 근거를 댈 수 없다.
    """
    path = Path(chunks_path)
    if not path.exists():
        raise AssetMissing(f"chunks_v3 를 찾지 못했습니다: {path}")
    index = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                index[(c["document_id"], c.get("location_label"))] = c
    errors = []
    for item in items:
        qid = item.get("id")
        chunk_evs = [ev for ev in _evidence_of(item) if ev.get("kind") == "chunk"]
        if not chunk_evs:
            continue
        # 청크 외 근거(추출표·identity)가 함께 있으면 유일 근거가 아니다.
        has_other = any(ev.get("kind") != "chunk" for ev in _evidence_of(item))
        states = []
        for ev in chunk_evs:
            loc = ev.get("location") or {}
            c = index.get((loc.get("document"), loc.get("ref_no")))
            states.append(None if c is None else bool(c.get("retrieval_eligible")))
        known = [s for s in states if s is not None]
        if known and not any(known) and not has_other:
            if item.get("blocked_reason"):
                # 이미 상위(청킹) 결함으로 escalate 된 항목 — 조용히 넘어가는 것이 아니라
                # blocked 목록으로 따로 낸다. 사유를 안 적었으면 아래처럼 그냥 실패다.
                continue
            errors.append(
                f"【S12】 {qid}: 정답 근거가 전부 검색 불가능한 청크"
                f"(retrieval_eligible=false) — 어떤 시스템도 이 근거를 댈 수 없다")
    return errors


def collect_blocked(items: list[dict]) -> list[str]:
    """차단 항목 — 평가셋에서 고칠 수 없어 담당자 확인이 필요한 문항."""
    return [f"【차단】 {it.get('id')}: {it.get('blocked_reason')}"
            for it in items if it.get("blocked_reason")]


def check_evidence_relevance(items: list[dict], chunks_path=OFFICIAL_CHUNKS) -> list[str]:
    """【S13】 근거 좌표가 존재만 하고 정답과 무관한 경우를 탐지한다.

    ★자동 판정에는 한계가 있다 — 표현이 달라도 같은 뜻일 수 있다. 그래서 여기서는
      '정답의 핵심 토큰이 근거 청크 본문 어디에도 없다'는 **강한 신호**일 때만
      사람 검수 대기로 표시하고, 단정하지 않는다.
    """
    path = Path(chunks_path)
    if not path.exists():
        raise AssetMissing(f"chunks_v3 를 찾지 못했습니다: {path}")
    index = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                index[(c["document_id"], c.get("location_label"))] = c
    out = []
    num = re.compile(r"[0-9][0-9,]{2,}")
    for item in items:
        qid = item.get("id")
        gold = item.get("answer_raw")
        texts = ([gold] if isinstance(gold, str) else
                 [g for g in (gold or []) if isinstance(g, str)])
        tokens = {t.replace(",", "") for txt in texts for t in num.findall(txt)}
        if not tokens:
            continue
        chunk_evs = [ev for ev in _evidence_of(item) if ev.get("kind") == "chunk"]
        if not chunk_evs:
            continue
        blob = ""
        for ev in chunk_evs:
            loc = ev.get("location") or {}
            c = index.get((loc.get("document"), loc.get("ref_no")))
            if c:
                blob += (c.get("search_text") or c.get("content") or "")
        if not blob:
            continue
        blob_n = blob.replace(",", "")
        if not any(t in blob_n for t in tokens):
            out.append(f"【S13·사람검수】 {qid}: 정답의 숫자 {sorted(tokens)[:4]} 가 "
                       f"근거 청크 본문에 하나도 없음 — 근거가 정답과 무관할 수 있음")
    return out


def check_table_value_compatibility(items: list[dict], table_path=OFFICIAL_TABLE) -> list[str]:
    """【S15】 추출표 기반 value 문항의 정답이 연결된 추출표 값과 의미상 호환되는가.

    ★status=value_present 만 맞추고 값이 어긋나면, 모델이 공식 표 값을 그대로 답해도
      오답이 되는 평가셋이 된다(EXT-05 실측). 금액이면 핵심 금액이 같아야 하고,
      금액이 아니면 정규화한 한쪽이 다른 쪽을 포함해야 한다.
    """
    path = Path(table_path)
    if not path.exists():
        raise AssetMissing(f"추출표를 찾지 못했습니다: {path}")
    from grader.normalize import normalize_text, parse_amount
    table = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
        if str(row.get("active", "true")).lower() == "true":
            table.setdefault(row["document_id"], {})[row["field_name"]] = row
    errors = []
    for item in items:
        if item.get("answer_type") != "value" or item.get("answer_source") != "table":
            continue
        gold = item.get("answer_raw")
        if not isinstance(gold, str):
            continue
        for ev in (item.get("evidence") or []):
            if ev.get("kind") != "extraction_table" or ev.get("status") != "value_present":
                continue
            row = (table.get(ev.get("document")) or {}).get(ev.get("field"))
            if row is None:
                continue
            tv = str(row.get("answer_raw") or "")
            ga, ta = parse_amount(gold), parse_amount(tv)
            if ga is not None or ta is not None:
                if ga != ta:
                    errors.append(f"【S15】 {item.get('id')}: 정답 금액 {ga} ≠ 추출표 {ev.get('document')}/"
                                  f"{ev.get('field')} 금액 {ta} — 표 값을 그대로 답하면 오답이 된다")
            else:
                g, t = normalize_text(gold), normalize_text(tv)
                if g and t and g not in t and t not in g:
                    errors.append(f"【S15】 {item.get('id')}: 정답과 추출표 값이 서로 포함하지 않음 "
                                  f"(정답 {gold[:40]!r} / 표 {tv[:40]!r})")
            # 허용 답 목록에 표 값이 있어야 표 원문 그대로 답해도 정답이 된다 —
            # 금액 일치 여부와 **무관하게** 검사한다(일치해도 목록에 없으면 군더더기로 오답).
            norm = item.get("answer_normalized")
            accepted = norm if isinstance(norm, list) else [norm]
            if tv and tv != gold and tv not in accepted:
                errors.append(f"【S15】 {item.get('id')}: 추출표 원문 값이 허용 답 목록(answer_normalized)에 없음")
    return errors


# ── S16 정답 출처 라벨 ↔ 근거 종류 (2026-09-04) ────────────────────────
# answer_source 는 "정답이 어디서 왔는가"를 말하는 라벨이고, evidence.kind 는
# "그것을 어디서 확인할 수 있는가"다. 둘이 어긋나면 채점기가 엉뚱한 자료로
# 근거·검색을 채점한다. 실제로 EXT-07 이 answer_source=metadata 인데 근거는
# kind=chunk / source=chunks_v3 이고 위치가 "CSV: bid_deadline" 이었다 —
# identity 근거가 생성 과정에서 사라진 결과였다(재현 확인).
_SOURCE_REQUIRED_KIND = {
    "table": "extraction_table",   # 추출표에서 온 정답 → 추출표 근거가 있어야 한다
    "metadata": "identity",        # identity/등록부 메타데이터에서 온 정답 → identity 근거
}


def check_answer_source_evidence_kind(
        items: list[dict], table_path=OFFICIAL_TABLE, identity_path=OFFICIAL_IDENTITY,
        chunks_path=OFFICIAL_CHUNKS, spec_evidence_path=None) -> list[str]:
    """【S16】 정답 출처 라벨과 근거 종류가 서로 맞는가.

    검사 항목
      ① answer_source=table 인 비어 있지 않은 사실 문항 → extraction_table 근거 필요
      ② answer_source=metadata 인 value 문항        → identity 근거 필요
      ③ "CSV:" 위치를 chunks_v3 근거로 기록하면 실패(원문 청크가 아니다)
      ④ kind=chunk → 실제 청크 좌표여야 한다
      ⑤ kind=identity → identity_v2 에 그 컬럼과 문서 행이 있어야 한다
      ⑥ kind=extraction_table → 표의 문서·필드·상태와 일치해야 한다
      ⑦ 원본 명세에 명시적 근거가 있는데 생성 결과에서 사라지면 실패
    예외
      · 정답이 빈 목록인 선별형과 기권형(unanswerable)은 ①②를 적용하지 않는다.
        다만 무조건 면제하지 않는다 — 근거를 **가지고 있으면** ③~⑥은 그대로 검사한다.
    """
    import csv as _csv
    ipath, tpath = Path(identity_path), Path(table_path)
    if not ipath.exists():
        raise AssetMissing(f"identity_v2 를 찾지 못했습니다: {ipath}")
    if not tpath.exists():
        raise AssetMissing(f"추출표를 찾지 못했습니다: {tpath}")
    identity = {r["document_id"]: r for r in _csv.DictReader(ipath.open(encoding="utf-8-sig"))}
    table: dict[str, dict] = {}
    for row in json.loads(tpath.read_text(encoding="utf-8"))["rows"]:
        if str(row.get("active", "true")).lower() == "true":
            table.setdefault(row["document_id"], {})[row["field_name"]] = row
    chunks = _chunk_index(chunks_path)
    spec: dict[str, list[dict]] = {}
    if spec_evidence_path:
        sp = Path(spec_evidence_path)
        if not sp.exists():
            raise AssetMissing(f"명세 근거 파일을 찾지 못했습니다: {sp}")
        spec = json.loads(sp.read_text(encoding="utf-8")).get("evidence", {})

    errors: list[str] = []
    for item in items:
        qid = item.get("id")
        evs = [e for e in (item.get("evidence") or []) if isinstance(e, dict)]
        kinds = {e.get("kind") for e in evs}
        gold = item.get("answer_raw")
        empty_docset = item.get("answer_type") == "document_set" and isinstance(gold, list) and not gold
        exempt = empty_docset or item.get("answer_type") == "unanswerable"
        src = item.get("answer_source")

        # ① · ② 출처 라벨이 요구하는 근거 종류
        need = _SOURCE_REQUIRED_KIND.get(src)
        if need and not exempt and item.get("answer_type") in _FACTUAL_ANSWER_TYPES:
            if need == "identity" and item.get("answer_type") != "value":
                need = None                      # metadata 규칙은 값 문항에만 적용한다
            if need and need not in kinds:
                errors.append(f"【S16】 {qid}: answer_source={src} 인데 {need} 근거가 없음 "
                              f"(있는 근거: {sorted(k for k in kinds if k) or '없음'})")

        for e in evs:
            kind, doc, field = e.get("kind"), e.get("document"), e.get("field")
            loc = e.get("location") if isinstance(e.get("location"), dict) else None
            ref = str((loc or {}).get("ref_no") or "")
            # ③ CSV 참조를 원문 청크 근거라고 적으면 실패
            if ref.startswith("CSV:") and (kind == "chunk" or str(e.get("source") or "").startswith("chunks")):
                errors.append(f"【S16】 {qid}: CSV 참조({ref!r})를 청크 근거"
                              f"(kind={kind}/source={e.get('source')})로 기록했다 — 원문 청크가 아니다")
                continue
            if kind == "chunk":
                # ④ 실제 청크 좌표여야 한다
                if loc is None or not ref:
                    errors.append(f"【S16】 {qid}: kind=chunk 인데 원문 좌표가 없음")
                elif chunks.get((doc, ref)) is None:
                    errors.append(f"【S16】 {qid}: kind=chunk 좌표가 chunks_v3 에 없음 "
                                  f"(document={doc}, ref_no={ref!r})")
            elif kind == "identity":
                # ⑤ identity 에 문서 행과 컬럼이 있어야 한다
                if doc not in identity:
                    errors.append(f"【S16】 {qid}: identity_v2 에 문서 {doc} 없음")
                elif not field or field not in identity[doc]:
                    errors.append(f"【S16】 {qid}: identity_v2 에 컬럼 {field!r} 없음(document={doc})")
            elif kind == "extraction_table":
                # ⑥ 표의 문서·필드·상태와 일치해야 한다
                row = (table.get(doc) or {}).get(field)
                if row is None:
                    errors.append(f"【S16】 {qid}: 추출표에 {doc}/{field} 행 없음")
                elif e.get("status") and str(row.get("status")) != str(e.get("status")):
                    errors.append(f"【S16】 {qid}: 근거 상태 {e.get('status')} ≠ 추출표 상태 "
                                  f"{row.get('status')} ({doc}/{field})")

        # ⑦ 명세가 요구한 근거가 생성 결과에 남아 있는가
        for want in spec.get(qid, []):
            def _same(e):
                if e.get("kind") != want.get("kind") or e.get("document") != want.get("document"):
                    return False
                if want.get("kind") == "chunk":
                    return True          # 청크는 문서까지 맞으면 같은 근거로 본다(좌표는 ④에서 검사)
                return e.get("field") == want.get("field")
            if not any(_same(e) for e in evs):
                errors.append(f"【S16】 {qid}: 명세가 요구한 근거가 생성 결과에 없음 "
                              f"(kind={want.get('kind')}, document={want.get('document')}, "
                              f"field={want.get('field')})")
    return errors

def run_all_report(records, *, final_mode: bool = False, registry_path=OFFICIAL_REGISTRY,
                   chunks_path=OFFICIAL_CHUNKS, identity_path=OFFICIAL_IDENTITY,
                   table_path=OFFICIAL_TABLE, spec_evidence_path=None, **kwargs) -> dict:
    """검사 결과를 PASS/FAIL/SKIP 개수와 함께 돌려준다.

    final_mode=True 면 공식 자료가 없을 때 SKIP 하지 않고 FAIL 로 기록한다.
    """
    if final_mode:
        kwargs.setdefault("strict", True)
        kwargs.setdefault("final_set", True)
    problems = list(run_all(records, **kwargs))
    results = {"PASS": [], "FAIL": [], "SKIP": []}
    if problems:
        results["FAIL"].append({"check": "base(C0-C6)", "problems": problems})
    else:
        results["PASS"].append({"check": "base(C0-C6)"})
    extra = [
        ("S1 선별형 문서 범위", lambda: check_selection_scope(records, registry_path)),
        ("S2 기준 시각", lambda: check_reference_time(records)),
        ("S3 정답·근거 문서 일치", lambda: check_answer_evidence_docs(records)),
        ("S4 비교형 구조", lambda: check_comparison_structure(records)),
        ("S5 근거 좌표 존재", lambda: check_locations_exist(records, chunks_path, identity_path)),
        ("S6 정답 출처 라벨", lambda: check_answer_source_present(records)),
        ("S7 사실 답변 근거 존재", lambda: check_factual_has_evidence(records)),
        ("S8 선별형 정답 독립 재계산", lambda: check_selection_gold_recomputed(
            records, registry_path, identity_path, table_path)),
        ("S9 공동수급 전면금지/방식제한 구분", lambda: check_consortium_states(
            records, registry_path, identity_path, table_path)),
        ("S10 답 유형·정답 구조 일치", lambda: check_answer_type_shape(records)),
        ("S11 근거 형식(출처별)", lambda: check_evidence_shape(records, table_path, identity_path)),
        ("S12 검색 가능한 근거", lambda: check_retrievable_evidence(records, chunks_path)),
        ("S14 되묻기 문항 일관성", lambda: check_unspecified_type_consistency(records)),
        ("S15 추출표 값 정합(value·table)", lambda: check_table_value_compatibility(records, table_path)),
        ("S16 정답 출처 라벨 ↔ 근거 종류", lambda: check_answer_source_evidence_kind(
            records, table_path, identity_path, chunks_path, spec_evidence_path)),
    ]
    for name, fn in extra:
        try:
            found = fn()
        except AssetMissing as exc:
            if final_mode:
                results["FAIL"].append({"check": name, "problems": [f"공식 자료 없음: {exc}"]})
            else:
                results["SKIP"].append({"check": name, "reason": str(exc)})
            continue
        if found:
            results["FAIL"].append({"check": name, "problems": found})
        else:
            results["PASS"].append({"check": name})
    # S13 은 자동 단정이 불가능한 영역이라 실패가 아니라 **사람 검수 대기**로 낸다.
    results["REVIEW"] = []
    blocked = collect_blocked(records)
    if blocked:
        results["REVIEW"].append({"check": "차단 항목(담당자 확인 필요)", "items": blocked})
    try:
        flagged = check_evidence_relevance(records, chunks_path)
    except AssetMissing as exc:
        flagged = []
        results["SKIP"].append({"check": "S13 근거-정답 관련성", "reason": str(exc)})
    if flagged:
        results["REVIEW"].append({"check": "S13 근거-정답 관련성", "items": flagged})
    results["counts"] = {k: len(v) for k, v in results.items()
                         if k in ("PASS", "FAIL", "SKIP", "REVIEW")}
    results["ok"] = not results["FAIL"]
    return results

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="평가셋 계약 검사 (C0-C6, 2-17)")
    parser.add_argument("items_path", help="검사할 items.jsonl 경로")
    parser.add_argument("--strict", action="store_true", help="freeze용: 총량·비율 하드검사")
    parser.add_argument("--final-set", action="store_true", help="검사 대상이 최종셋 — C6 유출 검사 켬")
    parser.add_argument("--practice", default="/srv/rfp/evalset/practice_items.jsonl")
    parser.add_argument("--doc-ids", default=str(DEFAULT_DOC_IDS))
    parser.add_argument("--evalset-version", default="/srv/rfp/evalset/v1/VERSION.txt")
    parser.add_argument("--corpus-version",
                        default="/srv/rfp/shared_data/processed/corpus_v2/VERSION.txt")
    parser.add_argument("--chunking-version",
                        default="/srv/rfp/shared_data/processed/chunks_v3/VERSION.txt")
    parser.add_argument("--leak-scan-root", default=None, help="【25】 추적 파일 유출 스캔 루트")
    parser.add_argument("--final-mode", action="store_true",
                        help="최종 평가셋 검사 — 강화 검사(S1-S6) 포함, 공식 자료 없으면 SKIP 대신 실패")
    parser.add_argument("--registry", default=OFFICIAL_REGISTRY)
    parser.add_argument("--table", default=OFFICIAL_TABLE,
                        help="추출표 경로(후보 추출표로 정답을 재계산할 때 지정)")
    parser.add_argument("--chunks", default=OFFICIAL_CHUNKS)
    parser.add_argument("--identity", default=OFFICIAL_IDENTITY)
    parser.add_argument("--spec-evidence", default=None,
                        help="생성 명세가 요구한 근거 목록(spec_evidence.json) — S16 이 "
                             "생성 결과에서 명시적 근거가 사라졌는지 확인한다")
    args = parser.parse_args(argv)

    try:
        items = load_jsonl(args.items_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"C0 FAIL: {e}")
        return 1
    print(f"C0 PASS: {len(items)} items loaded")

    if args.final_mode:
        report = run_all_report(
            items, final_mode=True,
            registry_path=args.registry, chunks_path=args.chunks, identity_path=args.identity, table_path=args.table,
            spec_evidence_path=args.spec_evidence,
            doc_ids_path=args.doc_ids, practice_path=args.practice,
            evalset_version_path=args.evalset_version, corpus_version_path=args.corpus_version,
            chunking_version_path=args.chunking_version,
            leak_repo_root=args.leak_scan_root, leak_exclude=[args.items_path],
        )
        for entry in report["FAIL"]:
            print(f"  FAIL {entry['check']}")
            for p in entry["problems"]:
                print(f"    {p}")
        for entry in report["SKIP"]:
            print(f"  SKIP {entry['check']}: {entry['reason']}")
        for entry in report.get("REVIEW", []):
            # 자동으로 단정할 수 없는 항목 — 조용히 통과시키지 않고 눈에 보이게 남긴다.
            print(f"  REVIEW {entry['check']}")
            for line in entry["items"]:
                print(f"    {line}")
        c = report["counts"]
        print(f"\n{'PASS' if report['ok'] else 'FAIL'} — {len(items)} items, "
              f"PASS {c['PASS']} / FAIL {c['FAIL']} / SKIP {c['SKIP']} / "
              f"REVIEW {c.get('REVIEW', 0)}")
        return 0 if report["ok"] else 1

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
