"""평가셋 계약 검사 (2-17) — 임현진 소관, **팀 단일 출처**.

원본: origin/HJ `src/checks/check_evalset.py` (김하루 벤더링 2026-09-01).
  - C4 참조 무결성 버그 수정 (정답 배열 필드명·item_id → id)
  - C5 버전 검사: [대기] 동안 안전하게 SKIP
  - C6 유출 검사 구현 (grader.validation.leakage 이식)
  - 좌표 정밀도 검사(2-9) 추가 — ref_no 자리표시자·part/of
  - run_all() 오케스트레이터 + main() argparse

HJ 의 실제 파일이 dev 에 머지되면 이 파일을 그 버전으로 재조정한다(구조를 최대한 맞춰 둠).

검사는 **raw dict**(JSONL 한 줄) 위에서 돈다 — pydantic 파싱 전에, 파일 자체를 믿기 전에.
grader 의 EvaluationItem 파싱(grader.validation.schema)은 별개 단계다.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import List

# status: "PASS" | "FAIL" | "SKIP"
CheckResult = namedtuple("CheckResult", ["name", "status", "message"])

# reference_time 은 정해진 상수값 (임현진 확정 — now 금지, 평가셋이 저절로 틀려짐)
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
    "location": {"document", "section", "ref_no"},
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
# 스키마 v0.2 에서 폐기된 필드 — 문항에 남아 있으면 FAIL
DEPRECATED_FIELDS = (
    "document_unspecified",
    "time_dependent",
    "conversational",
    "difficulty",
    "schema_version",
    "checkpoints",
    "unanswerable_reason",
)

ESSENTIAL_FIELDS = ["id", "question", "task_type", "answer_type", "answer_raw"]

# 1-9-1 확정: 수집 중복 2건은 검색 대상이 아니므로 정답 근거로 금지
DEFAULT_EXCLUDED_DOC_IDS = ("RFP-000006", "RFP-000017")
# 2-13: practice 전용 3개 문서 — 최종 50문항 제작 시 제외
PRACTICE_ONLY_DOCS = ("RFP-000038", "RFP-000043", "RFP-000001")

_PART_OF_RE = re.compile(r"\s*\(\d+\s*/\s*\d+\)\s*$")
_COORD_PLACEHOLDERS = {"", "<문서id>", "todo", "[대기]", "?", "-", "n/a", "na"}


# ────────────────────────────────────────────── 로딩

def load_jsonl(path: str | Path) -> List[dict]:
    """JSONL(대괄호·콤마 없이 줄마다 독립 JSON) 로드. 빈 줄은 건너뛴다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"line {line_num}: {e}")
    if errors:
        raise ValueError("\n".join(errors))
    return records


# ────────────────────────────────────────────── C1 스키마

def _gold_doc_ids(item: dict) -> list[str]:
    """이 문항이 정답 근거로 가리키는 문서 ID 전부.

    - 선별형(answer_type=document_set): answer_raw 가 문서 ID 배열
    - 그 외: document_id (문자열 또는 배열), intermediate_answer (문서 미특정형)
    - location.document
    """
    out: list[str] = []
    if item.get("answer_type") == "document_set":
        raw = item.get("answer_raw")
        if isinstance(raw, list):
            out += [str(d) for d in raw]
        elif isinstance(raw, str) and raw:
            out.append(raw)
    for key in ("document_id", "intermediate_answer", "active_document_id"):
        v = item.get(key)
        if isinstance(v, list):
            out += [str(d) for d in v]
        elif isinstance(v, str) and v:
            out.append(v)
    loc = item.get("location")
    if isinstance(loc, dict) and isinstance(loc.get("document"), str) and loc["document"]:
        out.append(loc["document"])
    return out


def check_schema(item: dict, line_num: int) -> List[str]:
    """C1: 스키마 준수. 결여·위반 사항을 리스트로 반환(빈 리스트 = 통과)."""
    errors: list[str] = []

    # 1: 공통 필수 5개 필드 존재
    missing = [f for f in ESSENTIAL_FIELDS if f not in item]
    errors += [f"line {line_num}: missing field '{f}'" for f in missing]

    # 2: FIELD_SPEC 타입/enum
    for field_name, spec in FIELD_SPEC.items():
        if field_name not in item:
            continue
        value = item[field_name]

        if field_name == "reference_time":
            if value != REFERENCE_TIME:
                errors.append(f"line {line_num}: reference_time must be {REFERENCE_TIME!r}, got {value!r}")
        elif spec == "string":
            if not isinstance(value, str) or value.strip() == "":
                errors.append(f"line {line_num}: field '{field_name}' must be a non-empty string, got {value!r}")
        elif spec == "any":
            pass
        elif isinstance(spec, tuple):
            if value not in spec:
                errors.append(f"line {line_num}: invalid '{value}' for field '{field_name}' (allowed: {spec})")
        elif isinstance(spec, list):
            allowed = tuple(TYPE_MAP[s] for s in spec)
            if not isinstance(value, allowed):
                errors.append(f"line {line_num}: type error {value!r} for field '{field_name}'")
        elif isinstance(spec, set):
            if not isinstance(value, dict):
                errors.append(f"line {line_num}: field '{field_name}' must be an object, got {value!r}")
            elif not spec.issubset(value.keys()):
                errors.append(f"line {line_num}: location missing key(s) {spec - value.keys()}")

    # 3: 폐기 필드
    for field_name in item:
        if field_name in DEPRECATED_FIELDS:
            errors.append(f"line {line_num}: deprecated field '{field_name}' (schema v0.2)")

    # 4: task_type x answer_type 조합
    task_type, answer_type = item.get("task_type"), item.get("answer_type")
    if task_type in TASK_ANSWER_COMBOS and answer_type not in TASK_ANSWER_COMBOS[task_type]:
        errors.append(
            f"line {line_num}: '{task_type}' cannot have answer_type '{answer_type}' "
            f"(allowed: {TASK_ANSWER_COMBOS[task_type]})")

    # 5: 필드 의존성 (FIELD_DEPENDENCIES)
    for trigger, required in FIELD_DEPENDENCIES.items():
        if trigger in item:
            absent = [r for r in required if not item.get(r)]
            if absent:
                errors.append(f"line {line_num}: '{trigger}' requires field(s) {absent}")

    # 6: selection 은 answer_source 필수
    if task_type == "selection" and "answer_source" not in item:
        errors.append(f"line {line_num}: task_type 'selection' needs 'answer_source'")

    # 7: 명시적 null 금지
    for k, v in item.items():
        if v is None:
            errors.append(f"line {line_num}: field '{k}' is null (drop the key instead)")

    # 8: 정답 형태 정합성
    if answer_type == "list":
        gold = item.get("answer_normalized", item.get("answer_raw"))
        if not isinstance(gold, list):
            errors.append(f"line {line_num}: answer_type=list but answer is not an array")
    if answer_type == "unanswerable":
        raw = item.get("answer_raw")
        if not (isinstance(raw, str) and raw):
            errors.append(f"line {line_num}: answer_type=unanswerable needs a reason string in answer_raw")

    return errors


# ────────────────────────────────────────────── C2 중복 id

def check_dup_ids(items: List[dict]) -> List[str]:
    counts = collections.Counter(item.get("id") for item in items)
    return [f"C2: duplicate id: {i} ({c}회)" for i, c in counts.items() if c > 1]


# ────────────────────────────────────────────── C3 할당량

def check_quota(items: List[dict], expected: dict | None = None, strict: bool = False) -> List[str]:
    """task_type 별 문항 수. expected 기본값 = 선별25 : 추출15 : QA10 (총 50)."""
    expected = expected or {"selection": 25, "extraction": 15, "qa": 10}
    errors: list[str] = []
    counts = collections.Counter(item.get("task_type") for item in items)
    for tt in counts:
        if tt not in expected:
            errors.append(f"C3: unknown task_type: {tt}")
    if strict:
        for tt, n in expected.items():
            got = counts.get(tt, 0)
            if got != n:
                errors.append(f"C3: {tt} expected {n}, got {got}")
    return errors


# ────────────────────────────────────────────── C4 참조 무결성

def check_ref_intg(
    items: List[dict],
    doc_ids_path: str | Path | None = None,
    valid_ids: set[str] | None = None,
    excluded_ids: set[str] | None = None,
) -> List[str]:
    """정답 근거 문서가 (1) 코퍼스에 실재하고 (2) 검색 대상인지.

    valid_ids 를 직접 주거나 doc_ids_path(corpus_doc_ids.json)로 준다. 둘 다 없으면 SKIP.
    excluded_ids: 검색 대상 아닌 문서(수집 중복 등). 정답 근거로 쓰면 에러.
    """
    if valid_ids is None and doc_ids_path is not None:
        p = Path(doc_ids_path)
        if not p.exists():
            raise FileNotFoundError(doc_ids_path)
        valid_ids = set(json.loads(p.read_text(encoding="utf-8")))
    excluded_ids = excluded_ids or set()

    errors: list[str] = []
    for item in items:
        iid = item.get("id")
        gold = sorted(set(_gold_doc_ids(item)))
        if valid_ids is not None:
            for d in gold:
                if d not in valid_ids:
                    errors.append(f"C4: {iid}: 코퍼스에 없는 문서 {d}")
        hit = sorted(set(gold) & excluded_ids)
        if hit:
            errors.append(f"C4: {iid}: 정답 근거가 검색 대상 아닌 문서 {hit} (수집 중복, 1-9-1)")
    return errors


# ────────────────────────────────────────────── 좌표 정밀도 (2-9)

def check_location_coords(items: List[dict]) -> List[str]:
    """location 이 있는 문항의 3필드가 실제 값인지 — ref_no 단위 좌표 채점(3-4-3) 성립 조건."""
    errors: list[str] = []
    for item in items:
        loc = item.get("location")
        if not isinstance(loc, dict):
            continue
        iid = item.get("id")
        for key in ("document", "section", "ref_no"):
            v = loc.get(key)
            if v is None or str(v).strip().lower() in _COORD_PLACEHOLDERS:
                errors.append(f"C-loc: {iid}: location.{key} 가 비었거나 자리표시자 ({v!r})")
        if _PART_OF_RE.search(str(loc.get("ref_no") or "")):
            errors.append(
                f"C-loc: {iid}: location.ref_no 에 조각 표기 (N/M) — 팀 확정 형식은 part/of 제외 "
                f"(예: '표 7'). 채점은 자동으로 뗀다")
    return errors


# ────────────────────────────────────────────── C5 코퍼스 버전

def check_version(items: List[dict], version_txt_path: str | Path | None = None) -> List[str]:
    """VERSION.txt 의 corpus 값과 평가셋이 기대하는 코퍼스 버전 대조.

    corpus 가 "[대기]" 이거나 VERSION.txt 가 없으면 SKIP(에러 없음). 1-19 도착 후 활성화.
    """
    if not version_txt_path:
        return []
    p = Path(version_txt_path)
    if not p.exists():
        return []
    corpus_val = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            if k.strip().lower().replace(" version", "") == "corpus":
                corpus_val = v.strip()
    if corpus_val in (None, "[대기]", "TBD", ""):
        return []
    # 활성화 시 대조 로직 — 지금은 값 존재만 확인
    return []


# ────────────────────────────────────────────── C6 최종셋 유출

def _tracked_files(repo_root: Path) -> List[Path]:
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        return [repo_root / line for line in out.stdout.splitlines()]
    except Exception:
        return []


_EVALSET_DIRS = ("/tests/fixtures/", "/data/", "/examples/", "/evalset/")


def check_leak(
    items: List[dict],
    practice_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    min_len: int = 12,
    exclude_paths: List[str | Path] = (),
) -> List[str]:
    """C6: 최종셋 유출 방지.

    1) 최종셋 id 에 'PRAC-' 접두어가 섞임
    2) practice 세트와 id 교집합
    3) practice 전용 문서(2-13)를 근거로 씀
    4) 문항 텍스트가 추적 파일(프롬프트·코드)에 그대로 있음 (repo_root 줄 때만)
    """
    errors: list[str] = []

    prac_prefixed = [i.get("id") for i in items if str(i.get("id", "")).startswith("PRAC-")]
    if prac_prefixed:
        errors.append(f"C6: 최종셋에 practice id 접두어(PRAC-) {prac_prefixed}")

    if practice_path and Path(practice_path).exists():
        prac_ids = {r.get("id") for r in load_jsonl(practice_path)}
        overlap = sorted({i.get("id") for i in items} & prac_ids)
        if overlap:
            errors.append(f"C6: practice 세트와 id 교집합 {overlap}")

    for item in items:
        used = set(_gold_doc_ids(item)) & set(PRACTICE_ONLY_DOCS)
        if used:
            errors.append(f"C6: {item.get('id')}: practice 전용 문서 {sorted(used)} 를 근거로 씀 (2-13)")

    if repo_root:
        root = Path(repo_root)
        skip = {str(Path(p).resolve()) for p in exclude_paths}
        needles = {i.get("id"): i.get("question") for i in items
                   if i.get("question") and len(i["question"]) >= min_len}
        for path in _tracked_files(root):
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
                    errors.append(f"C6: {qid}: 문항 텍스트가 추적 파일 {rel} 에 있음")
    return errors


# ────────────────────────────────────────────── 오케스트레이터

def run_all(
    records: List[dict],
    *,
    corpus_doc_ids: set[str] | None = None,
    excluded_doc_ids: set[str] | None = None,
    quota: dict | None = None,
    quota_strict: bool = False,
    practice_path: str | Path | None = None,
    version_txt_path: str | Path | None = None,
    leak_repo_root: str | Path | None = None,
    leak_exclude: List[str | Path] = (),
) -> List[str]:
    """평가셋 계약 검사 전체 (2-17). 값싼 것부터: C1 → C2 → C3 → 좌표 → C4 → C5 → C6."""
    problems: list[str] = []
    for i, item in enumerate(records, start=1):
        problems += check_schema(item, i)
    problems += check_dup_ids(records)
    problems += check_quota(records, quota, strict=quota_strict)
    problems += check_location_coords(records)
    problems += check_ref_intg(records, valid_ids=corpus_doc_ids, excluded_ids=excluded_doc_ids)
    problems += check_version(records, version_txt_path)
    problems += check_leak(records, practice_path, leak_repo_root, exclude_paths=leak_exclude)
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="평가셋 계약 검사 (2-17)")
    ap.add_argument("items_path", help="평가셋 JSONL")
    ap.add_argument("--doc-ids", help="corpus_doc_ids.json (참조 무결성)")
    ap.add_argument("--excluded-doc-ids", help="excluded_doc_ids.json")
    ap.add_argument("--practice", help="practice_items.jsonl (유출 검사)")
    ap.add_argument("--version-txt", help="VERSION.txt (코퍼스 버전)")
    ap.add_argument("--strict", action="store_true", help="할당량 정확히 대조 (총 50)")
    ap.add_argument("--leak-scan-root", help="추적 파일 유출 스캔 루트 (보통 저장소 루트)")
    a = ap.parse_args(argv)

    records = load_jsonl(a.items_path)
    corpus_ids = set(json.loads(Path(a.doc_ids).read_text(encoding="utf-8"))) if a.doc_ids else None
    excluded = set(json.loads(Path(a.excluded_doc_ids).read_text(encoding="utf-8"))) if a.excluded_doc_ids else None

    problems = run_all(
        records,
        corpus_doc_ids=corpus_ids,
        excluded_doc_ids=excluded,
        quota_strict=a.strict,
        practice_path=a.practice,
        version_txt_path=a.version_txt,
        leak_repo_root=a.leak_scan_root,
        leak_exclude=[a.items_path] + ([a.practice] if a.practice else []),
    )
    for p in problems:
        print(p)
    print(f"\n{'FAIL' if problems else 'PASS'} — {len(records)} items, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
