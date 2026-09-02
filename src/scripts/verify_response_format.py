"""
responses.jsonl 형식 검증기 — 김하루님 채점기 대체(2026-09-02 사용자 확정).

⚠️ 이것은 **채점기가 아니다**. 정답 여부를 매기지 않는다. 채점기가 읽을 수 있는
   형식·필드·근거 구조를 갖췄는지만 확인한다. 실제 채점은 채점기 실행 파일이
   공유 경로에 올라온 뒤에 별도로 해야 한다(base.yaml scorer: null).

사용법:
  python3 verify_response_format.py --responses <경로> [--evalset <경로>] \
      [--out <리포트 json 경로>]
종료코드: 위반 0건이면 0, 1건 이상이면 1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_TOP_LEVEL = [
    "id", "answer", "structured_answer", "contexts", "retrieved", "citations",
    "selected_document_ids", "abstained", "route", "failure", "latency_ms",
    "cost_usd",
]
CITATION_REQUIRED = ["document", "section", "ref_no"]
CHUNK_RECORD_REQUIRED = [
    "document_id", "section_path", "block_type", "block_index",
    "md_line_start", "md_line_end", "search_text",
]
DOC_ID_RE = re.compile(r"^RFP-\d{6}$")

# 문서를 아예 조회하지 않는 경로 — 근거가 없는 게 정상이다
NO_EVIDENCE_ROUTES = {"검색불필요_인사응답", "검색불필요_사용법안내"}

# ZIP·리포트에 원문 개인정보가 새지 않는지 확인하는 후보 패턴
PII_PATTERNS = {
    "api_key": re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b"),
}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i} JSON 파싱 실패: {e}") from e
    return out


def check_response(rec: dict, index: int) -> list[str]:
    problems: list[str] = []
    rid = rec.get("id", f"(id 없음, {index}번째 줄)")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in rec]
    if missing:
        problems.append(f"{rid}: 최상위 필수 키 누락 {missing}")
    extra = [k for k in rec if k not in REQUIRED_TOP_LEVEL]
    if extra:
        problems.append(f"{rid}: 최상위에 예상 밖 키 {extra}")

    for key in ("contexts", "retrieved", "citations", "selected_document_ids"):
        if key in rec and not isinstance(rec[key], list):
            problems.append(f"{rid}: {key}는 리스트여야 합니다(현재 {type(rec[key]).__name__})")

    for i, c in enumerate(rec.get("citations") or []):
        if not isinstance(c, dict):
            problems.append(f"{rid}: citations[{i}]가 객체가 아닙니다")
            continue
        for k in CITATION_REQUIRED:
            if k not in c:
                problems.append(f"{rid}: citations[{i}]에 '{k}'가 없습니다")
        if "document" in c and not DOC_ID_RE.match(str(c["document"])):
            problems.append(f"{rid}: citations[{i}].document 형식이 RFP-000000이 아닙니다"
                            f"({c['document']!r})")
        if "section" in c and not isinstance(c["section"], str):
            problems.append(f"{rid}: citations[{i}].section이 문자열이 아닙니다"
                            f"({type(c['section']).__name__}) — dict가 통째로 새면 안 됩니다")
        if "ref_no" in c and not isinstance(c["ref_no"], str):
            problems.append(f"{rid}: citations[{i}].ref_no가 문자열이 아닙니다")

    for key in ("contexts", "retrieved"):
        for i, ch in enumerate(rec.get(key) or []):
            if not isinstance(ch, dict):
                problems.append(f"{rid}: {key}[{i}]가 객체가 아닙니다")
                continue
            for k in CHUNK_RECORD_REQUIRED:
                if k not in ch:
                    problems.append(f"{rid}: {key}[{i}]에 '{k}'가 없습니다")
            if ch.get("block_type") not in (None, "table") and ch.get("block_index") is None:
                problems.append(f"{rid}: {key}[{i}] 텍스트 청크의 block_index가 None입니다")

    for did in rec.get("selected_document_ids") or []:
        if not DOC_ID_RE.match(str(did)):
            problems.append(f"{rid}: selected_document_ids에 형식이 다른 값 {did!r}")

    if rec.get("abstained") not in (True, False, None):
        problems.append(f"{rid}: abstained가 불리언이 아닙니다({rec.get('abstained')!r})")
    if rec.get("latency_ms") is not None and not isinstance(rec["latency_ms"], (int, float)):
        problems.append(f"{rid}: latency_ms가 숫자가 아닙니다")
    if rec.get("cost_usd") is not None and not isinstance(rec["cost_usd"], (int, float)):
        problems.append(f"{rid}: cost_usd가 숫자가 아닙니다")

    blob = json.dumps(rec, ensure_ascii=False)
    for name, pat in PII_PATTERNS.items():
        if pat.search(blob):
            problems.append(f"{rid}: 응답에 {name} 후보가 들어 있습니다")

    if (not rec.get("abstained") and rec.get("failure") is None
            and rec.get("route") not in NO_EVIDENCE_ROUTES):
        if not (rec.get("citations") or rec.get("selected_document_ids")):
            problems.append(f"{rid}: 기권도 실패도 아닌데 근거(citations)와 "
                            f"선택 문서가 모두 비어 있습니다")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--evalset", required=False,
                        help="주면 문항 수·id 일치까지 확인한다")
    parser.add_argument("--out", required=False, help="리포트 JSON 저장 경로")
    args = parser.parse_args()

    responses = load_jsonl(Path(args.responses))
    problems: list[str] = []
    if not responses:
        problems.append("responses.jsonl에 응답이 0건입니다")

    ids = [r.get("id") for r in responses]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"id 중복: {sorted(dupes)}")

    for i, rec in enumerate(responses, 1):
        problems.extend(check_response(rec, i))

    evalset_report = None
    if args.evalset:
        items = load_jsonl(Path(args.evalset))
        eval_ids = [str(it.get("id") or it.get("question_id")) for it in items]
        missing = [i for i in eval_ids if i not in ids]
        extra = [i for i in ids if i not in eval_ids]
        if missing:
            problems.append(f"평가셋에 있는데 응답에 없는 문항: {missing}")
        if extra:
            problems.append(f"평가셋에 없는 응답: {extra}")
        evalset_report = {
            "evalset_items": len(items), "responses": len(responses),
            "missing": missing, "extra": extra,
        }

    report = {
        "responses_path": str(args.responses),
        "response_count": len(responses),
        "required_top_level_keys": REQUIRED_TOP_LEVEL,
        "violations": problems,
        "violation_count": len(problems),
        "evalset_check": evalset_report,
        "per_item": [
            {"id": r.get("id"), "route": r.get("route"),
             "abstained": r.get("abstained"),
             "citation_count": len(r.get("citations") or []),
             "context_count": len(r.get("contexts") or []),
             "retrieved_count": len(r.get("retrieved") or []),
             "selected_document_ids": r.get("selected_document_ids"),
             "failure": r.get("failure"),
             "latency_ms": r.get("latency_ms"), "cost_usd": r.get("cost_usd")}
            for r in responses
        ],
        "note": ("이 리포트는 형식 검증 결과입니다. 정답 여부(채점)는 포함하지 "
                 "않습니다 — 김하루님 채점기가 공유 경로에 올라오면 별도로 채점해야 합니다."),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    if problems:
        print(f"❌ 형식 위반 {len(problems)}건")
        for p in problems[:30]:
            print(f"  - {p}")
        sys.exit(1)
    print(f"✅ 형식 위반 0건 (응답 {len(responses)}건)")


if __name__ == "__main__":
    main()
