"""후보 평가셋 v2 생성 — 질문·정답·답유형·근거의 의미를 맞춘다.

⚠️ 모델 코드(src/rag, src/scripts)를 가져다 쓰지 않는다. 공식 등록부·추출표·
   identity·청크만 읽어 정답을 독립적으로 계산한다.

바꾸는 것
  §2 선별형 정답 — 공동수급 7상태 재분류 + 마감 필터 + 미기재 질문 정정
  §3 답 유형   — 실제 정답 구조에 맞춰 value/list/summary 재지정
  §4 근거      — 출처별(청크·추출표·identity) 근거를 문항마다 채움
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_path_args, data_root, official_paths, repo_root, require,
)

TABLE_VERSION = "v3"   # main() 에서 실제 사용한 표의 extraction_version 으로 덮어쓴다
IDENTITY_VERSION = "v2"


# ── 근거 만들기 ──────────────────────────────────────────────────────

def table_evidence(official, doc, field) -> dict:
    """추출표 행 하나를 근거로. 원문 위치가 있으면 함께 담는다.

    ★status=field_absent 는 가리킬 원문 줄이 없다. 없는 좌표를 지어내지 않고
      '이 문서의 이 항목이 비어 있다'는 사실 자체를 근거로 기록한다.
    """
    row = (official["table"].get(doc) or {}).get(field) or {}
    ev = {"kind": "extraction_table", "document": doc, "field": field,
          "status": row.get("status", "missing"),
          "source": f"extraction_table_{TABLE_VERSION}"}
    loc = row.get("representative_location")
    if row.get("status") == "value_present" and isinstance(loc, dict):
        sec = loc.get("section_path") or []
        section = (sec[-1]["title"] if sec and isinstance(sec[-1], dict)
                   else (loc.get("heading") or ""))
        line = loc.get("line_start", loc.get("line"))
        if section and line is not None:
            ev["location"] = {
                "document": doc, "field": field, "section": section,
                "ref_no": f"{section} · {loc.get('block_type','')} {loc.get('block_index','')}".strip(),
                "line": line, "line_end": loc.get("line_end", line),
            }
            ev["raw_location_source_type"] = loc.get("source_type")
    return ev


def identity_evidence(doc, field, value) -> dict:
    return {"kind": "identity", "document": doc, "field": field,
            "value": value, "source": f"identity_{IDENTITY_VERSION}"}


def chunk_evidence(loc: dict, fact_field: str | None = None) -> dict:
    ev = {"kind": "chunk", "document": loc["document"],
          "field": loc.get("field"), "source": "chunks_v3", "location": loc}
    if fact_field:
        # 같은 사실을 다른 출처로도 확인할 수 있을 때 쓰는 표준 필드명(3-4-3 사실 묶음)
        ev["fact_field"] = fact_field
    return ev


# ── §2 선별형 ───────────────────────────────────────────────────────

# 조건이 보는 추출표 필드 — 근거는 '그 문서가 조건을 만족한다고 판단한 행'이다.
SELECTION_EVIDENCE_FIELDS = {
    "SEL-001": ["공고일"], "SEL-002": ["사업기간"], "SEL-003": ["사업분야"],
    "SEL-004": ["지역제한"], "SEL-005": ["필수 제출 서류"], "SEL-006": ["제출 방식"],
    "SEL-007": ["컨소시엄 요건"], "SEL-008": ["참가 자격(면허·실적)"], "SEL-009": ["예산"],
    "SEL-010": ["평가 배점"], "SEL-011": ["지역제한"], "SEL-012": ["필수 제출 서류"],
    "SEL-013": ["제출 방식"], "SEL-014": ["컨소시엄 요건"], "SEL-015": ["참가 자격(면허·실적)"],
    "SEL-016": ["예산"], "SEL-017": ["평가 배점"],
    "SEL-018": ["제출 방식", "필수 제출 서류"],
    "SEL-019": ["참가 자격(면허·실적)", "컨소시엄 요건"],
    "SEL-020": ["지역제한", "컨소시엄 요건"],
    "SEL-021": ["제출 방식", "컨소시엄 요건"],
    "SEL-022": ["컨소시엄 요건", "평가 배점"],
    "SEL-023": ["참가 자격(면허·실적)", "컨소시엄 요건"],
    "SEL-024": ["지역제한", "컨소시엄 요건"],
    "SEL-025": ["지역제한", "평가 배점"],
}

# §2-3 / §2-4 질문 재작성.
#   미기재(field_absent) 문항에서 "그러니 ~해야 한다 / ~라는 뜻이다" 류의 추론을 뺀다.
#   field_absent 는 "그 정책이 없다"가 아니라 "그 항목이 문서에 적혀 있지 않다"이다.
SELECTION_QUESTIONS = {
  "SEL-011": "지역제한 항목이 공고문에 적혀 있지 않은 공고만 모아서 알려줘.",
  "SEL-012": "필수 제출 서류 항목이 공고문에 적혀 있지 않은 공고들을 찾아줘.",
  "SEL-013": "제출 방식 항목 자체가 공고문에 적혀 있지 않은 공고만 따로 추려줘.",
  "SEL-014": "공동수급(컨소시엄) 요건 항목이 공고문에 적혀 있지 않은 공고를 모아줘.",
  "SEL-015": "참가 자격 항목이 공고문에 적혀 있지 않은 공고 목록을 줘.",
  "SEL-016": "예산 항목이 공고문에 적혀 있지 않은 공고들만 알려줘.",
  "SEL-017": "평가 배점 항목이 공고문에 적혀 있지 않은 공고 목록을 줘.",
  "SEL-018": "제출 방식 항목과 필수 제출 서류 항목이 둘 다 공고문에 적혀 있지 않은 공고를 뽑아줘.",
  "SEL-019": "참가 자격은 명시되어 있고, 공동수급 요건 항목은 공고문에 적혀 있지 않은 공고를 찾아줘.",
  "SEL-020": "지역 제한이 명시되어 있고, 공동수급 참여가 허용된다고 명시된 공고를 모두 정리해줘.",
  "SEL-022": "공동수급 요건은 명시되어 있고, 평가 배점 항목은 공고문에 적혀 있지 않은 공고를 알려줘.",
  "SEL-023": "참가 자격이 명시되어 있고, 공동수급은 명시적으로 금지된 공고를 모두 정리해줘.",
  "SEL-024": "지역 제한이 명시되어 있고, 공동수급은 명시적으로 금지된 공고를 찾아줘.",
  "SEL-025": "지역 제한은 명시되어 있고, 평가 배점 항목은 공고문에 적혀 있지 않은 공고를 정리해줘.",
}


def build_selection(items: dict, official: dict) -> tuple[dict, dict]:
    out, report = {}, {}
    for qid in sorted(SELECTION_SPEC):
        g = compute_selection_gold(qid, official, REFERENCE_TIME)
        it = dict(items[qid])
        before_q, before_gold = it["question"], list(it.get("answer_raw") or [])
        if qid in SELECTION_QUESTIONS:
            it["question"] = SELECTION_QUESTIONS[qid]
        # ★answer_set 은 answer_raw 와 같은 값을 중복 보관하던 필드다. 두 벌을 두면
        #   한쪽만 고쳐지는 사고가 난다 — 단일 출처(answer_raw)만 남긴다.
        it["answer_raw"] = g["gold"]
        it.pop("answer_set", None)
        it["reference_time"] = REFERENCE_TIME
        it["evidence"] = [table_evidence(official, d, f)
                          for d in g["gold"]
                          for f in SELECTION_EVIDENCE_FIELDS[qid]]
        it.pop("location", None)
        out[qid] = it
        report[qid] = {
            "condition": g["condition"],
            "question_before": before_q, "question_after": it["question"],
            "question_changed": before_q != it["question"],
            "candidates": g["candidate_count"],
            "removed_by_deadline": g["removed_by_deadline"],
            "passed_as_unknown_deadline": g["passed_as_unknown_deadline"],
            "gold_before": sorted(before_gold), "gold_after": g["gold"],
            "added": sorted(set(g["gold"]) - set(before_gold)),
            "removed": sorted(set(before_gold) - set(g["gold"])),
            "gold_changed": sorted(before_gold) != g["gold"],
            "n_evidence": len(it["evidence"]),
            "reference_time": REFERENCE_TIME,
        }
    return out, report


def main(out_dir: Path, official_items: Path, table_path=None, *,
         repo: Path | None = None, srv: Path | None = None) -> int:
    repo = repo or repo_root()
    srv = srv or data_root()
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))
    global REFERENCE_TIME, SELECTION_SPEC, compute_selection_gold, load_official
    from checks.selection_policy import (  # noqa: E402
        REFERENCE_TIME, SELECTION_SPEC, compute_selection_gold, load_official,
    )
    off = official_paths(srv)
    require("공식 등록부", off["registry"])
    require("공식 identity", off["identity"])
    require("공식 평가셋 items.jsonl", official_items)
    if table_path is not None:
        require("후보 추출표(--table)", table_path)
    # ★후보 추출표를 쓸 수 있게 한다 — 공식 표는 읽기만 하고, 정답은 후보 표로 재계산한다.
    official = (load_official(registry=off["registry"], identity=off["identity"],
                              table=table_path) if table_path
                else load_official(registry=off["registry"], identity=off["identity"]))
    # ★근거의 source 라벨은 **실제로 읽은 표**의 버전이어야 한다. 후보 표로 정답을
    #   재계산하면서 라벨만 v3 로 남기면, 근거가 공식 표에서 왔다는 거짓 기록이 된다.
    global TABLE_VERSION
    meta = json.loads(Path(table_path or off["table"]).read_text(encoding="utf-8"))
    TABLE_VERSION = str(meta.get("extraction_version") or TABLE_VERSION)
    items = {}
    for line in official_items.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items[r["id"]] = r
    sel, sel_report = build_selection(items, official)
    others, other_report = build_others(items, official)
    merged = {**items, **sel, **others}
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / "candidate"
    cand.mkdir(parents=True, exist_ok=True)
    with (cand / "items.jsonl").open("w", encoding="utf-8") as f:
        for qid in sorted(merged, key=lambda k: (k.split("-")[0], int(k.split("-")[1]))):
            f.write(json.dumps(merged[qid], ensure_ascii=False) + "\n")
    # ★명세가 요구한 근거를 그대로 기록한다 — 검사기(S16)가 생성 결과에서
    #   명시적 EVIDENCE 가 사라졌는지 독립적으로 확인할 수 있어야 한다.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from items_v2_spec import EVIDENCE as SPEC_EVIDENCE
    (out_dir / "spec_evidence.json").write_text(json.dumps(
        {"note": "items_v2_spec.EVIDENCE — 생성 결과에 반드시 남아야 하는 명시적 근거",
         "evidence": {q: [{k: v for k, v in e.items() if k in
                           ("kind", "document", "field", "fact_field")}
                          for e in evs] for q, evs in SPEC_EVIDENCE.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "changes_v2.json").write_text(json.dumps(
        {"reference_time": REFERENCE_TIME, "selection": sel_report, "others": other_report},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'문항':<9}{'이전':>5}{'이후':>5}  {'질문변경':<6} 조건")
    for qid in sorted(sel_report):
        r = sel_report[qid]
        print(f"{qid:<9}{len(r['gold_before']):>5}{len(r['gold_after']):>5}"
              f"  {'예' if r['question_changed'] else '-':<6} {r['condition'][:52]}")
    print("\n추출형·QA 변경:")
    for qid in sorted(other_report):
        print(f"  {qid}: {'; '.join(other_report[qid]['changes'])}")
    print(f"\n총 {len(merged)}문항 → {cand/'items.jsonl'}")
    return 0




# ── §3 · §4 추출형·QA 문항 ──────────────────────────────────────────

def _fill_evidence(ev: dict, official: dict) -> dict:
    """명세의 근거에 공식 자료에서 읽은 값(status/좌표/값)을 채운다."""
    ev = dict(ev)
    if ev["kind"] == "extraction_table":
        full = table_evidence(official, ev["document"], ev["field"])
        full.update({k: v for k, v in ev.items() if k not in ("kind",)})
        full.setdefault("kind", "extraction_table")
        row = (official["table"].get(ev["document"]) or {}).get(ev["field"]) or {}
        full["status"] = row.get("status", "missing")
        return full
    if ev["kind"] == "identity":
        row = official["identity"].get(ev["document"], {})
        ev.setdefault("value", row.get(ev["field"]))
        ev.setdefault("source", f"identity_{IDENTITY_VERSION}")
        return ev
    ev.setdefault("source", "chunks_v3")
    return ev  # chunk 근거 — fact_field 등 명세가 적은 키는 그대로 보존한다


def build_others(items: dict, official: dict) -> tuple[dict, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from items_v2_spec import (ANSWER_SOURCE_FIXES, BLOCKED, EVIDENCE, ITEM_FIXES)
    out, report = {}, {}
    for qid, it in items.items():
        if it.get("task_type") == "selection":
            continue
        new = dict(it)
        changes = []
        fix = ITEM_FIXES.get(qid)
        if fix:
            if "type" in fix and fix["type"] != new.get("answer_type"):
                changes.append(f"answer_type {new.get('answer_type')} → {fix['type']}")
                new["answer_type"] = fix["type"]
            if "raw" in fix:
                changes.append("answer_raw 재작성")
                new["answer_raw"] = fix["raw"]
                new.pop("answer_normalized", None)
            if "cells" in fix:
                rows = [dict(r) for r in (new.get("answer_raw") or [])]
                for (field, doc), value in fix["cells"].items():
                    for r in rows:
                        if r.get("항목") == field and doc in r:
                            changes.append(f"비교 셀 정리: {field}/{doc}")
                            r[doc] = value
                new["answer_raw"] = rows
            if fix.get("drop_unspecified_type") and new.pop("unspecified_type", None):
                changes.append("unspecified_type 제거(되묻기가 정답이 아님)")
        if fix and fix.get("answer_source") and fix["answer_source"] != new.get("answer_source"):
            changes.append(f"answer_source {new.get('answer_source')} → {fix['answer_source']}")
            new["answer_source"] = fix["answer_source"]
        if fix and fix.get("drop_location") and new.pop("location", None) is not None:
            changes.append("오분류 청크 좌표(location) 제거")
        if qid in ANSWER_SOURCE_FIXES:
            before = new.get("answer_source")
            new["answer_source"] = ANSWER_SOURCE_FIXES[qid]
            changes.append(f"answer_source {before} → {new['answer_source']}")
        if qid in EVIDENCE:
            new["evidence"] = [_fill_evidence(e, official) for e in EVIDENCE[qid]]
            changes.append(f"근거 {len(new['evidence'])}건 부착")
        # ★추출표 기반 value 문항: 연결된 추출표 행의 원문 값을 **허용 답 목록**에 넣는다.
        #   모델이 공식 추출표 값을 그대로 돌려줘도 정답이 되게 하되, 목록에 없는
        #   덧붙임(다른 금액·임의 설명)은 여전히 오답이다. 문항 ID 하드코딩이 아니라
        #   answer_source=table + extraction_table 근거를 가진 모든 value 문항에 적용.
        if new.get("answer_type") == "value" and new.get("answer_source") == "table" \
                and isinstance(new.get("answer_raw"), str):
            accepted = []
            for e in (new.get("evidence") or []):
                if e.get("kind") != "extraction_table" or e.get("status") != "value_present":
                    continue
                row = (official["table"].get(e["document"]) or {}).get(e["field"]) or {}
                for v in (row.get("answer_raw"), row.get("answer_normalized")):
                    if isinstance(v, str) and v.strip() and v != new["answer_raw"] and v not in accepted:
                        accepted.append(v)
            if accepted:
                new["answer_normalized"] = accepted
                changes.append(f"허용 답 목록에 추출표 값 {len(accepted)}건")
        # ★[2026-09-04 결함 수정] 명세(items_v2_spec.EVIDENCE)에 근거를 명시한 문항은
        #   그 근거가 **최우선**이다. 예전에는 이 분기가 위 조건과 elif 로 이어져 있어서
        #   answer_source 가 table 이 아닌 문항(EXT-07)은 바로 아래로 흘러 내려와
        #   방금 붙인 명세 근거(identity + 원문 청크)를 최상위 location 하나로 덮어썼다.
        #   그 결과 EXT-07 은 근거가 kind=chunk / source=chunks_v3 / 위치 "CSV: bid_deadline"
        #   한 건으로 바뀌고 identity 근거가 사라졌다(실제 재현).
        #   이제 명시적 EVIDENCE 가 없는 문항에서만 기존 location 을 청크 근거로 바꾼다.
        if qid not in EVIDENCE and new.get("location"):
            loc = new["location"]
            locs = loc if isinstance(loc, list) else [loc]
            new["evidence"] = [chunk_evidence(l) for l in locs if isinstance(l, dict)]
            changes.append(f"기존 좌표를 근거로 정규화({len(new['evidence'])}건)")
        if qid in BLOCKED:
            new["blocked_reason"] = BLOCKED[qid]
            changes.append("차단 항목 표시")
        elif new.pop("blocked_reason", None) is not None:
            changes.append("차단 해제(blocked_reason 제거)")
        out[qid] = new
        if changes:
            report[qid] = {"changes": changes,
                           "note": (fix or {}).get("note"),
                           "blocked": BLOCKED.get(qid)}
    return out, report


def _cli() -> int:
    ap = argparse.ArgumentParser(description="후보 평가셋 생성(공식 자료는 읽기만 한다)")
    ap.add_argument("--out", required=True, help="산출 폴더(candidate/items.jsonl 등이 생긴다)")
    ap.add_argument("--official-items", default=None,
                    help="공식 평가셋 items.jsonl(기본: <data-root>/evalset/v1/items.jsonl)")
    add_path_args(ap)
    a = ap.parse_args()
    repo, srv = repo_root(a.repo_root), data_root(a.data_root)
    items = Path(a.official_items) if a.official_items else official_paths(srv)["official_items"]
    return main(Path(a.out), require("공식 평가셋 items.jsonl", items),
                Path(a.table) if a.table else None, repo=repo, srv=srv)


if __name__ == "__main__":
    sys.exit(_cli())
