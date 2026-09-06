"""§12 검색·출처 결함 주입 — 채점기가 실제로 잡아내는지 확인한다.

평가셋(문항)이 아니라 **모델 응답**을 일부러 망가뜨린다. 각 결함마다
  · 정상 응답이 통과하는지(기준선)
  · 망가뜨린 응답이 걸리는지(검색 실패 종류·출처 점수·잘못된 인용 수)
를 같은 채점 경로(grade_retrieval / grade_citation)로 확인한다.

사용:  python tools/evalset/inject_defects_citation.py --out <결과.json> [--repo-root R]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_path_args, repo_root  # noqa: E402

DOC, DOC2 = "RFP-000014", "RFP-000020"
LOC_A = {"document": DOC, "section": "1. 개요", "ref_no": "1. 개요 · 문단 1", "line": 10}
LOC_B = {"document": DOC, "section": "2. 절차", "ref_no": "2. 절차 · 문단 2", "line": 20}
LOC_OTHER = {"document": DOC2, "section": "1. 개요", "ref_no": "1. 개요 · 문단 1", "line": 10}

CHUNK_A = {"kind": "chunk", "document": DOC, "fact_field": "예산",
           "source": "chunks_v3", "location": LOC_A}
CHUNK_B = {"kind": "chunk", "document": DOC, "fact_field": "사업기간",
           "source": "chunks_v3", "location": LOC_B}
TABLE_A = {"kind": "extraction_table", "document": DOC, "field": "예산",
           "fact_field": "예산", "status": "value_present",
           "source": "extraction_table_v3", "location": LOC_A}
ABSENT = {"kind": "extraction_table", "document": DOC, "field": "지역제한",
          "status": "field_absent", "source": "extraction_table_v3"}
IDENT = {"kind": "identity", "document": DOC, "field": "bid_deadline",
         "fact_field": "bid_deadline", "source": "identity_v2"}


def _item(evidence, answer_type="value", answer_source="verified"):
    from grader.models import EvaluationItem
    return EvaluationItem.model_validate(dict(
        id="INJ", question="q", task_type="qa", answer_type=answer_type,
        answer_source=answer_source, answer_raw="x", evidence=evidence))


def _resp(citations=(), retrieved=(), contexts=()):
    from grader.models import ModelResponse
    hit = lambda locs: [{"document_id": l["document"], "location": l} for l in locs]
    return ModelResponse.model_validate({
        "id": "INJ", "answer": "a", "citations": list(citations),
        "retrieved": hit(retrieved), "contexts": hit(contexts)})


def _grade(item, resp):
    from grader.retrieval import grade_citation, grade_retrieval
    stages = grade_retrieval(item, resp, 5, None, 5)
    return {"retrieval": stages[0], "citation": grade_citation(item, resp)}


# (키, 설명, 정상 응답, 망가뜨린 응답, 잡혔다고 볼 조건)
def _cases():
    two_chunk = _item([CHUNK_A, CHUNK_B])
    alt = _item([TABLE_A, CHUNK_A])            # 같은 사실의 두 출처
    absent_item = _item([ABSENT], answer_source="table")
    ident_item = _item([IDENT, CHUNK_A], answer_source="metadata")
    C = lambda **kw: _resp(**kw)
    caught_recall = lambda g: g["retrieval"].get("failure_kind") == "recall_failure"
    caught_rank = lambda g: g["retrieval"].get("failure_kind") == "rank_failure"
    # 옛 코드에는 사실 묶음 지표가 없다 — get 으로 읽어 "지표 자체가 없음"도 미차단으로 센다.
    partial = lambda g: (g["citation"].get("n_missing_facts") or 0) > 0 and g["citation"].get("score", 1.0) < 1.0
    wrong = lambda g: (g["citation"].get("n_wrong_citations") or 0) > 0
    none_cite = lambda g: g["citation"].get("n_citations") == 0 and g["citation"].get("score") == 0.0
    L = lambda f: f          # 지연 생성 — 옛 응답 계약에서 실패하는 것도 결과로 남긴다
    return [
        ("R01", "정답 청크가 검색 후보에 전혀 없음", lambda: two_chunk,
         lambda: C(retrieved=[LOC_A, LOC_B], contexts=[LOC_A, LOC_B], citations=[LOC_A, LOC_B]),
         lambda: C(retrieved=[LOC_OTHER], contexts=[LOC_OTHER], citations=[LOC_A, LOC_B]),
         caught_recall),
        ("R02", "후보에는 있지만 최종 컨텍스트에는 없음", lambda: two_chunk,
         lambda: C(retrieved=[LOC_A, LOC_B], contexts=[LOC_A, LOC_B], citations=[LOC_A, LOC_B]),
         lambda: C(retrieved=[LOC_A, LOC_B], contexts=[], citations=[LOC_A, LOC_B]),
         caught_rank),
        ("R03", "다중 근거 중 하나만 검색", lambda: two_chunk,
         lambda: C(retrieved=[LOC_A, LOC_B], contexts=[LOC_A, LOC_B], citations=[LOC_A, LOC_B]),
         lambda: C(retrieved=[LOC_A], contexts=[LOC_A], citations=[LOC_A, LOC_B]),
         caught_recall),
        ("C01", "다중 근거 중 하나만 인용", lambda: two_chunk,
         lambda: C(citations=[LOC_A, LOC_B]), lambda: C(citations=[LOC_A]), partial),
        ("C02", "전혀 다른 문서 인용", lambda: two_chunk,
         lambda: C(citations=[LOC_A, LOC_B]), lambda: C(citations=[LOC_OTHER]),
         lambda g: wrong(g) and g["citation"].get("score") == 0.0),
        ("C03", "같은 문서의 다른 필드 인용", lambda: absent_item,
         lambda: C(citations=[dict(ABSENT)]),
         lambda: C(citations=[dict(ABSENT, field="예산")]), wrong),
        ("C04", "같은 필드지만 다른 문서 인용", lambda: absent_item,
         lambda: C(citations=[dict(ABSENT)]),
         lambda: C(citations=[dict(ABSENT, document=DOC2)]), wrong),
        ("C05", "field_absent 인데 value_present 라고 인용", lambda: absent_item,
         lambda: C(citations=[dict(ABSENT)]),
         lambda: C(citations=[dict(ABSENT, status="value_present")]), wrong),
        ("C06", "field_absent 근거에 가짜 줄 번호 추가", lambda: absent_item,
         lambda: C(citations=[dict(ABSENT)]),
         lambda: C(citations=[{"document": DOC, "section": "", "ref_no": "지역제한", "line": 42}]),
         wrong),
        ("C07", "동일 사실의 추출표·청크 중 하나만 인용(정상이어야 함)", lambda: alt,
         lambda: C(citations=[dict(TABLE_A)]), lambda: C(citations=[LOC_A]),
         lambda g: g["citation"].get("score") == 1.0 and g["citation"].get("n_fact_bundles") == 1),
        ("C08", "서로 다른 두 사실 중 하나만 인용", lambda: two_chunk,
         lambda: C(citations=[LOC_A, LOC_B]), lambda: C(citations=[LOC_A]),
         lambda g: g["citation"].get("n_fact_bundles") == 2 and g["citation"].get("n_missing_facts") == 1),
        ("C09", "인용 없음", two_chunk, lambda: C(citations=[LOC_A, LOC_B]), lambda: C(citations=[]), none_cite),
        ("C10", "검색 안 한 추출형에서 추출표 근거 누락", lambda: absent_item,
         lambda: C(citations=[dict(ABSENT)]), lambda: C(citations=[]), none_cite),
        ("C11", "metadata 답변에 chunks 출처를 잘못 붙임", lambda: ident_item,
         lambda: C(citations=[dict(IDENT), LOC_A]),
         lambda: C(citations=[{"document": DOC, "section": "CSV", "ref_no": "CSV: bid_deadline",
                       "source": "chunks_v3"}]),
         lambda g: wrong(g) and (g["citation"].get("n_missing_facts") or 0) > 0),
        ("C12", "문서 ID만 같은 출처를 정답 처리하려는 경우", lambda: two_chunk,
         lambda: C(citations=[LOC_A, LOC_B]),
         lambda: C(citations=[{"document": DOC, "section": "9장", "ref_no": "9장 · 문단 9", "line": 900}]),
         lambda g: wrong(g) and g["citation"].get("score") == 0.0),
        ("C13", "정상 응답이 통과하는지(기준선)", lambda: two_chunk,
         lambda: C(retrieved=[LOC_A, LOC_B], contexts=[LOC_A, LOC_B], citations=[LOC_A, LOC_B]),
         lambda: C(retrieved=[LOC_A, LOC_B], contexts=[LOC_A, LOC_B], citations=[LOC_A, LOC_B]),
         lambda g: g["citation"].get("score") == 1.0 and g["retrieval"].get("failure_kind") == "none"),
    ]


def main(out: Path, repo: Path) -> int:
    sys.path.insert(0, str(repo / "src"))
    rows = []
    for key, title, item_fn, good_fn, bad_fn, caught in _cases():
        try:
            item = item_fn() if callable(item_fn) else item_fn
            good = good_fn() if callable(good_fn) else good_fn
            bad = bad_fn() if callable(bad_fn) else bad_fn
            g_ok, g_bad = _grade(item, good), _grade(item, bad)
        except Exception as exc:            # 옛 코드에서 응답 계약이 근거형 인용을 못 받는 경우
            rows.append({"key": key, "defect": title, "baseline_clean": False,
                         "blocked": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"★NG {key} {title}: {rows[-1]['error']}")
            continue
        # 인용만 보는 결함은 검색 단계 값을 기준선에 넣지 않는다(응답에 후보가 없다)
        retrieval_relevant = bool(good.retrieved or good.contexts)
        baseline_clean = (g_ok["citation"].get("score") == 1.0
                          and (not retrieval_relevant
                               or g_ok["retrieval"].get("failure_kind") in (None, "none")))
        try:
            blocked = bool(caught(g_bad))
        except Exception:
            blocked = False
        rows.append({
            "key": key, "defect": title,
            "baseline_clean": baseline_clean, "blocked": blocked,
            "before_fix_hint": g_ok["citation"].get("score"),
            "graded_bad": {"citation_score": g_bad["citation"].get("score"),
                           "n_fact_bundles": g_bad["citation"].get("n_fact_bundles"),
                           "n_missing_facts": g_bad["citation"].get("n_missing_facts"),
                           "n_wrong_citations": g_bad["citation"].get("n_wrong_citations"),
                           "failure_kind": g_bad["retrieval"].get("failure_kind")},
        })
        print(f"{'OK ' if blocked else '★NG'} {key} {title}: {rows[-1]['graded_bad']}")
    n, ok = len(rows), sum(1 for r in rows if r["blocked"])
    clean = sum(1 for r in rows if r["baseline_clean"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "blocked": ok, "baseline_clean": clean, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n합성 결함 {n}종 중 {ok}종 차단 (정상 응답 기준선 통과 {clean}/{n})")
    return 0 if ok == n else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="검색·출처 결함 주입 검사")
    ap.add_argument("--out", required=True)
    add_path_args(ap)
    a = ap.parse_args()
    sys.exit(main(Path(a.out), repo_root(a.repo_root)))
