"""
grader.retrieval — 3-2 검색 평가

근거:
  3-2   무엇을 '찾았다'로 인정할 것인가 / 지표 후보 / k / 종합형 채점
  2-9   근거의 단위·정밀도 = document + section + ref_no (임현진/김태윤 확정)
  4-9-3 k 값 통일 [논의 ⇄ 이태민] — ★최종 k는 아직 확정하지 않는다(2026-08-27 기준)
  1-13-1/4-7 표 청킹 정책(1000자 이하 통째 유지, 초과 시 header 복제 분할)과의 연결은
        이태민의 실제 청킹 구조가 들어오면 chunk 단위 좌표 매칭 규칙을 추가한다.

★ 핵심 설계 (D11·D2): 검색 실패를 두 종류로 가른다.
   - 재현율 실패 : 정답 근거가 후보 풀(retrieval_k) 안에도 없다
                   → 처방은 임베딩 교체 / 하이브리드 / 파싱 개선
   - 순위 실패   : 후보 풀 안에는 있는데 실제 컨텍스트(context_k)에 들어가는
                   상위 reranker_k 에 못 든다
                   → 처방은 재정렬(reranker)
   이 구분이 없으면 재정렬을 붙여놓고 왜 효과가 없는지 모르게 된다.

★ retrieval_k / reranker_k / context_k 는 서로 다른 단계다. 태민의 실제 검색 구조를 보기
  전까지 이 셋을 하나의 "k"로 뭉치지 않는다 — 뭉치면 어느 단계에서 실패했는지 못 가른다.
"""

from __future__ import annotations

from .models import EvaluationItem, Location, ModelResponse, RetrievedItem
from .normalize import match_location

# 3-2 매칭 정밀도. 2-9 확정본 = document+section+ref_no → 기본값도 ref_no 로 맞춘다.
# grade_retrieval(검색 recall/precision/MRR) 과 grade_citation(출처 정확도) 이 같은 단위를 쓴다.
DEFAULT_PRECISION = "ref_no"


def _gold_locations(item: EvaluationItem) -> list[Location]:
    return [item.location] if item.location is not None else []


def _found_ranks(golds: list[Location], cands: list[RetrievedItem], precision: str) -> list[int]:
    """각 정답 근거가 후보 목록에서 몇 위에 있는지(1-base). 없으면 -1."""
    ranks = []
    for g in golds:
        r = -1
        for i, c in enumerate(cands, 1):
            if c.location is not None and match_location(g, c.location, precision):
                r = i
                break
        ranks.append(r)
    return ranks


def grade_stage(golds: list[Location], candidates: list[RetrievedItem],
                k: int, precision: str = DEFAULT_PRECISION, stage: str = "retrieval_k") -> dict:
    """검색 파이프라인 한 단계(retrieval_k/reranker_k/context_k)에 대한 지표.

    ★같은 함수로 세 단계를 각각 잰다 — k를 하나로 확정하기 전에 단계별 비교가 가능해야 한다.
    """
    if not golds:
        return {"applicable": False, "stage": stage,
                "reason": "정답 근거 없음(location 미기록) — 검색 지표 계산 불가"}

    topk = candidates[:k] if k else candidates
    ranks_topk = _found_ranks(golds, topk, precision)
    found_topk = [r for r in ranks_topk if r > 0]
    recall_at_k = len(found_topk) / len(golds)
    hits_in_topk = sum(
        1 for c in topk
        if c.location is not None and any(match_location(g, c.location, precision) for g in golds)
    )
    precision_at_k = hits_in_topk / max(1, len(topk))
    mrr = max([1.0 / r for r in ranks_topk if r > 0], default=0.0)

    return {
        "applicable": True,
        "stage": stage,
        "n_gold": len(golds),
        "recall_at_k": recall_at_k,
        "recall_all": recall_at_k == 1.0,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "k": k,
        "precision_unit": precision,
    }


def grade_retrieval(item: EvaluationItem, response: ModelResponse,
                    retrieval_k: int, reranker_k: int, context_k: int,
                    precision: str = DEFAULT_PRECISION) -> list[dict]:
    """문항 하나의 검색 채점 — retrieval_k(후보 풀) / reranker_k(재정렬 후) /
    context_k(실제 LLM 입력) 세 단계를 각각 낸다.

    failure_kind 는 세 단계를 종합해 판정한다:
      recall_failure : 정답 근거가 retrieval_k 후보 풀에도 없음 → 재정렬은 무용
      rank_failure   : 후보 풀에는 있는데 context(실제 입력)까지 못 옴 → 재정렬(H)의 자리
      none           : context 안에 정답 근거가 있음
    """
    golds = _gold_locations(item)
    pool = response.retrieved
    reranked = response.reranked or response.retrieved
    ctx_items = [RetrievedItem(document_id=(c.document_id or ""), location=c.location)
                 for c in response.contexts]

    stages = [
        grade_stage(golds, pool, retrieval_k, precision, "retrieval_k"),
        grade_stage(golds, reranked, reranker_k, precision, "reranker_k"),
        grade_stage(golds, ctx_items, context_k, precision, "context_k"),
    ]
    if not golds:
        return stages

    in_pool = any(r > 0 for r in _found_ranks(golds, pool[:retrieval_k] if retrieval_k else pool, precision))
    in_ctx = any(r > 0 for r in _found_ranks(golds, ctx_items, precision))
    if in_ctx:
        failure_kind = "none"
    elif in_pool:
        failure_kind = "rank_failure"
    else:
        failure_kind = "recall_failure"
    for s in stages:
        s["failure_kind"] = failure_kind
    return stages


# ------------------------------------------------------------------ 3-4-3 출처 좌표 채점

def grade_citation(item: EvaluationItem, response: ModelResponse,
                   precision: str = DEFAULT_PRECISION, enabled: bool = True) -> dict:
    """3-4-3 출처 좌표 채점 — 정답 location과 응답 citations의 좌표가 실제로
    일치하는지를 채점한다.

    ★기존에는 citation 유무만 보고 그마저도 형식(FAIL) 판정에 안 썼다
      (task_scoring.check_format 참고 — 미표기를 FAIL로 만들지 않는다는 결정은 유지).
      "citation이 있다"와 "citation이 맞다"는 다른 질문이다 — 여기서 후자를 낸다.
    ★grade_retrieval과 같은 match_location/precision 규칙을 그대로 쓴다 — 좌표
      정밀도 기준을 검색용과 인용용으로 이원화하지 않는다(2-9 확정 단위를 그대로 재사용).
    """
    if not enabled:
        return {"applicable": False, "reason": "grading.grade_citations=false"}
    if item.location is None:
        return {"applicable": False, "reason": "정답 location 없음 — citation 채점 대상 아님"}
    if not response.citations:
        # ★형식 위반이 아니다 — check_format은 이걸로 FAIL을 만들지 않는다. 여기서만
        # score=0으로 반영해 진단(citation_accuracy/no_citation_rate)에 잡히게 한다.
        return {"applicable": True, "matched": False, "score": 0.0, "n_citations": 0,
                "precision_unit": precision, "reason": "citation 없음(출처 미표기)"}
    matched = any(match_location(item.location, c, precision) for c in response.citations)
    return {"applicable": True, "matched": matched, "score": 1.0 if matched else 0.0,
            "n_citations": len(response.citations), "precision_unit": precision}


def aggregate_citation(per_item: list[dict]) -> dict:
    """3-4-3 집계 — citation_accuracy(좌표가 실제로 맞았는지)와 no_citation_rate
    (아예 안 붙였는지)를 분리해서 낸다. 하나로 합치면 "안 붙임"과 "틀리게 붙임"이
    같은 숫자로 섞여 원인을 못 가른다."""
    rows = [r for r in per_item if r.get("applicable")]
    n = len(rows)
    if not n:
        return {"n": 0, "note": "정답 location 있는 문항 없음(또는 grade_citations=false) — "
                                "citation 채점 대상 없음"}
    matched = sum(1 for r in rows if r.get("matched"))
    no_citation = sum(1 for r in rows if r.get("n_citations") == 0)
    return {
        "n": n,
        "citation_accuracy": round(matched / n, 4),
        "no_citation_rate": round(no_citation / n, 4),
        "note": "3-4-3 출처 좌표 채점 — 정답 위치 vs 인용 좌표 일치율. "
                "check_format의 FAIL 판정 대상이 아니라 진단 지표다.",
    }


def aggregate_retrieval(per_item_stages: list[list[dict]]) -> dict:
    """3-2 집계. 단계별로 나눠 낸다 — 하나로 합치면 어느 단계가 문제인지 사라진다."""
    by_stage: dict[str, list[dict]] = {"retrieval_k": [], "reranker_k": [], "context_k": []}
    for stages in per_item_stages:
        for s in stages:
            if s.get("applicable"):
                by_stage.setdefault(s["stage"], []).append(s)

    out: dict[str, dict] = {}
    for stage, rows in by_stage.items():
        if not rows:
            out[stage] = {"n": 0}
            continue
        n = len(rows)
        kinds = {"none": 0, "rank_failure": 0, "recall_failure": 0}
        for r in rows:
            kinds[r.get("failure_kind", "none")] += 1
        out[stage] = {
            "n": n,
            "recall@k": sum(r["recall_at_k"] for r in rows) / n,
            "recall_all_rate": sum(1 for r in rows if r["recall_all"]) / n,
            "precision@k": sum(r["precision_at_k"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n,
            "failure_breakdown": kinds,
        }

    ctx_rows = by_stage.get("context_k", [])
    n = len(ctx_rows)
    kinds = {"none": 0, "rank_failure": 0, "recall_failure": 0}
    for r in ctx_rows:
        kinds[r.get("failure_kind", "none")] += 1
    if n:
        out["failure_rate"] = {
            "recall_failure": kinds["recall_failure"] / n,
            "rank_failure": kinds["rank_failure"] / n,
        }
        out["prescription_hint"] = (
            "재현율 실패 우세 → 임베딩 교체·하이브리드·파싱 개선"
            if kinds["recall_failure"] >= kinds["rank_failure"]
            else "순위 실패 우세 → reranker 확인 우선 (4-9-6)"
        )
    out["note"] = "★최종 k는 미확정 — 세 단계를 각각 비교하고, 이태민의 실제 구조가 들어온 뒤 확정한다."
    return out
