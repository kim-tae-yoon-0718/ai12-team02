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

from .models import EvaluationItem, Location, ModelResponse, RetrievedItem, _strip_part
from .normalize import match_location

# 3-2 매칭 정밀도. 2-9 확정본 = document+section+ref_no → 기본값도 ref_no 로 맞춘다.
# grade_retrieval(검색 recall/precision/MRR) 과 grade_citation(출처 정확도) 이 같은 단위를 쓴다.
DEFAULT_PRECISION = "ref_no"


def _ev_get(ev, key):
    """근거·인용이 Evidence/Location 객체든 dict 든 같은 방식으로 읽는다.

    pydantic 모델은 extra="allow" 라 선언에 없는 키도 실려 올 수 있다
    (실제 파이프라인이 citation 에 field·source·kind 를 함께 싣는다).
    """
    if isinstance(ev, dict):
        return ev.get(key)
    v = getattr(ev, key, None)
    if v is None:
        extra = getattr(ev, "model_extra", None)
        if isinstance(extra, dict):
            return extra.get(key)
    return v


def _evidence_chunk_locations(item: EvaluationItem) -> list[Location]:
    """근거 배열 중 **실제 원문 좌표가 있는 청크 근거**의 좌표만 모은다.

    ★추출표 행·identity 값 자체는 벡터 검색의 정답 위치가 아니다(원문 줄이 없다).
      그래서 kind=chunk 이고 location 이 실재하는 근거만 검색 정답 좌표로 쓴다.
    """
    out: list[Location] = []
    for ev in (getattr(item, "evidence", None) or []):
        if _ev_get(ev, "kind") != "chunk":
            continue
        loc = _ev_get(ev, "location")
        if loc is None:
            continue
        if isinstance(loc, dict):
            try:
                loc = Location.model_validate(loc)
            except Exception:
                continue
        if getattr(loc, "ref_no", None) and loc not in out:
            out.append(loc)
    return out


def _gold_locations(item: EvaluationItem) -> list[Location]:
    """좌표 매칭용 정답 좌표. 비교형은 문서별 배열(2-8-4).

    순서(2026-09-04 §5 확정)
      ① evidence 의 kind=chunk 근거에 실제 좌표가 있으면 그것을 쓴다.
         — 근거가 새 evidence 배열에만 들어 있는 문항(QA-007·QA-008)이 예전에는
           "정답 위치 없음"으로 검색 채점에서 통째로 빠졌다(실측).
      ② 그런 근거가 없을 때만 기존 최상위 location 으로 되돌아간다.
      ③ answer_source=metadata 의 최상위 좌표(section="CSV")는 본문 블록이 아니므로
         폴백 대상에서 뺀다 — 다만 ①의 실제 청크 근거는 metadata 문항이어도 쓴다.
    """
    ev_locs = _evidence_chunk_locations(item)
    if ev_locs:
        return ev_locs
    if item.answer_source == "metadata":
        return []
    return item.gold_locations()


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


def _search_not_used(response: ModelResponse, non_search_routes) -> str | None:
    """검색을 쓰지 않은 문항이면 이유 문자열, 썼으면 None (팀장).
    청크 검색 경로로 푼 문항만 검색 점수를 계산한다."""
    r = (response.route or "").strip().lower()
    if r and non_search_routes and r in {x.lower() for x in non_search_routes}:
        return f"route={response.route} — 검색을 사용하지 않은 문항(추출표·identity 경로)"
    return None


def grade_retrieval(item: EvaluationItem, response: ModelResponse,
                    retrieval_k: int, reranker_k: int | None, context_k: int,
                    precision: str = DEFAULT_PRECISION,
                    eval_k: tuple[int, ...] = (3, 5),
                    non_search_routes=frozenset()) -> list[dict]:
    """문항 하나의 검색 채점 — retrieval_k(후보 풀) / reranker_k(재정렬 후) /
    context_k(실제 LLM 입력) 세 단계를 각각 낸다.

    eval_k: top_k=5(이태민 확정) 결과를 채점 때 k=3/5 등으로 잘라 별도 기록한다
      (김하루 협의, base.yaml top_k 주석). retrieval_k 단계에 `by_k` 로 붙는다.

    failure_kind 는 세 단계를 종합해 판정한다:
      recall_failure : 정답 근거가 retrieval_k 후보 풀에도 없음 → 재정렬은 무용
      rank_failure   : 후보 풀에는 있는데 context(실제 입력)까지 못 옴 → 재정렬(H)의 자리
      none           : context 안에 정답 근거가 있음
    """
    skip = _search_not_used(response, non_search_routes)
    if skip:
        return [{"applicable": False, "stage": s, "reason": skip}
                for s in ("retrieval_k", "reranker_k", "context_k")]

    golds = _gold_locations(item)
    pool = response.retrieved
    ctx_items = [RetrievedItem(document_id=(c.document_id or ""), location=c.location)
                 for c in response.contexts]

    # ⚠️ [2026-09-04 정정] 예전에는 `reranked = response.reranked or response.retrieved` 로
    #    재정렬 결과가 없으면 후보 풀을 그대로 복사해 "재정렬을 한 것처럼" 채점했다.
    #    베이스라인에는 리랭커가 없으므로 그 점수는 실제로 존재하지 않는 단계의 점수였다.
    #    이제 재정렬 결과가 없거나 reranker_k 가 없으면 **미적용(N/A)** 으로 기록한다.
    rerank_applied = bool(response.reranked) and bool(reranker_k)
    if rerank_applied:
        rerank_stage = grade_stage(golds, response.reranked, reranker_k, precision, "reranker_k")
    else:
        rerank_stage = {
            "applicable": False, "stage": "reranker_k",
            "reason": ("재정렬 미적용 — "
                       + ("응답에 reranked 가 없음" if not response.reranked
                          else "설정에 reranker_k 가 없음")),
        }

    stages = [
        grade_stage(golds, pool, retrieval_k, precision, "retrieval_k"),
        rerank_stage,
        grade_stage(golds, ctx_items, context_k, precision, "context_k"),
    ]
    if not golds:
        return stages

    # 다단계 k 분석 (k=3, k=5 …) — 후보 풀을 각 k로 잘라 recall/precision/mrr
    stages[0]["by_k"] = {
        str(kk): {m: grade_stage(golds, pool, kk, precision, "retrieval_k")[m]
                  for m in ("recall_at_k", "precision_at_k", "mrr")}
        for kk in eval_k
    }

    # ★[2026-09-04 §6 정정] 예전에는 정답 근거 중 **하나라도** 들어 있으면 성공으로
    #   봤다(any). 그러면 근거가 둘인 문항에서 하나만 검색돼도 failure_kind="none" 이
    #   되어 부분 누락이 성공으로 집계된다(실측). 다중 근거는 **전부** 있어야 한다.
    #     · 필요한 근거가 모두 최종 컨텍스트에 있음        → none
    #     · 하나라도 최초 후보(retrieval_k)에 없음          → recall_failure
    #     · 후보에는 다 있는데 일부가 컨텍스트에 없음        → rank_failure
    pool_ranks = _found_ranks(golds, pool[:retrieval_k] if retrieval_k else pool, precision)
    ctx_ranks = _found_ranks(golds, ctx_items, precision)
    n_pool = sum(1 for r in pool_ranks if r > 0)
    n_ctx = sum(1 for r in ctx_ranks if r > 0)
    if n_ctx == len(golds):
        failure_kind = "none"
    elif n_pool < len(golds):
        failure_kind = "recall_failure"
    else:
        failure_kind = "rank_failure"
    for s in stages:
        s["failure_kind"] = failure_kind
        s["n_gold_total"] = len(golds)
        s["n_gold_in_pool"] = n_pool
        s["n_gold_in_context"] = n_ctx
    return stages


# ------------------------------------------------------------------ 3-4-3 출처 좌표 채점

# ────────────────────────────────────────── 4-근거: 출처 중립 정규화

def _loc_keys(loc, doc=None) -> set:
    """청크 좌표 하나를 비교 가능한 키로. (문서 단독 키는 약한 키다.)"""
    if loc is None:
        return set()
    d = _ev_get(loc, "document") or doc
    if not d:
        return set()
    return {("doc", d)}


def evidence_status(x) -> str | None:
    """근거·인용이 밝힌 추출표 상태(field_absent / value_present …). 없으면 None."""
    v = _ev_get(x, "status")
    return str(v) if v else None


def fact_key(ev) -> tuple:
    """같은 사실을 가리키는 근거를 하나로 묶는 키(3-4-3 사실 묶음).

    ★같은 (문서, 필드)를 추출표·identity·원문 청크로 각각 확인할 수 있으면 그건
      **대체 가능한 출처**다. 원소마다 필수 근거로 세면 같은 사실을 두 경로로 모두
      인용해야 만점이 되어 버린다(QA-007·QA-008·EXT-07 에서 실측).
    ★반대로 서로 다른 (문서, 필드)는 각각 별개의 사실이므로 모두 인용해야 한다 —
      비교형의 문서×필드, 선별형의 문서×조건 필드가 여기에 해당한다.
    ★필드명이 출처마다 다를 수 있어 명세가 표준 필드명(fact_field)을 적어 두면 그것을 쓴다.
    ★필드가 없는 청크 근거는 좌표마다 다른 사실이므로 좌표를 키에 넣는다 —
      문서 ID만 같은 근거를 한 묶음으로 뭉치지 않는다.
    """
    doc = _ev_get(ev, "document")
    field = _ev_get(ev, "fact_field") or _ev_get(ev, "field")
    if doc and field:
        return ("fact", doc, str(field))
    loc = _as_location(ev)
    if doc and loc is not None:
        return ("loc", doc, loc.key("ref_no"))
    return ("raw", doc, _ev_get(ev, "kind"))


def evidence_keys(ev) -> set:
    """정답 근거 하나를 '경로 중립' 키 집합으로 편다.

    ★같은 사실을 추출표 행으로 가리키든 청크 좌표로 가리키든 같은 근거다.
      그래서 추출표 근거는 (표, 문서, 필드) 키와 함께, 원문 위치가 있으면
      (청크, 문서, 절, 위치) 키도 같이 낸다. 어느 쪽으로 인용해도 맞는다.
    """
    kind = ev.kind if hasattr(ev, "kind") else ev.get("kind")
    doc = ev.document if hasattr(ev, "document") else ev.get("document")
    field = ev.field if hasattr(ev, "field") else ev.get("field")
    loc = ev.location if hasattr(ev, "location") else ev.get("location")
    keys = {("doc", doc)} if doc else set()
    if kind == "extraction_table" and doc and field:
        keys.add(("table", doc, field))
    if kind == "identity" and doc and field:
        keys.add(("identity", doc, field))
    return keys


def citation_keys(c) -> set:
    """모델이 낸 인용 하나를 같은 키 공간으로 편다.

    응답의 citation 은 Location(extra 허용)이라 field/source 를 함께 실어 보낸다
    — 실제 시스템 출력이 그렇게 나온다(source="extraction_table_v3(raw_location)").
    """
    doc = _ev_get(c, "document")
    field = _ev_get(c, "field")
    src = str(_ev_get(c, "source") or "")
    kind = _ev_get(c, "kind")    # Evidence 형 인용(위치 없는 추출표 근거)은 kind 를 싣는다
    keys = _loc_keys(c, doc)
    if doc and field:
        if kind == "identity" or (not kind and "identity" in src):
            keys.add(("identity", doc, field))
        elif kind == "chunk":
            pass                 # 청크 인용은 좌표로만 인정한다(문서+필드로 표 근거를 대신하지 않음)
        else:
            # 출처 표기가 없어도 (문서, 필드) 인용은 추출표 근거로 인정한다 —
            # 표기 방식 차이로 맞는 근거를 틀렸다고 하지 않는다.
            keys.add(("table", doc, field))
    return keys


def _as_location(x):
    """근거/인용에서 Location 을 꺼낸다(없으면 None)."""
    if x is None:
        return None
    loc = getattr(x, "location", None) if not isinstance(x, dict) else x.get("location")
    if loc is None and (getattr(x, "ref_no", None) or (isinstance(x, dict) and x.get("ref_no"))):
        loc = x
    if loc is None:
        return None
    if isinstance(loc, Location):
        return loc
    if isinstance(loc, dict):
        try:
            return Location.model_validate(loc)
        except Exception:
            return None
    return loc if hasattr(loc, "key") else None


def evidence_hit(gev, citation, precision: str) -> bool:
    """정답 근거 하나가 이 인용으로 충족되는가.

    ★두 경로를 모두 인정한다:
      ① 같은 원문 좌표를 가리킴 — 좌표 정밀도 규칙(match_location)을 그대로 쓴다.
         줄 범위 판정을 여기서 느슨하게 만들지 않는다.
      ② 같은 추출표 행 / identity 필드를 가리킴 — 원문 줄이 없거나(미기재 행),
         표기 경로가 달라도 같은 근거다.
    """
    # ★[2026-09-04 §8] 상태를 밝힌 인용이 정답 근거의 상태와 다르면 같은 근거가 아니다.
    #   "미기재(field_absent) 행"을 "값 있음(value_present)"이라고 인용하면 사실이 다르다.
    g_status, c_status = evidence_status(gev), evidence_status(citation)
    if g_status and c_status and g_status != c_status:
        return False
    # ★미기재 근거는 가리킬 원문 줄이 **없다**. 줄 번호를 붙여 온 인용은 없는 좌표를
    #   지어낸 것이므로 인정하지 않는다(빈 section·가짜 ref_no 로 위장 금지).
    if g_status == "field_absent" and _ev_get(gev, "location") is None:
        cloc = _as_location(citation)
        if cloc is not None and getattr(cloc, "line", None) is not None:
            return False
    strong_g = {k for k in evidence_keys(gev) if k[0] != "doc"}
    strong_c = {k for k in citation_keys(citation) if k[0] != "doc"}
    if strong_g & strong_c:
        return True
    gloc, cloc = _as_location(gev), _as_location(citation)
    if gloc is not None and cloc is not None:
        return match_location(gloc, cloc, precision)
    return False


def _gold_evidence(item: EvaluationItem) -> list:
    """문항의 근거 목록 — evidence 가 있으면 그걸, 없으면 location 을 청크 근거로."""
    ev = list(getattr(item, "evidence", None) or [])
    if ev:
        return ev
    return [{"kind": "chunk", "document": g.document, "location": g}
            for g in _gold_locations(item)]


def grade_citation(item: EvaluationItem, response: ModelResponse,
                   precision: str = DEFAULT_PRECISION, enabled: bool = True,
                   non_search_routes=frozenset()) -> dict:
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
    # ★검색 미사용 경로(추출표·identity)라고 출처 채점을 끄지 않는다 — 검색 단계
    #   점수(grade_retrieval)는 N/A 가 맞지만, 추출표·identity 로 답해도 "그 값이
    #   어느 문서 어느 좌표에서 나왔는지"는 여전히 채점 대상이다. 여기서 끄면
    #   비검색 경로 문항은 근거를 아무렇게나 붙여도 감점이 없어진다.
    search_used = _search_not_used(response, non_search_routes) is None

    golds = _gold_evidence(item)
    if not golds:
        return {"applicable": False, "search_used": search_used,
                "reason": "정답 근거 없음 — citation 채점 대상 아님"}
    if not response.citations:
        # ★형식 위반이 아니다 — check_format 은 이걸로 FAIL 을 만들지 않는다. 여기서만
        # score=0 으로 반영해 진단(citation_accuracy/no_citation_rate)에 잡히게 한다.
        n_facts0 = len({fact_key(g) for g in golds})
        return {"applicable": True, "matched": False, "score": 0.0, "n_citations": 0,
                "n_gold_evidence": len(golds), "n_fact_bundles": n_facts0,
                "n_matched_facts": 0, "n_missing_facts": n_facts0,
                "n_missing_evidence": n_facts0, "n_wrong_citations": 0,
                "precision_unit": precision,
                "search_used": search_used, "reason": "citation 없음(출처 미표기)"}

    per_gold = [(g, any(evidence_hit(g, c, precision) for c in response.citations))
                for g in golds]
    # ★[2026-09-04 §7] 같은 사실의 대체 가능한 출처를 하나의 묶음으로 센다.
    #   묶음 안에서는 올바른 출처 하나만 인용해도 그 사실의 근거를 충족한 것이고,
    #   서로 다른 사실(문서×필드)은 여전히 **모두** 인용해야 한다.
    bundles: dict[tuple, bool] = {}
    for g, ok in per_gold:
        k = fact_key(g)
        bundles[k] = bundles.get(k, False) or ok
    n_facts = len(bundles)
    hit = sum(1 for ok in bundles.values() if ok)
    score = hit / n_facts if n_facts else 0.0

    # 3-4-3: "일부 근거 누락"(정답 위치인데 인용 안 함) vs "잘못된 근거"(인용했는데 아무
    # 정답 위치와도 안 맞음) 를 구분한다 — 둘은 다른 실패고 처방이 다르다.
    wrong_cites = [
        (c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else dict(c))
        for c in response.citations
        if not any(evidence_hit(g, c, precision) for g in golds)
    ]
    out = {"applicable": True, "matched": hit == n_facts, "score": score,
           "n_citations": len(response.citations), "n_gold_locations": len(golds),
           # 보고서가 "원래 근거 원소 수"와 "실제 채점한 사실 묶음 수"를 나눠 볼 수 있게 한다
           "n_gold_evidence": len(golds),          # 원래 근거 원소 수
           "n_fact_bundles": n_facts,              # 실제 채점한 사실 묶음 수
           "n_matched_facts": hit,                 # 맞춘 사실 수
           "n_missing_facts": n_facts - hit,       # 누락한 사실 수
           "n_missing_evidence": n_facts - hit,    # (기존 이름 유지 — 값은 사실 기준)
           "n_wrong_citations": len(wrong_cites),  # 잘못 인용한 출처 수
           "wrong_citations": wrong_cites,
           "search_used": search_used,
           "precision_unit": precision,
           "fact_keys": [list(map(str, k)) for k in bundles]}
    # 필드별 내역도 사실 묶음 기준으로 낸다(같은 사실을 두 번 세지 않는다).
    by_field: dict[str, dict] = {}
    seen: set[tuple] = set()
    for g, ok in per_gold:
        k = fact_key(g)
        if k in seen:
            continue
        seen.add(k)
        f = _ev_get(g, "fact_field") or _ev_get(g, "field")
        b = by_field.setdefault(str(f) if f else "(no field)", {"matched": 0, "total": 0})
        b["total"] += 1
        b["matched"] += int(bundles[k])
    if any(k != "(no field)" for k in by_field):
        out["by_field"] = by_field
    return out


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
    partial_missing = sum(1 for r in rows if r.get("n_missing_evidence", 0) > 0 and r.get("n_citations"))
    wrong_cite = sum(1 for r in rows if r.get("n_wrong_citations", 0) > 0)
    return {
        "n": n,
        "citation_accuracy": round(matched / n, 4),       # 정답 위치를 모두 정확히 인용
        "no_citation_rate": round(no_citation / n, 4),    # 출처 아예 안 붙임
        "partial_missing_rate": round(partial_missing / n, 4),  # 인용은 했으나 일부 근거 누락
        "wrong_citation_rate": round(wrong_cite / n, 4),  # 정답 위치 아닌 곳을 인용(잘못된 근거)
        "n_gold_evidence_total": sum(r.get("n_gold_evidence", 0) for r in rows),
        "n_fact_bundles_total": sum(r.get("n_fact_bundles", 0) for r in rows),
        "n_matched_facts_total": sum(r.get("n_matched_facts", 0) for r in rows),
        "n_missing_facts_total": sum(r.get("n_missing_facts", 0) for r in rows),
        "n_wrong_citations_total": sum(r.get("n_wrong_citations", 0) for r in rows),
        "note": "3-4-3 — 정답 위치 vs 인용 좌표. 안 붙임/일부 누락/틀리게 붙임을 분리 집계."
                " check_format의 FAIL 대상 아님(진단 지표).",
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
        # 다단계 k (retrieval_k 단계에만 by_k 가 붙음) — k별 recall/precision/mrr 평균
        byk_rows = [r["by_k"] for r in rows if r.get("by_k")]
        if byk_rows:
            ks = sorted(byk_rows[0], key=int)
            out[stage]["by_k"] = {
                k: {f"{m.split('_')[0]}@k": round(sum(b[k][m] for b in byk_rows) / len(byk_rows), 4)
                    for m in ("recall_at_k", "precision_at_k", "mrr")}
                for k in ks
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
