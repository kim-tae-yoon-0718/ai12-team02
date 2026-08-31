from grader.models import EvaluationItem, Location, ModelResponse, RetrievedItem
from grader.retrieval import aggregate_citation, grade_citation, grade_retrieval


def _item(**over):
    base = dict(
        id="T", question="q", task_type="qa", answer_type="value",
        answer_raw="x",
        location=Location(document="A", section="4장", ref_no="본문"),
    )
    base.update(over)
    return EvaluationItem(**base)


def _resp(**over):
    base = dict(id="T", answer="x")
    base.update(over)
    return ModelResponse(**base)


def test_rank_failure():
    """후보 풀엔 있는데 실제 context엔 못 듦 → 재정렬(reranker)의 자리."""
    resp = _resp(
        retrieved=[
            RetrievedItem(document_id="A", location=Location(document="A", section="5장", ref_no="본문")),
            RetrievedItem(document_id="A", location=Location(document="A", section="4장", ref_no="본문")),
        ],
        contexts=[],
    )
    stages = grade_retrieval(_item(), resp, retrieval_k=5, reranker_k=5, context_k=5)
    assert all(s["failure_kind"] == "rank_failure" for s in stages)


def test_recall_failure():
    """후보 풀에 아예 없음 → 재정렬은 무용. 임베딩/하이브리드/파싱이 처방."""
    resp = _resp(
        retrieved=[RetrievedItem(document_id="B", location=Location(document="B", section="1장", ref_no="본문"))],
        contexts=[],
    )
    stages = grade_retrieval(_item(), resp, retrieval_k=5, reranker_k=5, context_k=5)
    assert all(s["failure_kind"] == "recall_failure" for s in stages)


def test_no_location_is_not_scored():
    it = _item(location=None)
    stages = grade_retrieval(it, _resp(), retrieval_k=5, reranker_k=5, context_k=5)
    assert all(not s["applicable"] for s in stages)


def test_context_hit_means_no_failure():
    loc = Location(document="A", section="4장", ref_no="본문")
    resp = _resp(
        retrieved=[RetrievedItem(document_id="A", location=loc)],
        contexts=[{"document_id": "A", "text": "본문 내용", "location": loc.model_dump()}],
    )
    stages = grade_retrieval(_item(), resp, retrieval_k=5, reranker_k=5, context_k=5)
    assert all(s["failure_kind"] == "none" for s in stages)


# ------------------------------------------------------------------ 3-4-3 출처 좌표 채점


def test_citation_matches_gold_location():
    """정답과 같은 좌표를 인용하면 맞은 것으로 채점된다."""
    loc = Location(document="A", section="4장", ref_no="본문")
    resp = _resp(citations=[loc])
    result = grade_citation(_item(), resp)
    assert result["applicable"] is True
    assert result["matched"] is True
    assert result["score"] == 1.0


def test_citation_wrong_location_is_not_matched():
    """citation은 있지만 정답 좌표와 다르면 '틀리게 붙인 것' — score 0, matched False."""
    wrong = Location(document="A", section="9장", ref_no="본문")
    resp = _resp(citations=[wrong])
    result = grade_citation(_item(), resp)
    assert result["applicable"] is True
    assert result["matched"] is False
    assert result["score"] == 0.0
    assert result["n_citations"] == 1


def test_missing_citation_is_scored_zero_not_format_fail():
    """citation 미표기는 check_format을 FAIL로 만들지 않지만(별도 유지),
    3-4-3 진단에서는 score 0으로 반영돼야 한다 — 그래야 no_citation_rate를 잴 수 있다."""
    result = grade_citation(_item(), _resp())  # citations=[] 기본값
    assert result["applicable"] is True
    assert result["matched"] is False
    assert result["score"] == 0.0
    assert result["n_citations"] == 0
    assert "출처 없음" in result["reason"] or "citation 없음" in result["reason"]


def test_no_gold_location_is_not_applicable():
    it = _item(location=None)
    result = grade_citation(it, _resp(citations=[Location(document="A", section="1장", ref_no="본문")]))
    assert result["applicable"] is False


def test_disabled_via_config_is_not_applicable():
    loc = Location(document="A", section="4장", ref_no="본문")
    result = grade_citation(_item(), _resp(citations=[loc]), enabled=False)
    assert result["applicable"] is False
    assert "grade_citations" in result["reason"]


def test_aggregate_citation_separates_wrong_from_missing():
    """오표기(matched=False, n_citations>0)와 미표기(n_citations=0)를 no_citation_rate로
    구분해서 낼 수 있어야 한다 — citation_accuracy만으로는 둘을 못 가른다."""
    rows = [
        {"applicable": True, "matched": True, "n_citations": 1},   # 맞음
        {"applicable": True, "matched": False, "n_citations": 1},  # 틀리게 붙임
        {"applicable": True, "matched": False, "n_citations": 0},  # 아예 안 붙임
        {"applicable": False, "reason": "정답 location 없음"},      # 집계 제외
    ]
    agg = aggregate_citation(rows)
    assert agg["n"] == 3
    assert agg["citation_accuracy"] == round(1 / 3, 4)
    assert agg["no_citation_rate"] == round(1 / 3, 4)


def test_aggregate_citation_empty_is_silent():
    assert aggregate_citation([])["n"] == 0
