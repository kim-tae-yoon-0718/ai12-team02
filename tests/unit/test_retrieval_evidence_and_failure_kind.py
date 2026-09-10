"""§5 검색 정답 좌표는 evidence 의 청크 근거를 먼저 쓴다 / §6 다중 근거 부분 누락.

수정 전:
  · 근거가 evidence 배열에만 있는 문항(QA-007·QA-008)은 최상위 location 이 없어
    검색 채점에서 통째로 빠졌다(applicable=False).
  · 정답 근거 둘 중 하나만 컨텍스트에 있어도 failure_kind="none"(성공)이었다.
"""
import pytest

from grader.models import EvaluationItem, Location, ModelResponse
from grader.retrieval import grade_retrieval

L1 = {"document": "A", "section": "1장", "ref_no": "1장 · 문단 1", "line": 10}
L2 = {"document": "A", "section": "2장", "ref_no": "2장 · 문단 2", "line": 20}


def _item(**over):
    base = dict(id="R", question="q", task_type="qa", answer_type="value",
                answer_source="verified", answer_raw="x")
    base.update(over)
    return EvaluationItem.model_validate(base)


def _resp(retrieved, contexts):
    return ModelResponse.model_validate({
        "id": "R", "answer": "a",
        "retrieved": [{"document_id": loc["document"], "location": loc} for loc in retrieved],
        "contexts": [{"document_id": loc["document"], "location": loc} for loc in contexts]})


def _stages(item, retrieved, contexts):
    return grade_retrieval(item, _resp(retrieved, contexts), 5, None, 5)


# ── §5 evidence 의 청크 좌표를 쓴다 ──────────────────────────────────
def test_evidence_chunk_location_is_used_when_top_level_location_is_missing():
    item = _item(evidence=[{"kind": "extraction_table", "document": "A", "field": "예산",
                            "status": "value_present"},
                           {"kind": "chunk", "document": "A", "source": "chunks_v3",
                            "location": L1}])
    st = _stages(item, [L1], [L1])
    assert st[0]["applicable"] is True
    assert st[0]["n_gold"] == 1 and st[0]["recall_at_k"] == 1.0
    assert st[0]["failure_kind"] == "none"


def test_table_and_identity_evidence_are_not_search_gold_locations():
    """추출표 행·identity 값 자체는 벡터 검색의 정답 위치가 아니다."""
    item = _item(evidence=[{"kind": "extraction_table", "document": "A", "field": "예산",
                            "status": "field_absent"},
                           {"kind": "identity", "document": "A", "field": "bid_deadline"}])
    st = _stages(item, [L1], [L1])
    assert st[0]["applicable"] is False


def test_metadata_item_with_real_chunk_evidence_is_graded():
    """answer_source=metadata 여도 실제 청크 근거가 있으면 검색 채점 대상이다(EXT-07)."""
    item = _item(answer_source="metadata",
                 evidence=[{"kind": "identity", "document": "A", "field": "bid_deadline"},
                           {"kind": "chunk", "document": "A", "location": L1}])
    assert _stages(item, [L1], [L1])[0]["applicable"] is True


def test_metadata_item_without_chunk_evidence_stays_not_applicable():
    item = _item(answer_source="metadata",
                 location=Location.model_validate(
                     {"document": "A", "section": "CSV", "ref_no": "CSV: bid_deadline"}))
    assert _stages(item, [L1], [L1])[0]["applicable"] is False


def test_top_level_location_is_still_the_fallback():
    item = _item(location=Location.model_validate(L1))
    st = _stages(item, [L1], [L1])
    assert st[0]["applicable"] is True and st[0]["recall_at_k"] == 1.0


# ── §6 다중 근거 부분 누락 ───────────────────────────────────────────
TWO = [{"kind": "chunk", "document": "A", "location": L1},
       {"kind": "chunk", "document": "A", "location": L2}]


@pytest.mark.parametrize("retrieved,contexts,want", [
    ([L1, L2], [L1, L2], "none"),            # 둘 다 컨텍스트에 있음
    ([L1, L2], [L1], "rank_failure"),        # 후보엔 둘 다, 컨텍스트엔 하나
    ([L1], [L1], "recall_failure"),          # 후보에 하나만
    ([], [], "recall_failure"),
])
def test_multi_evidence_failure_kind(retrieved, contexts, want):
    st = _stages(_item(evidence=TWO), retrieved, contexts)
    assert st[0]["failure_kind"] == want
    assert st[0]["n_gold_total"] == 2


def test_partial_context_is_never_none():
    """정답 근거 2개 중 컨텍스트에 1개만 있으면 성공(none)이 아니다."""
    st = _stages(_item(evidence=TWO), [L1, L2], [L2])
    assert st[0]["failure_kind"] != "none"
    assert st[0]["n_gold_in_context"] == 1 and st[0]["n_gold_in_pool"] == 2


def test_single_evidence_behaviour_unchanged():
    one = [{"kind": "chunk", "document": "A", "location": L1}]
    assert _stages(_item(evidence=one), [L1], [L1])[0]["failure_kind"] == "none"
    assert _stages(_item(evidence=one), [L1], [])[0]["failure_kind"] == "rank_failure"
    assert _stages(_item(evidence=one), [], [])[0]["failure_kind"] == "recall_failure"
