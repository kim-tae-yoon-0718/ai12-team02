"""§7 같은 사실의 대체 가능한 출처 / §8 미기재 근거 인용 계약.

수정 전:
  · 같은 사실(문서+필드)을 추출표와 청크로 각각 확인할 수 있는 문항(QA-007·QA-008·
    EXT-07)은 두 경로를 **모두** 인용해야 만점이었다 — 대체 가능한 출처를 중복으로 셌다.
  · 미기재(field_absent) 근거를 value_present 라고 인용하거나 가짜 줄 번호를 붙여도
    같은 근거로 인정됐다.
"""
import pytest

from grader.models import EvaluationItem, ModelResponse
from grader.retrieval import fact_key, grade_citation

TABLE_EV = {"kind": "extraction_table", "document": "RFP-000096", "field": "예산",
            "fact_field": "예산", "status": "value_present",
            "source": "extraction_table_v3",
            "location": {"document": "RFP-000096", "field": "예산", "section": "1. 개요",
                         "ref_no": "1. 개요 · 문단 1", "line": 80}}
CHUNK_EV = {"kind": "chunk", "document": "RFP-000096", "fact_field": "예산",
            "source": "chunks_v3",
            "location": {"document": "RFP-000096", "field": "예산", "section": "1. 개요",
                         "ref_no": "1. 개요 · 문단 1", "line": 80}}
ABSENT_EV = {"kind": "extraction_table", "document": "RFP-000011", "field": "지역제한",
             "status": "field_absent", "source": "extraction_table_v3"}


def _item(evidence, **over):
    base = dict(id="C", question="q", task_type="qa", answer_type="value",
                answer_source="verified", answer_raw="x", evidence=evidence)
    base.update(over)
    return EvaluationItem.model_validate(base)


def _resp(citations):
    return ModelResponse.model_validate({"id": "C", "answer": "a", "citations": citations})


# ── 대체 가능한 출처 = 한 묶음 ───────────────────────────────────────
@pytest.mark.parametrize("cites", [
    [{"kind": "extraction_table", "document": "RFP-000096", "field": "예산",
      "status": "value_present", "source": "extraction_table_v3"}],
    [{"document": "RFP-000096", "section": "1. 개요", "ref_no": "1. 개요 · 문단 1", "line": 80}],
])
def test_one_correct_source_of_the_same_fact_is_enough(cites):
    out = grade_citation(_item([TABLE_EV, CHUNK_EV]), _resp(cites))
    assert out["n_gold_evidence"] == 2      # 원래 근거 원소 수
    assert out["n_fact_bundles"] == 1       # 실제 채점한 사실 묶음 수
    assert out["n_matched_facts"] == 1 and out["n_missing_facts"] == 0
    assert out["score"] == 1.0 and out["matched"] is True


def test_identity_and_chunk_of_one_fact_share_a_bundle():
    ident = {"kind": "identity", "document": "RFP-000014", "field": "bid_deadline",
             "fact_field": "bid_deadline", "source": "identity_v2"}
    chunk = {"kind": "chunk", "document": "RFP-000014", "fact_field": "bid_deadline",
             "source": "chunks_v3",
             "location": {"document": "RFP-000014", "section": "2. 절차",
                          "ref_no": "2. 절차 · 문단 2-86", "line": 141}}
    assert fact_key(ident) == fact_key(chunk)
    out = grade_citation(_item([ident, chunk]),
                         _resp([{"kind": "identity", "document": "RFP-000014",
                                 "field": "bid_deadline", "source": "identity_v2"}]))
    assert out["score"] == 1.0 and out["n_fact_bundles"] == 1


# ── 서로 다른 사실은 모두 필요 ───────────────────────────────────────
def test_different_document_field_pairs_are_separate_facts():
    evs = [{"kind": "extraction_table", "document": d, "field": f, "status": "value_present"}
           for d in ("RFP-000014", "RFP-000020") for f in ("예산", "사업기간")]
    out = grade_citation(_item(evs, answer_type="comparison"),
                         _resp([{"kind": "extraction_table", "document": "RFP-000014",
                                 "field": "예산", "status": "value_present"}]))
    assert out["n_fact_bundles"] == 4 and out["n_matched_facts"] == 1
    assert out["score"] == 0.25 and out["matched"] is False


def test_same_document_different_condition_fields_are_separate_facts():
    evs = [{"kind": "extraction_table", "document": "RFP-000067", "field": "컨소시엄 요건",
            "status": "value_present"},
           {"kind": "extraction_table", "document": "RFP-000067", "field": "평가 배점",
            "status": "field_absent"}]
    out = grade_citation(_item(evs, answer_type="document_set"),
                         _resp([{"kind": "extraction_table", "document": "RFP-000067",
                                 "field": "컨소시엄 요건", "status": "value_present"}]))
    assert out["n_fact_bundles"] == 2 and out["n_missing_facts"] == 1


def test_document_id_alone_is_not_enough():
    out = grade_citation(_item([TABLE_EV, CHUNK_EV]),
                         _resp([{"document": "RFP-000096", "section": "다른 절",
                                 "ref_no": "다른 절 · 문단 9", "line": 900}]))
    assert out["score"] == 0.0 and out["n_wrong_citations"] == 1


def test_chunk_evidences_without_field_stay_separate_facts():
    a = {"kind": "chunk", "document": "RFP-000014", "source": "chunks_v3",
         "location": {"document": "RFP-000014", "section": "1장", "ref_no": "1장 · 문단 1", "line": 5}}
    b = {"kind": "chunk", "document": "RFP-000014", "source": "chunks_v3",
         "location": {"document": "RFP-000014", "section": "2장", "ref_no": "2장 · 문단 2", "line": 9}}
    assert fact_key(a) != fact_key(b)
    out = grade_citation(_item([a, b]), _resp([a["location"]]))
    assert out["n_fact_bundles"] == 2 and out["n_matched_facts"] == 1


# ── 미기재 근거 계약 ─────────────────────────────────────────────────
def test_field_absent_evidence_citation_without_location_is_accepted():
    out = grade_citation(_item([ABSENT_EV]), _resp([dict(ABSENT_EV)]))
    assert out["score"] == 1.0 and out["n_citations"] == 1


def test_field_absent_cited_as_value_present_is_wrong():
    bad = dict(ABSENT_EV, status="value_present")
    out = grade_citation(_item([ABSENT_EV]), _resp([bad]))
    assert out["score"] == 0.0 and out["n_wrong_citations"] == 1


def test_fabricated_line_for_field_absent_is_wrong():
    out = grade_citation(_item([ABSENT_EV]),
                         _resp([{"document": "RFP-000011", "section": "",
                                 "ref_no": "지역제한", "line": 42}]))
    assert out["score"] == 0.0 and out["n_wrong_citations"] == 1


@pytest.mark.parametrize("bad", [
    {"kind": "extraction_table", "document": "RFP-000011", "field": "예산",
     "status": "field_absent"},                       # 같은 문서 다른 필드
    {"kind": "extraction_table", "document": "RFP-000012", "field": "지역제한",
     "status": "field_absent"},                       # 같은 필드 다른 문서
])
def test_wrong_field_or_document_is_not_the_same_evidence(bad):
    out = grade_citation(_item([ABSENT_EV]), _resp([bad]))
    assert out["score"] == 0.0


def test_no_citation_reports_fact_counts():
    out = grade_citation(_item([TABLE_EV, CHUNK_EV]), _resp([]))
    assert out["n_citations"] == 0 and out["n_fact_bundles"] == 1
    assert out["n_missing_facts"] == 1 and out["score"] == 0.0
