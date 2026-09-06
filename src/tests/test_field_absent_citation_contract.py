#@title 위치 없는 근거를 거짓 좌표 없이 내는 응답 계약 (§8)
#@markdown 미기재(field_absent)와 identity 컬럼은 가리킬 원문 줄이 없다. 빈 section 과
#@markdown 가짜 ref_no 로 위치처럼 위장하지 않고 Evidence 형으로 낸다.
"""선별형에서는 선택된 문서마다 **판정에 쓴 모든 조건 필드**의 근거를 낸다."""
from __future__ import annotations

import pytest

from answer_pipeline import build_field_evidence, table_state_citation
from test_selection_fix import absent, cfg, present, run_answer, write_identity, write_table

LOCATION_KEYS = ("location", "line", "line_end", "ref_no", "section",
                 "block_type", "block_index")


def assert_no_fake_location(cite):
    for key in LOCATION_KEYS:
        assert key not in cite, (key, cite)


@pytest.fixture
def world(tmp_path):
    table = write_table(tmp_path, {
        # 값 있음 + 미기재가 섞인 문서
        "RFP-000901": {"컨소시엄 요건": present("공동수급 허용"), "평가 배점": absent(),
                       "지역제한": absent()},
        "RFP-000902": {"컨소시엄 요건": absent(), "지역제한": absent()},
        "RFP-000903": {"컨소시엄 요건": present("단독 참여만"), "평가 배점": present("100점"),
                       "지역제한": present("서울")},
    })
    identity = write_identity(tmp_path, [
        ("RFP-000901", "가온재단", "하늘 관측망 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000902", "별빛공사", "바다 관측망 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000903", "초록협회", "숲길 안내 시스템 구축 사업", "2025-01-01 17:00:00"),
    ])
    return table, identity


def test_absent_row_is_cited_as_evidence_not_a_location(world):
    table, _ = world
    ev = build_field_evidence(table, "RFP-000902", "컨소시엄 요건", cfg())
    assert len(ev.citations) == 1
    cite = ev.citations[0]
    assert cite["kind"] == "extraction_table"
    assert cite["document"] == "RFP-000902" and cite["field"] == "컨소시엄 요건"
    assert cite["status"] == "field_absent"
    assert str(cite["source"]).startswith("extraction_table")
    assert_no_fake_location(cite)


def test_value_row_still_uses_a_real_location(world):
    table, _ = world
    ev = build_field_evidence(table, "RFP-000901", "컨소시엄 요건", cfg())
    assert len(ev.citations) == 1
    assert ev.citations[0]["line"] == 10          # 실재하는 좌표는 그대로 쓴다
    assert "kind" not in ev.citations[0]


def test_selection_cites_every_condition_field_per_document(world):
    """값 있는 조건만 내고 미기재 조건 근거를 빠뜨리지 않는다."""
    table, identity = world
    r = run_answer("공동수급 요건은 명시되어 있고 평가 배점 항목은 적혀 있지 않은 공고 알려줘",
                   table, identity=identity)
    assert r.selected_document_ids == ["RFP-000901"]
    pairs = {(c["document"], c.get("field")) for c in r.citations}
    assert pairs == {("RFP-000901", "컨소시엄 요건"), ("RFP-000901", "평가 배점")}
    absent_cites = [c for c in r.citations if c.get("status") == "field_absent"]
    assert len(absent_cites) == 1
    assert_no_fake_location(absent_cites[0])


def test_field_absent_only_selection_cites_all_documents(world):
    table, identity = world
    r = run_answer("지역제한 항목이 공고문에 적혀 있지 않은 공고만 모아줘",
                   table, identity=identity)
    assert sorted(r.selected_document_ids) == ["RFP-000901", "RFP-000902"]
    assert len(r.citations) == 2
    for cite in r.citations:
        assert cite["status"] == "field_absent" and cite["field"] == "지역제한"
        assert_no_fake_location(cite)


def test_identity_deadline_is_evidence_shaped(world):
    table, identity = world
    r = run_answer("RFP-000901의 입찰 마감일 언제야?", table, identity=identity)
    ident = [c for c in r.citations if c.get("kind") == "identity"]
    assert len(ident) == 1
    assert ident[0]["field"] == "bid_deadline" and ident[0]["source"] == "identity_v2"
    assert_no_fake_location(ident[0])


def test_table_state_citation_helper_never_emits_coordinates():
    cite = table_state_citation("RFP-000901", "지역제한", "field_absent")
    assert cite == {"kind": "extraction_table", "document": "RFP-000901",
                    "field": "지역제한", "status": "field_absent",
                    "source": "extraction_table"}


def test_response_contract_keeps_twelve_top_level_fields(world):
    from answer_pipeline import answer_to_response
    table, identity = world
    r = run_answer("지역제한 항목이 공고문에 적혀 있지 않은 공고만 모아줘",
                   table, identity=identity)
    payload = answer_to_response("X", r)
    assert set(payload) == {"id", "answer", "structured_answer", "contexts", "retrieved",
                            "citations", "selected_document_ids", "abstained", "route",
                            "failure", "latency_ms", "cost_usd"}


def test_grader_accepts_both_citation_shapes(world):
    """채점기 응답 계약도 좌표형·Evidence형을 함께 받는다(하위 호환)."""
    from grader.models import ModelResponse
    table, identity = world
    r = run_answer("지역제한 항목이 공고문에 적혀 있지 않은 공고만 모아줘",
                   table, identity=identity)
    payload = {"id": "X", "answer": r.text, "citations": r.citations + [
        {"document": "RFP-000903", "section": "1. 개요", "ref_no": "1. 개요 · 문단 1", "line": 10}]}
    parsed = ModelResponse.model_validate(payload)
    kinds = [type(c).__name__ for c in parsed.citations]
    assert "Evidence" in kinds and "Location" in kinds
