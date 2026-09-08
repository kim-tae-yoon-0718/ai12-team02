"""
PR #10 수정분 검증 (2026-09-02) — 실제 API 호출 0건.

각 절 번호는 요청서의 항목 번호를 따른다.
  3-1 문서 특정 정책      3-2 QA 구조화 자료 결합    3-3 숫자 쉼표 버그
  4-1 비교 연산자·지역제한 4-2 버전·해시 검증         4-3 근거 위치·응답 형식
  4-4 마감일 = identity_v2 4-5 GPT-5 Mini 요청 인자
  5-1 비용 격리           5-2 시스템 사용법          5-5 실패 종료코드
  5-6 세션 분리
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


def assert_identity_deadline_citation(cite):
    """[2026-09-04 §8] 마감일 근거는 identity_v2 의 컬럼 자체다 — 좌표가 아니다.

    예전에는 section="CSV" / ref_no="CSV: bid_deadline" 로 **좌표 모양**을 냈다.
    원문에 그런 위치는 없으므로 위장이다. 이제 kind/document/field/source 로 내고,
    좌표를 나타내는 키는 하나도 없어야 한다.
    """
    assert cite.get("kind") == "identity", cite
    assert cite.get("field") == "bid_deadline", cite
    assert cite.get("source") == "identity_v2", cite
    for forbidden in ("ref_no", "section", "line", "line_end", "block_type",
                      "block_index", "location"):
        assert forbidden not in cite, (forbidden, cite)



# ===========================================================================
# 3-1 문서 특정 정책
# ===========================================================================

class TestDocumentResolution:
    def test_by_document_id(self, identity_index):
        from doc_resolver import resolve_document, RESOLVE_EXPLICIT
        r = resolve_document("RFP-000002의 예산 알려줘", identity_index)
        assert r.document_id == "RFP-000002" and r.method == RESOLVE_EXPLICIT

    def test_by_active_document_id_followup(self, identity_index):
        from doc_resolver import resolve_document, RESOLVE_ACTIVE
        r = resolve_document("거기 소요예산은 얼마야?", identity_index,
                             active_document_id="RFP-000001")
        assert r.document_id == "RFP-000001" and r.method == RESOLVE_ACTIVE

    def test_active_document_not_hijacking_named_question(self, identity_index):
        """직전 문서가 있어도, 질문이 다른 기관을 명시하면 그 기관을 쓴다."""
        from doc_resolver import resolve_document
        r = resolve_document("서울특별시교육청 사업 예산 알려줘", identity_index,
                             active_document_id="RFP-000001")
        assert r.document_id == "RFP-000099"

    def test_org_exact_match(self, identity_index):
        from doc_resolver import resolve_document, RESOLVE_ORG
        r = resolve_document("서울특별시교육청 사업 예산 알려줘", identity_index)
        assert r.document_id == "RFP-000099" and r.method == RESOLVE_ORG

    def test_org_prefix_sa_normalized(self, identity_index):
        """(사) 접두어 차이 — EXT-004 재현 케이스."""
        from doc_resolver import resolve_document
        r = resolve_document("벤처기업협회의 예산 알려줘", identity_index)
        assert r.document_id == "RFP-000001"

    @pytest.mark.parametrize("written", [
        "테스트정보", "㈜테스트정보", "(주)테스트정보", "주식회사 테스트정보",
    ])
    def test_org_prefix_ju_variants(self, identity_index, written):
        from doc_resolver import resolve_document
        r = resolve_document(f"{written} 예산 알려줘", identity_index)
        assert r.document_id == "RFP-000002"

    @pytest.mark.parametrize("written", [
        "한국테스트재단", "(재)한국테스트재단", "재단법인 한국테스트재단",
    ])
    def test_org_prefix_jae_variants(self, identity_index, written):
        from doc_resolver import resolve_document
        r = resolve_document(f"{written} 예산 알려줘", identity_index)
        assert r.document_id == "RFP-000003"

    def test_core_org_name_never_stripped(self):
        """협회·연구원·진흥원 같은 핵심 이름을 지우면 안 된다."""
        from text_normalize import normalize_org_key
        for token in ("협회", "연구원", "진흥원", "재단"):
            assert token in normalize_org_key(f"(사)한국{token}")
        assert normalize_org_key("(사)한국협회") != normalize_org_key("(사)한국연구원")

    def test_project_only_single_candidate(self, identity_index):
        """사업명만 있고 기관명이 없음 — EXT-002 재현 케이스."""
        from doc_resolver import resolve_document
        r = resolve_document(
            "통합관리시스템 구축 사업에 입찰하려면 어떤 서류가 필요해?", identity_index)
        assert r.document_id == "RFP-000002"
        assert r.method == "project"

    def test_org_and_project_both_match(self, identity_index):
        from doc_resolver import resolve_document, RESOLVE_ORG_AND_PROJECT
        r = resolve_document(
            "서울특별시교육청 지능정보화전략계획 수립 사업 예산 알려줘", identity_index)
        assert r.document_id == "RFP-000099"
        assert r.method == RESOLVE_ORG_AND_PROJECT

    def test_ambiguous_candidates_ask_back(self, tmp_path):
        """같은 기관에 문서가 여러 건이면 첫 문서를 임의로 고르지 않는다."""
        from identity_metadata import load_identity
        from doc_resolver import resolve_document
        header = ('"document_id","source_filename_nfc","collection_system",'
                  '"source_record_id","source_url","notice_number","notice_round",'
                  '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')
        rows = [
            '"RFP-000010","a.md","","","","1","0","한국수자원공사","통합정보시스템 고도화 용역",'
            '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
            '"RFP-000011","b.md","","","","2","0","한국수자원공사","수자원 데이터 플랫폼",'
            '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
        ]
        p = tmp_path / "identity_multi.csv"
        p.write_text("﻿" + "\n".join([header] + rows) + "\n", encoding="utf-8")
        idx = load_identity(p)
        r = resolve_document("한국수자원공사 사업 예산 알려줘", idx)
        assert r.document_id is None
        assert set(r.candidates) == {"RFP-000010", "RFP-000011"}
        assert r.is_ambiguous

    def test_known_and_unknown_org_together(self, identity_index):
        from doc_resolver import resolve_documents_for_compare
        docs, unknown = resolve_documents_for_compare(
            "벤처기업협회 사업이랑 가짜미래재단 사업을 비교해줘", identity_index)
        assert "RFP-000001" in docs
        assert "가짜미래재단" in unknown

    def test_ambiguity_surfaces_in_answer(self, tmp_path, table_rows, small_store,
                                          base_cfg):
        """되묻기 답변이 후보를 실제로 보여주고 abstain해야 한다."""
        from identity_metadata import load_identity
        from answer_pipeline import answer, SessionState, ROUTE_CLARIFY
        from conftest import forbidden
        header = ('"document_id","source_filename_nfc","collection_system",'
                  '"source_record_id","source_url","notice_number","notice_round",'
                  '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')
        rows = [
            '"RFP-000001","a.md","","","","1","0","한국수자원공사","A 사업",'
            '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
            '"RFP-000002","b.md","","","","2","0","한국수자원공사","B 사업",'
            '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
        ]
        p = tmp_path / "identity_multi2.csv"
        p.write_text("﻿" + "\n".join([header] + rows) + "\n", encoding="utf-8")
        idx = load_identity(p)
        r = answer("한국수자원공사 예산 알려줘", small_store, forbidden("embed"),
                   forbidden("gen"), table_rows, base_cfg, identity=idx,
                   session=SessionState())
        assert r.abstained is True and r.route == ROUTE_CLARIFY
        assert "RFP-000001" in r.text and "RFP-000002" in r.text


# ===========================================================================
# 3-2 / 4-4  QA 구조화 자료 결합 + 마감일 = identity_v2
# ===========================================================================

class TestQaStructuredFusion:
    def test_qa001_deadline_keeps_qa_and_uses_identity(
        self, table_rows, small_store, base_cfg, identity_index, fake_clients,
    ):
        """QA-001 필수 검사 — QA 분류 유지 / identity_v2 값 / 근거 위치 /
        청크 보조 사용 / 정확한 마감일 포함 / citations / route 기록."""
        from answer_pipeline import answer, SessionState, ROUTE_STRUCTURED_LLM
        e, g, get_e, get_g = fake_clients
        q = "오늘(2024.6.1)을 기준으로 벤처기업협회 사업에 입찰 참여할 수 있어?"
        r = answer(q, small_store, get_e, get_g, table_rows, base_cfg,
                   identity=identity_index, session=SessionState())
        assert r.task_type == "qa"                      # 분류 유지
        assert r.route == ROUTE_STRUCTURED_LLM          # 실제 경로 기록
        assert "2025-01-01 17:00" in r.text             # 정확한 마감일 포함
        assert "참여 가능" in r.text
        assert r.citations, "citations가 비면 안 됨"
        ident = [c for c in r.citations if c.get("kind") == "identity"]
        assert len(ident) == 1
        assert_identity_deadline_citation(ident[0])
        assert e.calls >= 1 and g.calls >= 1            # 청크 검색도 보조로 사용
        assert g.last_structured_context is not None
        assert r.structured_answer["입찰 참여 마감일"]["source"] == "identity_v2"
        assert r.structured_answer["입찰 참여 마감일"]["reference_datetime"] == "2024-06-01"

    def test_qa_12field_uses_extraction_table(
        self, table_rows, small_store, base_cfg, identity_index, fake_clients,
    ):
        """12필드 QA도 구조화 자료를 함께 쓴다."""
        from answer_pipeline import answer, SessionState, ROUTE_STRUCTURED_LLM
        e, g, get_e, get_g = fake_clients
        r = answer("RFP-000001 사업의 예산이 왜 그렇게 잡혔는지 설명해줘",
                   small_store, get_e, get_g, table_rows, base_cfg,
                   identity=identity_index, session=SessionState())
        assert r.route == ROUTE_STRUCTURED_LLM
        assert "1억 5천만" in r.text
        assert g.last_structured_context and "예산" in g.last_structured_context
        assert r.citations

    def test_structured_value_marked_authoritative_in_prompt(self):
        from generation_client import GenerationClient
        gc = GenerationClient.__new__(GenerationClient)
        gc._system_prompt_template = "SYS {format_instruction}"
        msgs = gc.build_messages("질문", ["청크1"], structured_context="- 값: 100")
        user = msgs[1]["content"]
        assert "공식 확정 값" in user
        assert "바꾸거나 다시 계산하지 마세요" in user
        assert user.index("공식 확정 값") < user.index("원문 근거")

    @pytest.mark.parametrize("doc_id,field,expect", [
        ("RFP-000001", "지역제한", "항목 자체가 없"),          # field_absent
        ("RFP-000003", "예산", "외부 공고문"),                 # external_reference
        ("RFP-000003", "평가 배점", "비공개"),                  # not_disclosed
    ])
    def test_status_handling(self, table_rows, base_cfg, doc_id, field, expect):
        from answer_pipeline import build_field_evidence
        ev = build_field_evidence(table_rows, doc_id, field, base_cfg)
        assert expect in ev.answer_text

    def test_conflict_keeps_both_locations(self, table_rows, base_cfg):
        """RFP-000028 유형 — 두 위치(L385 / L669)를 모두 보존해야 한다."""
        from answer_pipeline import build_field_evidence
        ev = build_field_evidence(table_rows, "RFP-000002", "컨소시엄 요건", base_cfg)
        assert ev.status == "conflict" and ev.abstained is True
        assert "불허" in ev.answer_text and "허용" in ev.answer_text
        lines = {c["line"] for c in ev.citations}
        assert {385, 669} <= lines
        assert len(ev.structured["conflict_values"]) == 2

    def test_extraction_failed_not_answered_as_value(self, base_cfg):
        from answer_pipeline import build_field_evidence
        rows = [{"document_id": "D1", "field_name": "예산", "status": "extraction_failed",
                 "answer_raw": "", "answer_normalized": "", "active": "true",
                 "representative_location": None, "additional_locations": []}]
        ev = build_field_evidence(rows, "D1", "예산", base_cfg)
        assert ev.abstained is True
        assert "정상 값으로 취급하지 않습니다" in ev.answer_text


class TestDeadlineSource:
    def test_deadline_answer_has_all_required_parts(self, identity_index, base_cfg):
        from answer_pipeline import build_deadline_evidence
        ev = build_deadline_evidence("RFP-000001", identity_index, base_cfg)
        assert "2025-01-01 17:00" in ev.answer_text
        assert ev.structured["value_normalized"] == "2025-01-01"
        assert_identity_deadline_citation(ev.citations[0])
        assert ev.used_source["source"] == "identity_v2"
        assert ev.structured["reference_datetime"] == "2024-06-01"

    def test_missing_value_reported_not_silently_replaced(self, identity_index, base_cfg):
        from answer_pipeline import build_deadline_evidence
        ev = build_deadline_evidence("RFP-000003", identity_index, base_cfg)
        assert ev.abstained is True
        assert "identity_v2" in ev.answer_text and "미상" in ev.answer_text

    def test_no_data_list_csv_path_left_in_code(self):
        """4-4 — data_list.csv를 직접 읽는 경로가 남아 있으면 안 된다."""
        src = Path(__file__).resolve().parent.parent
        offenders = []
        for p in list((src / "rag").glob("*.py")) + list((src / "scripts").glob("*.py")):
            text = p.read_text(encoding="utf-8")
            if "data_list.csv" in text or "deadline_csv_path" in text:
                offenders.append(p.name)
        assert offenders == [], f"data_list.csv 직접 참조가 남아 있음: {offenders}"


# ===========================================================================
# 3-3 숫자 안의 쉼표
# ===========================================================================

class TestNumberCommaBug:
    @pytest.mark.parametrize("text,expected", [
        ("49,500", 49_500.0),
        ("49,500,000", 49_500_000.0),
        ("1,234.5", 1_234.5),
    ])
    def test_amount_parsing(self, text, expected):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount(text) == expected

    @pytest.mark.parametrize("text", ["49,500", "49,500,000", "1,234.5"])
    def test_number_never_split_by_condition_splitter(self, text):
        from table_query import split_conditions
        assert split_conditions(text) == [text]

    def test_single_amount_condition(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("예산 49,500만원 이상")
        assert full and len(conds) == 1
        assert conds[0].field == "예산" and conds[0].operator == ">="
        assert conds[0].value == 495_000_000.0

    def test_two_conditions_split_on_real_separator(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("예산 49,500만원 이상, 지역제한 없음")
        assert full and len(conds) == 2
        assert (conds[0].field, conds[0].operator, conds[0].value) == \
               ("예산", ">=", 495_000_000.0)
        assert (conds[1].field, conds[1].operator) == ("지역제한", "no_restriction")

    def test_number_comparison_returns_expected_document_ids(self, table_rows):
        from table_query import parse_conditions, run_conditions_query
        conds, _ = parse_conditions("예산 4,900만원 이상인 사업")
        results, _, _ = run_conditions_query(table_rows, conds)
        assert sorted(r.document_id for r in results) == ["RFP-000001", "RFP-000002"]
        conds2, _ = parse_conditions("예산 1억 이상인 사업")
        results2, _, _ = run_conditions_query(table_rows, conds2)
        assert [r.document_id for r in results2] == ["RFP-000001"]


# ===========================================================================
# 4-1 비교 연산자·지역제한
# ===========================================================================

class TestComparisonOperators:
    @pytest.mark.parametrize("phrase,op", [
        ("5억 이상인 사업", ">="),
        ("5억 초과인 사업", ">"),
        ("5억 넘는 사업", ">"),
        ("5억 이하인 사업", "<="),
        ("5억 미만인 사업", "<"),
    ])
    def test_operators_distinct(self, phrase, op):
        from table_query import parse_conditions
        conds, full = parse_conditions(phrase)
        assert full and conds[0].operator == op
        assert conds[0].value == 500_000_000.0

    @pytest.mark.parametrize("phrase,expected", [
        ("1억 5천만원 이상인 사업", ["RFP-000001"]),
        ("1억 5천만원 초과인 사업", []),
        ("1억 5천만원 이하인 사업", ["RFP-000001", "RFP-000002"]),
        ("1억 5천만원 미만인 사업", ["RFP-000002"]),
        ("4,950만원 넘는 사업", ["RFP-000001"]),
    ])
    def test_operator_document_ids(self, table_rows, phrase, expected):
        from table_query import parse_conditions, run_conditions_query
        conds, full = parse_conditions(phrase)
        assert full
        results, _, _ = run_conditions_query(table_rows, conds)
        assert sorted(r.document_id for r in results) == expected


class TestRegionRestriction:
    def test_blank_is_not_no_restriction(self, table_rows):
        from table_query import lookup_field, classify_region_restriction, REGION_UNKNOWN
        row = lookup_field(table_rows, "RFP-000001", "지역제한")
        assert classify_region_restriction(row)[0] == REGION_UNKNOWN

    def test_value_present_alone_is_not_restricted(self):
        from table_query import classify_region_restriction, REGION_NONE, REGION_UNKNOWN
        assert classify_region_restriction(
            {"status": "value_present", "answer_raw": "지역제한 없음"})[0] == REGION_NONE
        assert classify_region_restriction(
            {"status": "value_present", "answer_raw": ""})[0] == REGION_UNKNOWN

    def test_meaning_based_classification(self):
        from table_query import classify_region_restriction, REGION_RESTRICTED, REGION_NONE
        restricted = "1) 입찰방식 : 제한경쟁입찰, 지역제한(광주광역시)"
        assert classify_region_restriction(
            {"status": "value_present", "answer_raw": restricted})[0] == REGION_RESTRICTED
        assert classify_region_restriction(
            {"status": "value_present", "answer_raw": "지역 제한 없이 전국 참가 가능"}
        )[0] == REGION_NONE

    def test_region_query_document_ids(self, table_rows):
        from table_query import parse_conditions, run_conditions_query
        conds, _ = parse_conditions("지역제한 있는 사업 알려줘")
        results, _, _ = run_conditions_query(table_rows, conds)
        assert [r.document_id for r in results] == ["RFP-000002"]
        conds2, _ = parse_conditions("지역제한 없는 사업 알려줘")
        results2, _, warnings2 = run_conditions_query(table_rows, conds2)
        assert [r.document_id for r in results2] == []
        assert any("RFP-000001" in w for w in warnings2)


# ===========================================================================
# 4-2 버전·해시 검증
# ===========================================================================

def _full_size_table_doc():
    """공식 규모(100문서 x 12필드 = 1,200행)와 같은 합성 추출표."""
    from conftest import OFFICIAL_FIELDS
    rows = []
    for i in range(1, 101):
        doc_id = f"RFP-{i:06d}"
        for f in OFFICIAL_FIELDS:
            rows.append({
                "document_id": doc_id, "field_name": f, "status": "field_absent",
                "answer_raw": "", "answer_normalized": "", "active": "true",
                "representative_location": None, "additional_locations": [],
                "schema_version": "1-12-2/v3", "extraction_version": "v3",
                "corpus_version": "v2", "registry_version": "v2",
            })
    doc = {
        "corpus_version": "v2", "document_count": 100, "extraction_version": "v3",
        "field_count": 12, "fields": list(OFFICIAL_FIELDS), "registry_version": "v2",
        "row_count": len(rows), "schema_version": "1-12-2/v3", "rows": rows,
    }
    meta = {
        "row_count": 1200, "document_count": 100, "field_count": 12,
        "schema_version": "1-12-2/v3", "extraction_version": "v3",
        "corpus_version": "v2", "registry_version": "v2",
    }
    return doc, meta


class TestVersionAndHashGuards:
    def test_chunk_missing_corpus_version_fails(self, chunks_jsonl_path, base_cfg):
        from build_index import load_chunks, map_chunk, validate_chunk_versions, ChunkContractError
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        for c in chunks:
            c["corpus_version"] = ""
        with pytest.raises(ChunkContractError, match="corpus_version"):
            validate_chunk_versions(chunks, base_cfg)

    def test_chunk_missing_chunking_version_fails(self, chunks_jsonl_path, base_cfg):
        from build_index import load_chunks, map_chunk, validate_chunk_versions, ChunkContractError
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        for c in chunks:
            c["chunking_version"] = ""
        with pytest.raises(ChunkContractError, match="chunking_version"):
            validate_chunk_versions(chunks, base_cfg)

    def test_chunk_version_mismatch_with_config_fails(self, chunks_jsonl_path, base_cfg):
        from build_index import load_chunks, map_chunk, validate_chunk_versions, ChunkContractError
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        for c in chunks:
            c["chunking_version"] = "v1"
        with pytest.raises(ChunkContractError, match="설정 요구값"):
            validate_chunk_versions(chunks, base_cfg)

    def test_internal_consistency_alone_is_not_enough(self, chunks_jsonl_path,
                                                      chunks_version_txt, base_cfg):
        from build_index import load_chunks, map_chunk, validate_chunk_versions, ChunkContractError
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        chunks_version_txt.write_text(
            "chunking version   : v1\ncorpus version     : v2\n", encoding="utf-8")
        with pytest.raises(ChunkContractError):
            validate_chunk_versions(chunks, base_cfg, chunks_version_txt)

    def test_version_file_count_mismatch_fails(self, chunks_jsonl_path,
                                               chunks_version_txt, base_cfg):
        from build_index import load_chunks, map_chunk, validate_chunk_versions, ChunkContractError
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        chunks_version_txt.write_text(
            "chunking version   : v3\ncorpus version     : v2\n"
            "preprocess version : v2\n청크 수       : 9999\n", encoding="utf-8")
        with pytest.raises(ChunkContractError, match="청크 수"):
            validate_chunk_versions(chunks, base_cfg, chunks_version_txt)

    def test_version_file_ok(self, chunks_jsonl_path, chunks_version_txt, base_cfg):
        from build_index import load_chunks, map_chunk, validate_chunk_versions
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        declared = validate_chunk_versions(chunks, base_cfg, chunks_version_txt)
        assert declared["chunking_version"] == "v3"

    @pytest.mark.parametrize("field", ["document_version", "processed_sha256",
                                       "sidecar_sha256"])
    def test_registry_hash_mismatch_blocks_index(self, chunks_jsonl_path, registry_path,
                                                 field):
        from build_index import (load_chunks, map_chunk, load_registry_documents,
                                 cross_check_registry, ChunkContractError)
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        docs = load_registry_documents(registry_path)
        for c in chunks:
            if c["document_id"] == "RFP-000002":
                c[field] = "9" * 64
        with pytest.raises(ChunkContractError, match=field):
            cross_check_registry(chunks, docs)

    def test_registry_missing_document_blocks_index(self, chunks_jsonl_path,
                                                    registry_path):
        from build_index import (load_chunks, map_chunk, load_registry_documents,
                                 cross_check_registry, ChunkContractError)
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        docs = load_registry_documents(registry_path)
        chunks = [c for c in chunks if c["document_id"] != "RFP-000003"]
        with pytest.raises(ChunkContractError, match="청크에 없는 문서"):
            cross_check_registry(chunks, docs)

    def test_registry_extra_document_blocks_index(self, chunks_jsonl_path,
                                                  registry_path):
        from build_index import (load_chunks, map_chunk, load_registry_documents,
                                 cross_check_registry, ChunkContractError)
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        docs = load_registry_documents(registry_path)
        extra = dict(chunks[0])
        extra["document_id"] = "RFP-999999"
        extra["chunk_id"] = "RFP-999999-0001"
        chunks.append(extra)
        with pytest.raises(ChunkContractError, match="등록부에 없는 문서"):
            cross_check_registry(chunks, docs)

    def test_registry_reports_eligibility_and_duplicates(self, chunks_jsonl_path,
                                                         registry_path):
        from build_index import (load_chunks, map_chunk, load_registry_documents,
                                 cross_check_registry)
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        docs = load_registry_documents(registry_path)
        report = cross_check_registry(chunks, docs)
        assert report["hash_checked_documents"] == 4
        assert report["retrieval_eligible_count"] == 3
        assert report["excluded"][0]["document_id"] == "RFP-000099"
        assert report["excluded"][0]["duplicate_of_document_id"] == "RFP-000001"

    def test_extraction_table_official_counts(self, extraction_table_path,
                                              extraction_metadata_path, base_cfg):
        from table_query import load_extraction_table, ExtractionTableFormatError
        with pytest.raises(ExtractionTableFormatError, match="1200행"):
            load_extraction_table(extraction_table_path, cfg=base_cfg,
                                  metadata_path=extraction_metadata_path, official=True)

    def test_extraction_table_version_mismatch(self, extraction_table_path, base_cfg):
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc = json.loads(extraction_table_path.read_text(encoding="utf-8"))
        doc["document_count"] = 100
        doc["row_count"] = 1200
        cfg = dict(base_cfg)
        cfg["extraction_version"] = "v2"
        with pytest.raises(ExtractionTableFormatError):
            validate_extraction_table(doc, cfg=cfg, official=True)

    def test_extraction_table_missing_field_detected(self, extraction_table_path):
        from table_query import load_extraction_table, ExtractionTableFormatError
        doc = json.loads(extraction_table_path.read_text(encoding="utf-8"))
        doc["rows"] = [r for r in doc["rows"]
                       if not (r["document_id"] == "RFP-000002" and r["field_name"] == "예산")]
        doc["row_count"] = len(doc["rows"])
        doc["field_count"] = 12
        p = extraction_table_path.parent / "missing_field.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ExtractionTableFormatError, match="누락된 필드"):
            load_extraction_table(p)

    def test_official_full_size_table_passes(self, base_cfg):
        from table_query import validate_extraction_table
        doc, meta = _full_size_table_doc()
        validate_extraction_table(doc, cfg=base_cfg, metadata=meta, official=True)

    def test_extraction_metadata_mismatch_detected(self, base_cfg):
        """실제 집계와 extraction_metadata.json이 다르면 막아야 한다."""
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc, meta = _full_size_table_doc()
        meta["row_count"] = 999
        with pytest.raises(ExtractionTableFormatError, match="metadata"):
            validate_extraction_table(doc, cfg=base_cfg, metadata=meta, official=True)

    def test_official_row_and_document_counts_enforced(self, base_cfg):
        """99문서(1,188행)짜리 표는 공식 규모 검사에서 막혀야 한다."""
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc, meta = _full_size_table_doc()
        doc["rows"] = [r for r in doc["rows"] if r["document_id"] != "RFP-000100"]
        doc["row_count"] = len(doc["rows"])
        doc["document_count"] = 99
        meta["row_count"] = doc["row_count"]
        meta["document_count"] = 99
        with pytest.raises(ExtractionTableFormatError) as ei:
            validate_extraction_table(doc, cfg=base_cfg, metadata=meta, official=True)
        assert "1200행" in str(ei.value) or "100문서" in str(ei.value)

    def test_document_count_rule_enforced_at_full_row_count(self, base_cfg):
        """행 수가 1,200이어도 문서 수가 100이 아니면 막는다."""
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc, meta = _full_size_table_doc()
        # 마지막 문서의 12행을 새 문서 2건으로 쪼개 문서 수만 101로 만든다(행 수 유지)
        tail = doc["rows"][-12:]
        for i, r in enumerate(tail):
            r["document_id"] = "RFP-000101" if i < 6 else "RFP-000102"
        doc["document_count"] = 101
        meta["document_count"] = 101
        with pytest.raises(ExtractionTableFormatError) as ei:
            validate_extraction_table(doc, cfg=base_cfg, metadata=meta, official=True)
        msg = str(ei.value)
        assert "100문서" in msg or "누락된 필드" in msg

    def test_unexpected_field_name_rejected(self, base_cfg):
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc, meta = _full_size_table_doc()
        for r in doc["rows"]:
            if r["field_name"] == "예산":
                r["field_name"] = "사업예산"      # 공식 12필드 이름과 다름
        doc["fields"] = sorted({r["field_name"] for r in doc["rows"]})
        with pytest.raises(ExtractionTableFormatError, match="공식 12필드"):
            validate_extraction_table(doc, cfg=base_cfg, metadata=meta, official=True)

    def test_row_level_version_must_match_header(self, base_cfg):
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc, meta = _full_size_table_doc()
        doc["rows"][0]["extraction_version"] = "v2"
        with pytest.raises(ExtractionTableFormatError, match="extraction_version"):
            validate_extraction_table(doc, cfg=base_cfg, metadata=meta, official=True)


# ===========================================================================
# 4-3 근거 위치·응답 형식
# ===========================================================================

class TestEvidenceAndResponseShape:
    def test_chunk_record_preserves_all_fields(self, small_store):
        from answer_pipeline import chunk_to_record
        rec = chunk_to_record(small_store.metadata[0])
        for key in ("document_id", "section_path", "section_paths", "block_type",
                    "block_index", "md_line_start", "md_line_end",
                    "location_label", "search_text"):
            assert key in rec
        assert rec["block_index"] is not None, "텍스트 청크의 block_index를 비우면 안 됨"
        assert rec["section_paths"], "section_paths가 있으면 보존해야 함"

    def test_citation_minimum_shape(self, table_rows):
        from answer_pipeline import row_citations
        from table_query import lookup_field
        row = lookup_field(table_rows, "RFP-000001", "예산")
        cites = row_citations("RFP-000001", row)
        assert cites[0]["document"] == "RFP-000001"
        assert cites[0]["section"] == "2. 사업개요"
        assert "paragraph 6" in cites[0]["ref_no"]
        assert "line 61" in cites[0]["ref_no"]

    def test_all_locations_kept_not_just_first(self, table_rows):
        from answer_pipeline import row_citations
        from table_query import lookup_field
        row = lookup_field(table_rows, "RFP-000001", "예산")
        assert len(row_citations("RFP-000001", row)) == 2

    def test_compare_records_every_doc_field_pair(self, table_rows, base_cfg,
                                                  identity_index):
        from answer_pipeline import answer_compare_by_table
        r = answer_compare_by_table(
            "RFP-000001이랑 RFP-000002 예산이랑 사업기간 비교해줘", table_rows,
            base_cfg, identity=identity_index)
        docs = {c["document"] for c in r.citations}
        assert {"RFP-000001", "RFP-000002"} <= docs
        assert len(r.citations) >= 4

    def test_select_records_conditions(self, table_rows, base_cfg, identity_index):
        from answer_pipeline import answer_select_by_table
        r = answer_select_by_table("4,900만원 이상인 사업", table_rows,
                                   base_cfg, identity=identity_index)
        assert isinstance(r.structured_answer, list)
        assert r.citations
        assert {c["document"] for c in r.citations} == set(r.selected_document_ids)

    def test_response_envelope_is_uniform(self, table_rows, small_store, base_cfg,
                                          identity_index, fake_clients):
        from answer_pipeline import answer, answer_to_response, SessionState
        _, _, get_e, get_g = fake_clients
        required = {"id", "answer", "structured_answer", "contexts", "retrieved",
                    "citations", "selected_document_ids", "abstained", "route",
                    "failure", "latency_ms", "cost_usd"}
        for q in ["1억 이상인 사업 알려줘",
                  "RFP-000001의 예산이 얼마야",
                  "RFP-000001이랑 RFP-000002 예산 비교해줘",
                  "RFP-000001의 추진배경이 뭐야",
                  "안녕하세요",
                  "이 시스템은 어떻게 사용하나요?"]:
            r = answer(q, small_store, get_e, get_g, table_rows, base_cfg,
                       identity=identity_index, session=SessionState())
            assert set(answer_to_response("qid", r)) == required, q


# ===========================================================================
# 4-5 GPT-5 Mini 요청 인자 (모의)
# ===========================================================================

class TestGpt5MiniRequestArgs:
    def test_no_unsupported_params(self, base_cfg):
        from generation_client import build_request_kwargs
        kw = build_request_kwargs(base_cfg, "gpt-5-mini",
                                  [{"role": "user", "content": "x"}])
        assert "temperature" not in kw
        assert "top_p" not in kw
        assert "logprobs" not in kw
        assert "max_tokens" not in kw
        assert kw["max_completion_tokens"] == 4096
        assert kw["model"] == "gpt-5-mini"

    def test_max_completion_tokens_configurable(self, base_cfg):
        from generation_client import build_request_kwargs
        cfg = dict(base_cfg)
        cfg["max_completion_tokens"] = 1024
        assert build_request_kwargs(cfg, "gpt-5-mini", [])["max_completion_tokens"] == 1024

    def test_temperature_never_sent_even_if_set_for_gpt5(self, base_cfg):
        from generation_client import build_request_kwargs
        cfg = dict(base_cfg)
        cfg["temperature"] = 0
        cfg["top_p"] = 1
        kw = build_request_kwargs(cfg, "gpt-5-mini", [])
        assert "temperature" not in kw and "top_p" not in kw

    def test_other_model_settings_do_not_leak(self, base_cfg):
        from generation_client import build_request_kwargs
        cfg = dict(base_cfg)
        cfg["temperature"] = 0.7
        assert "temperature" not in build_request_kwargs(cfg, "gpt-5-mini", [])
        assert build_request_kwargs(cfg, "gpt-4o-mini", [])["temperature"] == 0.7

    def test_api_key_never_in_request(self, base_cfg):
        from generation_client import build_request_kwargs
        blob = json.dumps(build_request_kwargs(
            base_cfg, "gpt-5-mini", [{"role": "user", "content": "x"}]),
            ensure_ascii=False)
        assert "sk-" not in blob and "api_key" not in blob

    def test_base_yaml_declares_pricing_per_1m(self):
        import yaml
        root = Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load((root / "config" / "base.yaml").read_text(encoding="utf-8"))
        assert cfg["pricing_unit"] == "per_1m_tokens"
        assert cfg["pricing"]["gpt-5-mini"]["input_per_1m"] == 0.25
        assert cfg["pricing"]["gpt-5-mini"]["cached_input_per_1m"] == 0.025
        assert cfg["pricing"]["gpt-5-mini"]["output_per_1m"] == 2.0
        assert cfg["pricing"]["text-embedding-3-small"]["input_per_1m"] == 0.02
        assert cfg["temperature"] is None
        assert cfg["max_completion_tokens"] == 4096
        assert "max_tokens" not in cfg


# ===========================================================================
# 5-1 비용 격리
# ===========================================================================

def _write_index(tmp_path, store, **over):
    from conftest import write_valid_index
    return write_valid_index(tmp_path, store, **over)


def _run_eval(tmp_path, base_yaml_path, extraction_table_path, identity_csv_path,
              store, monkeypatch, items, extra_argv=(), tag="run"):
    """run_eval.main()을 모의 클라이언트로 실행하고 결과 폴더를 반환한다."""
    import importlib
    import run_eval
    from conftest import FakeEmbed, FakeGen

    idx = _write_index(tmp_path, store)
    evalset = tmp_path / f"evalset_{tag}.jsonl"
    evalset.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items),
                       encoding="utf-8")
    out = tmp_path / f"out_{tag}"

    monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
    importlib.reload(run_eval)
    fake_e, fake_g = FakeEmbed(), FakeGen()
    monkeypatch.setattr(run_eval, "EmbeddingClient", lambda cfg: fake_e)
    monkeypatch.setattr(run_eval, "GenerationClient", lambda cfg: fake_g)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--evalset", str(evalset), "--index", str(idx),
        "--extraction-table", str(extraction_table_path),
        "--identity", str(identity_csv_path),
        "--allow-unofficial-table", "--allow-no-chunks",
        "--out", str(out), *extra_argv])
    run_eval.main()
    return out


def _details(out):
    return [json.loads(l) for l in
            (out / "details.jsonl").read_text(encoding="utf-8").splitlines()]


class TestCostIsolation:
    def test_cost_per_1m_math(self, base_cfg):
        from pricing import Usage, compute_cost
        u = Usage()
        u.add_generation(1_000_000, 1_000_000, cached_tokens=0)
        _, detail = compute_cost(base_cfg, u)
        assert detail["generation_cost_usd"] == pytest.approx(2.25)
        assert detail["embedding_cost_usd"] == 0.0

    def test_cached_input_priced_separately(self, base_cfg):
        from pricing import Usage, compute_cost
        u = Usage()
        u.add_generation(1_000_000, 0, cached_tokens=1_000_000)
        _, detail = compute_cost(base_cfg, u)
        assert detail["generation_cost_usd"] == pytest.approx(0.025)

    def test_no_api_call_means_zero_generation_cost(self, base_cfg):
        from pricing import Usage, compute_cost
        total, detail = compute_cost(base_cfg, Usage())
        assert total == 0.0
        assert detail["generation_cost_usd"] == 0.0
        assert detail["embedding_cost_usd"] == 0.0

    def test_reset_usage_clears_previous_item(self):
        from conftest import FakeGen
        g = FakeGen()
        g.generate("q", ["c"])
        assert g.usage.generation_requests == 1
        g.reset_usage()
        assert g.usage.generation_requests == 0
        assert g.usage.generation_input_tokens == 0

    def test_consecutive_items_do_not_share_cost(self, tmp_path, extraction_table_path,
                                                 identity_csv_path, small_store,
                                                 base_yaml_path, monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="cost", items=[
                            {"id": "q1", "question": "RFP-000001의 추진배경이 뭐야",
                             "task_type": "qa"},
                            {"id": "q2", "question": "RFP-000001의 예산이 얼마야",
                             "task_type": "extraction"},
                        ])
        d = _details(out)
        assert d[0]["called_generation_api"] is True and d[0]["cost_usd"] > 0
        assert d[1]["called_generation_api"] is False
        assert d[1]["called_embedding_api"] is False
        assert d[1]["cost_usd"] == 0.0
        assert d[1]["cost_detail"]["generation_cost_usd"] == 0.0

    def test_embedding_and_generation_cost_separated(self, tmp_path,
                                                     extraction_table_path,
                                                     identity_csv_path, small_store,
                                                     base_yaml_path, monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="split", items=[
                            {"id": "q1", "question": "RFP-000001의 추진배경이 뭐야",
                             "task_type": "qa"}])
        api = json.loads((out / "api_cost_summary.json").read_text(encoding="utf-8"))
        assert api["cost_breakdown"]["generation_cost_usd"] > 0
        assert api["cost_breakdown"]["embedding_cost_usd"] > 0
        assert api["tokens"]["embedding_tokens"] == 11
        assert api["tokens"]["generation_input_tokens"] == 100


# ===========================================================================
# 5-2 시스템 사용법
# ===========================================================================

class TestSystemHelp:
    def test_system_help_lists_five_capabilities(self, table_rows, small_store,
                                                 base_cfg, identity_index):
        from answer_pipeline import answer, SessionState, ROUTE_SYSTEM_HELP
        from conftest import forbidden
        r = answer("이 시스템은 어떻게 사용하나요?", small_store, forbidden("embed"),
                   forbidden("gen"), table_rows, base_cfg, identity=identity_index,
                   session=SessionState())
        assert r.route == ROUTE_SYSTEM_HELP
        for must in ("조건에 맞는 공고", "12필드", "비교", "일반 문서 질문", "재확인"):
            assert must in r.text
        assert r.text.strip() != "안녕하세요! RFP 관련 질문을 도와드릴게요."

    def test_rfp_budget_usage_question_is_not_system_help(self, table_rows, small_store,
                                                          base_cfg, identity_index,
                                                          fake_clients):
        from answer_pipeline import answer, SessionState, ROUTE_SYSTEM_HELP
        _, _, get_e, get_g = fake_clients
        r = answer("RFP-000001 이 사업의 예산을 어떻게 사용하나요?", small_store,
                   get_e, get_g, table_rows, base_cfg, identity=identity_index,
                   session=SessionState())
        assert r.route != ROUTE_SYSTEM_HELP


# ===========================================================================
# 5-5 실패 종료코드
# ===========================================================================

class _BrokenEmbed:
    def __init__(self, msg="의도적 오류 주입 sk-ABCDEF1234567890"):
        from pricing import Usage
        self.usage = Usage()
        self.msg = msg

    def reset_usage(self):
        from pricing import Usage
        self.usage = Usage()

    def embed_query(self, q):
        raise RuntimeError(self.msg)


class TestFailureExitCodes:
    def test_empty_evalset_fails(self, tmp_path, base_yaml_path, extraction_table_path,
                                 identity_csv_path, small_store, monkeypatch):
        import importlib, run_eval
        idx = _write_index(tmp_path, small_store)
        evalset = tmp_path / "empty.jsonl"
        evalset.write_text("", encoding="utf-8")
        monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
        importlib.reload(run_eval)
        monkeypatch.setattr(sys, "argv", [
            "run_eval.py", "--evalset", str(evalset), "--index", str(idx),
            "--extraction-table", str(extraction_table_path),
            "--identity", str(identity_csv_path), "--allow-unofficial-table",
            "--allow-no-chunks", "--out", str(tmp_path / "out_empty")])
        with pytest.raises(SystemExit) as ei:
            run_eval.main()
        assert ei.value.code == 1

    def test_item_error_fails_run(self, tmp_path, base_yaml_path, extraction_table_path,
                                  identity_csv_path, small_store, monkeypatch):
        import importlib, run_eval
        from conftest import FakeGen
        idx = _write_index(tmp_path, small_store)
        evalset = tmp_path / "err.jsonl"
        evalset.write_text(json.dumps(
            {"id": "q1", "question": "RFP-000001의 추진배경이 뭐야", "task_type": "qa"},
            ensure_ascii=False), encoding="utf-8")
        out = tmp_path / "out_err"
        monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
        importlib.reload(run_eval)
        monkeypatch.setattr(run_eval, "EmbeddingClient", lambda cfg: _BrokenEmbed())
        monkeypatch.setattr(run_eval, "GenerationClient", lambda cfg: FakeGen())
        monkeypatch.setattr(sys, "argv", [
            "run_eval.py", "--evalset", str(evalset), "--index", str(idx),
            "--extraction-table", str(extraction_table_path),
            "--identity", str(identity_csv_path), "--allow-unofficial-table",
            "--allow-no-chunks", "--out", str(out)])
        with pytest.raises(SystemExit) as ei:
            run_eval.main()
        assert ei.value.code == 1
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["error_count"] == 1
        d = _details(out)
        assert d[0]["error_stage"] is not None
        assert "qa" in d[0]["error"]
        assert "sk-ABCDEF" not in d[0]["error"]
        assert d[0]["retries"] == 1

    def test_allow_errors_flag_permits_success(self, tmp_path, base_yaml_path,
                                               extraction_table_path,
                                               identity_csv_path, small_store,
                                               monkeypatch):
        import importlib, run_eval
        from conftest import FakeGen
        idx = _write_index(tmp_path, small_store)
        evalset = tmp_path / "err2.jsonl"
        evalset.write_text(json.dumps(
            {"id": "q1", "question": "RFP-000001의 추진배경이 뭐야", "task_type": "qa"},
            ensure_ascii=False), encoding="utf-8")
        monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
        importlib.reload(run_eval)
        monkeypatch.setattr(run_eval, "EmbeddingClient",
                            lambda cfg: _BrokenEmbed("의도적 오류"))
        monkeypatch.setattr(run_eval, "GenerationClient", lambda cfg: FakeGen())
        monkeypatch.setattr(sys, "argv", [
            "run_eval.py", "--evalset", str(evalset), "--index", str(idx),
            "--extraction-table", str(extraction_table_path),
            "--identity", str(identity_csv_path), "--allow-unofficial-table",
            "--allow-no-chunks", "--allow-errors", "--out", str(tmp_path / "out_err2")])
        run_eval.main()

    def test_single_question_error_exit_code(self, tmp_path, base_yaml_path,
                                             extraction_table_path,
                                             identity_csv_path, small_store,
                                             monkeypatch):
        import importlib, answer_pipeline
        from conftest import FakeGen
        idx = _write_index(tmp_path, small_store)
        monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
        importlib.reload(answer_pipeline)
        monkeypatch.setattr(answer_pipeline, "EmbeddingClient",
                            lambda cfg: _BrokenEmbed("의도적 오류"))
        monkeypatch.setattr(answer_pipeline, "GenerationClient", lambda cfg: FakeGen())
        monkeypatch.setattr(sys, "argv", [
            "answer_pipeline.py", "--question", "RFP-000001의 추진배경이 뭐야",
            "--index", str(idx), "--extraction-table", str(extraction_table_path),
            "--identity", str(identity_csv_path), "--allow-unofficial-table",
            "--allow-no-chunks"])
        with pytest.raises(SystemExit) as ei:
            answer_pipeline.main()
        assert ei.value.code == 1

    def test_unlimited_retry_rejected(self, tmp_path, base_yaml_path,
                                      extraction_table_path, identity_csv_path,
                                      small_store, monkeypatch):
        import importlib, run_eval
        idx = _write_index(tmp_path, small_store)
        evalset = tmp_path / "r.jsonl"
        evalset.write_text(json.dumps({"id": "q1", "question": "안녕"},
                                      ensure_ascii=False), encoding="utf-8")
        monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
        importlib.reload(run_eval)
        monkeypatch.setattr(sys, "argv", [
            "run_eval.py", "--evalset", str(evalset), "--index", str(idx),
            "--extraction-table", str(extraction_table_path),
            "--identity", str(identity_csv_path), "--allow-unofficial-table",
            "--allow-no-chunks", "--max-item-retries", "5", "--out", str(tmp_path / "out_r")])
        with pytest.raises(SystemExit) as ei:
            run_eval.main()
        assert ei.value.code == 2

    def test_sanitize_error_removes_pii(self):
        from answer_pipeline import sanitize_error
        out = sanitize_error("키 sk-ABCDEFGHIJKLMN 메일 a.b@c.com 전화 010-1234-5678")
        assert "sk-ABCDEF" not in out
        assert "a.b@c.com" not in out
        assert "010-1234-5678" not in out


# ===========================================================================
# 5-6 세션 분리
# ===========================================================================

class TestSessionSeparation:
    def test_default_independent_sessions(self, tmp_path, base_yaml_path,
                                          extraction_table_path, identity_csv_path,
                                          small_store, monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="indep", items=[
                            {"id": "q1", "question": "RFP-000001의 예산이 얼마야",
                             "task_type": "extraction"},
                            {"id": "q2", "question": "그 사업 지역제한 알려줘",
                             "task_type": "extraction"},
                        ])
        assert _details(out)[1]["abstained"] is True

    def test_continuous_session_flag_links_questions(self, tmp_path, base_yaml_path,
                                                     extraction_table_path,
                                                     identity_csv_path, small_store,
                                                     monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="cont",
                        items=[
                            {"id": "q3", "question": "RFP-000001의 예산이 얼마야",
                             "task_type": "extraction"},
                            {"id": "q4", "question": "그 사업 사업기간 알려줘",
                             "task_type": "extraction"},
                        ],
                        extra_argv=("--continuous-session",))
        d = _details(out)
        assert d[1]["abstained"] is False
        # 후속 문항이 앞 문항의 문서를 이어받았는지는 답변 문자열이 아니라
        # 문서 특정 결과로 확인한다(answer 는 값 자체라 문서 ID가 안 들어간다)
        assert d[1]["condition_result_doc_ids"] == ["RFP-000001"]
        assert d[1]["active_document_after"] == "RFP-000001"

    def test_partial_session_ids_do_not_mix(self, tmp_path, base_yaml_path,
                                            extraction_table_path, identity_csv_path,
                                            small_store, monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="mix", items=[
                            {"id": "s1", "session_id": "A",
                             "question": "RFP-000001의 예산이 얼마야",
                             "task_type": "extraction"},
                            {"id": "s2", "question": "그 사업 사업기간 알려줘",
                             "task_type": "extraction"},
                            {"id": "s3", "session_id": "B",
                             "question": "그 사업 사업기간 알려줘",
                             "task_type": "extraction"},
                        ])
        d = _details(out)
        assert d[1]["abstained"] is True
        assert d[2]["abstained"] is True

    def test_same_session_id_continues(self, tmp_path, base_yaml_path,
                                       extraction_table_path, identity_csv_path,
                                       small_store, monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="same", items=[
                            {"id": "t1", "session_id": "A",
                             "question": "RFP-000001의 예산이 얼마야",
                             "task_type": "extraction"},
                            {"id": "t2", "session_id": "A",
                             "question": "그 사업 사업기간 알려줘",
                             "task_type": "extraction"},
                        ])
        d = _details(out)
        assert d[1]["abstained"] is False
        assert d[1]["condition_result_doc_ids"] == ["RFP-000001"]
        assert d[1]["active_document_after"] == "RFP-000001"

    def test_task_type_route_mismatch_is_not_an_error(self, tmp_path, base_yaml_path,
                                                      extraction_table_path,
                                                      identity_csv_path, small_store,
                                                      monkeypatch):
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="mismatch",
                        items=[{"id": "m1", "question": "RFP-000001의 예산이 얼마야",
                                "task_type": "qa"}])
        d = _details(out)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert d[0]["expected_task_type"] == "qa"
        assert d[0]["actual_task_type"] == "extract"
        assert summary["error_count"] == 0


# ===========================================================================
# 5-3 README·스크립트 정합성 / 응답 형식 검증기
# ===========================================================================

class TestReadmeAndScripts:
    def test_referenced_scripts_exist_and_are_executable(self):
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        for name in ("real_api_smoke_test.sh", "run_practice_evalset.sh",
                     "build_index.py", "answer_pipeline.py", "run_eval.py",
                     "verify_response_format.py"):
            p = scripts / name
            assert p.exists(), f"README가 안내하는 {name}이 없습니다"
            if name.endswith(".sh"):
                assert os.access(p, os.X_OK), f"{name}에 실행 권한이 없습니다"

    def test_readme_route_count_matches_code(self):
        import re as _re
        from answer_pipeline import ALL_ROUTES
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(
            encoding="utf-8")
        m = _re.search(r"route 값 (\d+)가지", readme)
        assert m, "README에 route 개수 표기가 없습니다"
        assert int(m.group(1)) == len(ALL_ROUTES)
        for r in ALL_ROUTES:
            assert f"`{r}`" in readme, f"README에 route {r}가 없습니다"

    def test_readme_uses_v5_extraction_table(self):
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(
            encoding="utf-8")
        assert "extraction_table_v5.json" in readme
        assert "extraction_table_v2.json" not in readme
        assert "rfp_extraction_table_v2" not in readme

    def test_readme_documents_identity_v2_as_deadline_source(self):
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(
            encoding="utf-8")
        assert "document_identity_v2.csv" in readme
        assert "data_list.csv" not in readme

    def test_stray_merge_message_file_removed(self):
        root = Path(__file__).resolve().parents[2]
        strays = [p.name for p in root.iterdir()
                  if p.is_file() and any(0xE000 <= ord(ch) <= 0xF8FF for ch in p.name)]
        assert strays == [], f"이상한 이름의 파일이 남아 있음: {strays}"


class TestResponseFormatVerifier:
    def _write(self, tmp_path, recs):
        p = tmp_path / "responses.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs),
                     encoding="utf-8")
        return p

    def _good(self):
        return {
            "id": "q1", "answer": "답", "structured_answer": None,
            "contexts": [], "retrieved": [],
            "citations": [{"document": "RFP-000001", "section": "2. 사업개요",
                           "ref_no": "paragraph 6 · line 61"}],
            "selected_document_ids": ["RFP-000001"], "abstained": False,
            "route": "추출테이블_값조회", "failure": None,
            "latency_ms": 10, "cost_usd": 0.0,
        }

    def test_valid_record_passes(self, tmp_path):
        from verify_response_format import check_response
        assert check_response(self._good(), 1) == []

    def test_missing_key_detected(self, tmp_path):
        from verify_response_format import check_response
        rec = self._good()
        del rec["citations"]
        assert any("citations" in p for p in check_response(rec, 1))

    def test_dict_section_detected(self):
        from verify_response_format import check_response
        rec = self._good()
        rec["citations"][0]["section"] = {"heading": "x"}
        assert any("문자열이 아닙니다" in p for p in check_response(rec, 1))

    def test_text_chunk_block_index_none_detected(self):
        from verify_response_format import check_response
        rec = self._good()
        rec["contexts"] = [{"document_id": "RFP-000001", "section_path": [],
                            "block_type": "text", "block_index": None,
                            "md_line_start": 1, "md_line_end": 2,
                            "search_text": "x"}]
        assert any("block_index" in p for p in check_response(rec, 1))

    def test_api_key_in_response_detected(self):
        from verify_response_format import check_response
        rec = self._good()
        rec["answer"] = "키는 sk-ABCDEFGHIJKLMNOP 입니다"
        assert any("api_key" in p for p in check_response(rec, 1))

    def test_real_run_eval_output_passes(self, tmp_path, base_yaml_path,
                                         extraction_table_path, identity_csv_path,
                                         small_store, monkeypatch):
        from verify_response_format import load_jsonl, check_response
        out = _run_eval(tmp_path, base_yaml_path, extraction_table_path,
                        identity_csv_path, small_store, monkeypatch, tag="fmt", items=[
                            {"id": "f1", "question": "1억 이상인 사업 알려줘"},
                            {"id": "f2", "question": "RFP-000001의 예산이 얼마야"},
                            {"id": "f3", "question": "RFP-000001의 추진배경이 뭐야"},
                            {"id": "f4", "question": "RFP-000001이랑 RFP-000002 예산 비교해줘"},
                            {"id": "f5", "question": "이 시스템은 어떻게 사용하나요?"},
                        ])
        recs = load_jsonl(out / "responses.jsonl")
        problems = []
        for i, r in enumerate(recs, 1):
            problems.extend(check_response(r, i))
        assert problems == [], problems


class TestRouteAccuracyAndSummaries:
    def test_no_generation_call_is_not_recorded_as_llm_route(self, table_rows, base_cfg,
                                                             identity_index):
        """구조화 값만으로 답하고 생성 API를 안 태웠으면 LLM 경로로 기록하지 않는다."""
        from vector_store import VectorStore
        from answer_pipeline import (answer_qa_or_extract_by_search,
                                     build_deadline_evidence, ROUTE_IDENTITY_VALUE,
                                     ROUTE_STRUCTURED_LLM)
        from conftest import FakeEmbed, FakeGen
        empty_store = VectorStore()
        ev = build_deadline_evidence("RFP-000001", identity_index, base_cfg)
        g = FakeGen()
        r = answer_qa_or_extract_by_search("마감일 언제야", empty_store, FakeEmbed(), g,
                                           base_cfg, structured=[ev])
        assert g.calls == 0
        assert r.route == ROUTE_IDENTITY_VALUE
        assert r.route != ROUTE_STRUCTURED_LLM

    def test_compare_keeps_conflict_framing(self, table_rows, base_cfg, identity_index):
        """비교표 한 줄에서도 conflict가 정상 값처럼 보이면 안 된다."""
        from answer_pipeline import answer_compare_by_table
        r = answer_compare_by_table(
            "RFP-000001이랑 RFP-000002 컨소시엄 비교해줘", table_rows, base_cfg,
            identity=identity_index)
        assert "상충" in r.text
        assert r.abstained is True

    def test_compare_keeps_status_framing(self, table_rows, base_cfg, identity_index):
        from answer_pipeline import answer_compare_by_table
        r = answer_compare_by_table(
            "RFP-000001이랑 RFP-000003 예산 비교해줘", table_rows, base_cfg,
            identity=identity_index)
        assert "외부 공고문" in r.text   # external_reference를 값처럼 쓰지 않는다

    def test_qa_reports_unknown_org_even_when_document_found(
        self, table_rows, small_store, base_cfg, identity_index, fake_clients,
    ):
        from answer_pipeline import answer, SessionState
        _, _, get_e, get_g = fake_clients
        r = answer("벤처기업협회 사업 추진배경이랑 가짜미래재단 사업을 알려줘",
                   small_store, get_e, get_g, table_rows, base_cfg,
                   identity=identity_index, session=SessionState())
        assert "가짜미래재단" in r.text
        assert "공식 목록에 없는 기관" in r.text


# ===========================================================================
# 6 회귀 방지 — 이미 고쳐진 기능이 되돌아가지 않았는지
# ===========================================================================

def _run_build_index(tmp_path, base_yaml_path, chunks_path, monkeypatch,
                     extra_argv=(), registry=None, version_file=None):
    import importlib, build_index
    from conftest import FakeEmbed

    class BatchEmbed(FakeEmbed):
        def embed_batch(self, texts, batch_size=100, max_retries=3):
            self.usage.add_embedding(sum(len(t) for t in texts))
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
    importlib.reload(build_index)
    fake = BatchEmbed()
    fake.model = "text-embedding-3-small"
    monkeypatch.setattr(build_index, "EmbeddingClient", lambda cfg: fake)
    argv = ["build_index.py", "--chunks", str(chunks_path),
            "--index-out", str(tmp_path / "idx_built")]
    if registry:
        argv += ["--registry", str(registry)]
    if version_file:
        argv += ["--chunks-version-file", str(version_file)]
    argv += list(extra_argv)
    monkeypatch.setattr(sys, "argv", argv)
    return build_index


class TestBuildIndexRegressions:
    def test_registry_required_by_default(self, tmp_path, base_yaml_path,
                                          chunks_jsonl_path, chunks_version_txt,
                                          monkeypatch):
        bi = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                              version_file=chunks_version_txt)
        with pytest.raises(SystemExit) as ei:
            bi.main()
        assert ei.value.code == 1

    def test_allow_no_registry_is_explicit_opt_in(self, tmp_path, base_yaml_path,
                                                  chunks_jsonl_path, chunks_version_txt,
                                                  monkeypatch):
        bi = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                              extra_argv=("--allow-no-registry",),
                              version_file=chunks_version_txt)
        bi.main()
        assert (tmp_path / "idx_built" / "index_tag.json").exists()

    def test_chunks_version_file_required_by_default(self, tmp_path, base_yaml_path,
                                                     chunks_jsonl_path, registry_path,
                                                     monkeypatch):
        """청크 옆에 VERSION.txt가 없고 RAG_ROOT도 못 찾으면 막아야 한다."""
        lone = tmp_path / "lonely" / "chunks.jsonl"
        lone.parent.mkdir()
        lone.write_bytes(chunks_jsonl_path.read_bytes())
        bi = _run_build_index(tmp_path, base_yaml_path, lone, monkeypatch,
                              registry=registry_path)
        with pytest.raises(SystemExit) as ei:
            bi.main()
        assert ei.value.code == 1

    def test_typo_document_id_in_only_is_blocked(self, tmp_path, base_yaml_path,
                                                 chunks_jsonl_path, registry_path,
                                                 chunks_version_txt, monkeypatch):
        """--only에 등록부에 없는(오타) 문서 ID를 주면 중단해야 한다."""
        bi = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                              extra_argv=("--only", "RFP-00001"),
                              registry=registry_path, version_file=chunks_version_txt)
        with pytest.raises(SystemExit) as ei:
            bi.main()
        assert ei.value.code == 1

    def test_excluded_document_distinguished_from_missing(self, tmp_path,
                                                          base_yaml_path,
                                                          chunks_jsonl_path,
                                                          registry_path,
                                                          chunks_version_txt,
                                                          monkeypatch, capsys):
        """검색 제외 문서(RFP-000099)와 '등록부에 없는 ID'를 구분해야 한다."""
        import json as _json
        reg = _json.loads(registry_path.read_text(encoding="utf-8"))
        # 청크에 없는 RFP-000099는 '이번에 검색 제외로 바뀐 문서'로 다뤄져야 한다
        bi = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                              extra_argv=("--allow-no-registry",),
                              version_file=chunks_version_txt)
        bi.main()   # 전체 생성으로 기존 인덱스 준비
        capsys.readouterr()
        bi2 = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                               extra_argv=("--only", "RFP-000099"),
                               registry=registry_path, version_file=chunks_version_txt)
        bi2.main()
        out = capsys.readouterr().out
        assert "검색 제외로 바뀐 문서" in out
        assert "등록부에 아예 없습니다" not in out

    def test_full_build_records_versions_in_tag(self, tmp_path, base_yaml_path,
                                                chunks_jsonl_path, registry_path,
                                                chunks_version_txt, monkeypatch):
        bi = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                              registry=registry_path, version_file=chunks_version_txt)
        bi.main()
        tag = json.loads((tmp_path / "idx_built" / "index_tag.json").read_text(
            encoding="utf-8"))
        assert tag["corpus_version"] == "v2"
        assert tag["chunking_version"] == "v3"
        assert tag["registry_version"] == "v2"
        assert tag["extraction_version"] == "v3"
        assert tag["document_count"] == 3   # 등록부 4건 중 청크가 있는 3건

    def test_index_replacement_is_all_or_nothing(self, tmp_path, small_store):
        """벡터·메타데이터·설정 파일이 반쪽으로 섞이지 않는다."""
        from vector_store import VectorStore, IndexTag
        idx = tmp_path / "idx_atomic"
        tag = IndexTag(chunk_size=1500, chunk_overlap=150, embedding_model="m",
                       embedding_provider="openai", preprocess_version="v2",
                       corpus_version="v2", build_timestamp="t1")
        small_store.save(idx, tag)
        before = (idx / "metadata.jsonl").read_text(encoding="utf-8")

        broken = VectorStore()
        import numpy as np
        broken.vectors = np.array([[1.0, 2.0]], dtype=np.float32)

        class Boom(list):
            def __iter__(self):
                raise RuntimeError("저장 도중 실패")

        broken.metadata = Boom()
        with pytest.raises(RuntimeError):
            broken.save(idx, tag)
        # 실패 후에도 기존 인덱스가 그대로 남아 있어야 한다
        assert (idx / "metadata.jsonl").read_text(encoding="utf-8") == before
        assert (idx / "vectors.npy").exists() and (idx / "index_tag.json").exists()
        assert VectorStore.load(idx).metadata


class TestEmbeddingUsageAccumulation:
    def test_multi_batch_usage_is_accumulated(self, base_cfg, monkeypatch):
        """회귀 — 예전엔 마지막 배치의 토큰만 기록해 인덱싱 비용이 크게 과소 집계됐다."""
        import embedding_client as ec

        class FakeResp:
            def __init__(self, n):
                self.data = [type("D", (), {"embedding": [0.0, 1.0]})() for _ in range(n)]
                self.usage = type("U", (), {"total_tokens": 10 * n,
                                            "prompt_tokens": 10 * n})()

        class FakeEmbeddings:
            def create(self, model, input):
                return FakeResp(len(input))

        client = ec.EmbeddingClient.__new__(ec.EmbeddingClient)
        client.client = type("C", (), {"embeddings": FakeEmbeddings()})()
        client.model = "text-embedding-3-small"
        from pricing import Usage
        client.usage = Usage()

        vectors = client.embed_batch([f"t{i}" for i in range(250)], batch_size=100)
        assert len(vectors) == 250
        assert client.usage.embedding_requests == 3      # 100+100+50
        assert client.usage.embedding_tokens == 2500     # 마지막 배치(500)만이면 회귀
