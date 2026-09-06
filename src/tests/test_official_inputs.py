"""
공식 입력 자료를 **읽기 전용**으로 검증하는 계약 테스트 + 연습용 8문항 모의 종단 테스트.

공식 자료가 없는 환경(다른 머신 등)에서는 통째로 skip된다.
⚠️ 어떤 공식 파일도 쓰지 않는다. OpenAI API도 호출하지 않는다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

RAG_ROOT = Path(os.environ.get("RAG_ROOT_OFFICIAL", "/srv/rfp"))
PROCESSED = RAG_ROOT / "shared_data" / "processed"
REGISTRY_DIR = PROCESSED / "document_registry_v2"
EXTRACTION_DIR = PROCESSED / "rfp_extraction_table_v4"
CHUNKS_DIR = PROCESSED / "chunks_v3"
IDENTITY_CSV = REGISTRY_DIR / "document_identity_v2.csv"
PRACTICE_ITEMS = RAG_ROOT / "evalset" / "practice_items.jsonl"

pytestmark = pytest.mark.skipif(
    not (IDENTITY_CSV.exists() and EXTRACTION_DIR.exists() and CHUNKS_DIR.exists()),
    reason="공식 입력 자료(/srv/rfp)가 없는 환경",
)

OFFICIAL_CFG = {
    "top_k": 5, "corpus": "v2", "preprocess": "v2", "table": "v4",
    "document_registry_version": "v2", "chunking_version": "v3",
    "extraction_version": "v4", "schema_version": "1-12-2/v3",
    "routing_method": "rule_based", "routing_fallback": "qa",
    "reference_datetime_source": "external", "reference_datetime": "2024-06-01",
    "deadline_filter_field": "bid_deadline",
    "deadline_missing_policy": "show_as_unknown",
    "deadline_filter_disclosure": False,
    "deadline_filter_default": {"select": False, "extract": False,
                                "qa": False, "compare": False},
    "generation_model": "gpt-5-mini", "embedding_model": "text-embedding-3-small",
    "pricing_unit": "per_1m_tokens",
    "pricing": {"gpt-5-mini": {"input_per_1m": 0.25, "cached_input_per_1m": 0.025,
                               "output_per_1m": 2.0},
                "text-embedding-3-small": {"input_per_1m": 0.02}},
}


@pytest.fixture(scope="module")
def official_table():
    from table_query import load_extraction_table
    return load_extraction_table(
        EXTRACTION_DIR / "extraction_table_v4.json", cfg=OFFICIAL_CFG,
        metadata_path=EXTRACTION_DIR / "extraction_metadata.json", official=True)


@pytest.fixture(scope="module")
def official_identity():
    from identity_metadata import load_identity
    return load_identity(IDENTITY_CSV)


class TestOfficialExtractionTable:
    def test_official_contract(self, official_table):
        """1,200행 · 100문서 · 문서당 12필드 · 중복 0 · 누락 0 · 예상 밖 0."""
        from collections import Counter
        from table_query import OFFICIAL_FIELDS
        assert len(official_table) == 1200
        docs = {r["document_id"] for r in official_table}
        fields = {r["field_name"] for r in official_table}
        assert len(docs) == 100
        assert fields == set(OFFICIAL_FIELDS)
        combos = Counter((r["document_id"], r["field_name"]) for r in official_table)
        assert max(combos.values()) == 1
        assert len(combos) == 1200

    def test_declared_versions(self):
        doc = json.loads((EXTRACTION_DIR / "extraction_table_v4.json").read_text(
            encoding="utf-8"))
        assert doc["schema_version"] == "1-12-2/v3"
        assert doc["extraction_version"] == "v4"
        assert doc["corpus_version"] == "v2"
        assert doc["registry_version"] == "v2"

    def test_metadata_agrees_with_aggregate(self, official_table):
        meta = json.loads((EXTRACTION_DIR / "extraction_metadata.json").read_text(
            encoding="utf-8"))
        assert meta["row_count"] == len(official_table)
        assert meta["document_count"] == len({r["document_id"] for r in official_table})
        assert meta["field_count"] == 12

    def test_stale_version_is_blocked(self):
        """폴더·파일 이름만 v4인 것을 인정하지 않는다."""
        from table_query import load_extraction_table, ExtractionTableFormatError
        stale = dict(OFFICIAL_CFG)
        stale["extraction_version"] = "v2"
        with pytest.raises(ExtractionTableFormatError):
            load_extraction_table(EXTRACTION_DIR / "extraction_table_v4.json",
                                  cfg=stale, official=True)

    def test_rfp_000028_conflict_keeps_both_locations(self, official_table):
        """RFP-000028 / 컨소시엄 요건 — L385(불허)와 L669(허용)를 모두 보존."""
        from answer_pipeline import build_field_evidence
        ev = build_field_evidence(official_table, "RFP-000028", "컨소시엄 요건",
                                  OFFICIAL_CFG)
        assert ev.status == "conflict"
        assert ev.abstained is True
        lines = {c["line"] for c in ev.citations}
        assert 385 in lines and 669 in lines
        assert "불허" in ev.answer_text and "허용" in ev.answer_text


class TestOfficialIdentity:
    def test_identity_shape(self, official_identity):
        from identity_metadata import IDENTITY_REQUIRED_COLUMNS
        assert len(official_identity.records) == 100
        rec = official_identity.get("RFP-000038")
        assert rec is not None
        assert rec.buyer_org == "서민금융진흥원"
        assert rec.bid_deadline.strftime("%Y-%m-%d %H:%M") == "2024-06-24 16:00"

    def test_missing_deadlines_are_reported_not_filled(self, official_identity):
        missing = [d for d, r in official_identity.records.items()
                   if r.bid_deadline is None]
        assert len(missing) == 8   # 미상 8건 — 다른 자료로 메우지 않는다

    def test_no_false_unknown_org_across_all_orgs(self, official_identity):
        from doc_resolver import detect_unknown_orgs
        for key, docs in official_identity.by_org_key.items():
            rec = official_identity.get(docs[0])
            assert detect_unknown_orgs(f"{rec.buyer_org} 사업 예산 알려줘",
                                       official_identity) == []


class TestOfficialChunks:
    def test_version_file_matches_config(self):
        from build_index import parse_chunks_version_file
        declared = parse_chunks_version_file(CHUNKS_DIR / "VERSION.txt")
        assert declared["chunking_version"] == "v3"
        assert declared["corpus_version"] == "v2"
        assert declared["preprocess_version"] == "v2"

    def test_registry_cross_check_on_real_data(self):
        """공식 청크 100문서가 등록부와 버전·해시까지 일치하는지."""
        from build_index import (map_chunk, load_registry_documents,
                                 cross_check_registry, validate_chunk_versions)
        chunks = []
        with open(CHUNKS_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(map_chunk(json.loads(line)))
        validate_chunk_versions(chunks, OFFICIAL_CFG, CHUNKS_DIR / "VERSION.txt")
        docs = load_registry_documents(REGISTRY_DIR / "document_registry_v2.json")
        report = cross_check_registry(chunks, docs)
        assert report["registry_document_count"] == 100
        assert report["chunk_document_count"] == 100
        assert report["retrieval_eligible_count"] == 98
        assert len(report["excluded"]) == 2

    def test_stale_chunking_version_blocked(self):
        from build_index import map_chunk, validate_chunk_versions, ChunkContractError
        with open(CHUNKS_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
            first = map_chunk(json.loads(f.readline()))
        stale = dict(OFFICIAL_CFG)
        stale["chunking_version"] = "v1"
        with pytest.raises(ChunkContractError):
            validate_chunk_versions([first], stale, CHUNKS_DIR / "VERSION.txt")


class TestOfficialDocumentResolution:
    """연습 문항이 실제 공식 identity_v2에서 기대 문서를 찾는지."""

    @pytest.mark.parametrize("question,expected", [
        # EXT-001 기관명
        ("서울특별시교육청 지능정보화전략계획의 사업 예산 알려줘.", "RFP-000043"),
        # EXT-002 사업명만 (기관명 없음)
        ("서민금융 채팅 상담시스템 구축 사업에 입찰하려면 어떤 서류가 필요해?", "RFP-000038"),
        # EXT-003 기관명만
        ("서민금융진흥원 사업 마감일이 언제야?", "RFP-000038"),
        # EXT-004 (사) 접두어 차이
        ("벤처기업협회의 2024년 벤처확인종합관리시스템 기능 고도화 용역사업에 "
         "입찰 참가 지역 제한이 있어?", "RFP-000001"),
        # QA-001 기관명+사업명
        ("오늘(2024.6.1)을 기준으로 서민금융진흥원 서민금융 채팅 상담시스템 구축 "
         "사업에 입찰 참여할 수 있어?", "RFP-000038"),
    ])
    def test_practice_document_resolution(self, official_identity, question, expected):
        from doc_resolver import resolve_document
        r = resolve_document(question, official_identity)
        assert r.document_id == expected, f"method={r.method} cands={r.candidates}"

    def test_qa002_anaphora_uses_active_document(self, official_identity):
        from doc_resolver import resolve_document, RESOLVE_ACTIVE
        r = resolve_document("거기 소요예산은 얼마야?", official_identity,
                             active_document_id="RFP-000001")
        assert r.document_id == "RFP-000001" and r.method == RESOLVE_ACTIVE

    def test_qa003_comparison_finds_both(self, official_identity):
        from doc_resolver import resolve_documents_for_compare
        docs, unknown = resolve_documents_for_compare(
            "서민금융진흥원 서민금융 채팅 상담시스템 구축 사업이랑 서울특별시교육청 "
            "지능정보화전략계획(ISP) 사업 예산이랑 사업기간만 비교해줘.",
            official_identity)
        assert set(docs) == {"RFP-000038", "RFP-000043"}
        assert unknown == []

    def test_qa004_unknown_org_detected(self, official_identity):
        from doc_resolver import resolve_document
        r = resolve_document(
            "은하수정보진흥재단에서 발주한 우주정거장 데이터센터 구축사업 제안요청서 "
            "좀 요약해줘.", official_identity)
        assert r.document_id is None
        assert "은하수정보진흥재단" in r.unknown_orgs


class TestOfficialConditionQueries:
    def test_region_restricted_documents(self, official_table):
        from table_query import parse_conditions, run_conditions_query
        conds, full = parse_conditions("지역제한 있는 사업 알려줘")
        assert full
        results, _, _ = run_conditions_query(official_table, conds, max_results=200)
        assert sorted(r.document_id for r in results) == [
            "RFP-000002", "RFP-000005", "RFP-000035", "RFP-000054", "RFP-000060"]

    def test_region_unrestricted_is_empty_not_guessed(self, official_table):
        """빈칸(field_absent) 95건을 '제한 없음'으로 추정하지 않는다."""
        from table_query import parse_conditions, run_conditions_query
        conds, _ = parse_conditions("지역제한 없는 사업 알려줘")
        results, _, warnings = run_conditions_query(official_table, conds,
                                                    max_results=200)
        assert results == []
        assert len(warnings) == 95

    @pytest.mark.parametrize("phrase", [
        "5억 이상인 사업", "5억 초과인 사업", "5억 이하인 사업", "5억 미만인 사업",
        "5억 넘는 사업",
    ])
    def test_operators_run_on_official_table(self, official_table, phrase):
        from table_query import parse_conditions, run_conditions_query
        conds, full = parse_conditions(phrase)
        assert full and conds[0].value == 500_000_000.0
        run_conditions_query(official_table, conds, max_results=200)

    def test_operator_result_sets_are_consistent(self, official_table):
        from table_query import parse_conditions, run_conditions_query

        def ids(phrase):
            conds, _ = parse_conditions(phrase)
            res, _, _ = run_conditions_query(official_table, conds, max_results=500)
            return {r.document_id for r in res}

        ge, gt = ids("5억 이상인 사업"), ids("5억 초과인 사업")
        le, lt = ids("5억 이하인 사업"), ids("5억 미만인 사업")
        assert gt <= ge and lt <= le           # 초과 ⊆ 이상, 미만 ⊆ 이하
        assert ge & lt == set()                # 이상과 미만은 겹치지 않음
        assert (ge - gt) == (le - lt)          # 경계값(정확히 5억)이 양쪽 차집합

    def test_comma_amount_on_official_table(self, official_table):
        from table_query import parse_conditions, run_conditions_query
        conds, full = parse_conditions("예산 49,500만원 이상, 지역제한 있음")
        assert full and len(conds) == 2
        assert conds[0].value == 495_000_000.0
        run_conditions_query(official_table, conds, max_results=200)


# ---------------------------------------------------------------------------
# 연습용 8문항 모의 종단 테스트 (API 호출 0건)
# ---------------------------------------------------------------------------

MOCK_DIM = 8


class _FakeEmbedDim:
    """모의 인덱스 차원에 맞는 가짜 임베딩 클라이언트."""

    def __init__(self, dim=MOCK_DIM):
        from pricing import Usage
        self.dim = dim
        self.calls = 0
        self.usage = Usage()

    def reset_usage(self):
        from pricing import Usage
        self.usage = Usage()

    def embed_query(self, q):
        self.calls += 1
        self.usage.add_embedding(len(q))
        return [1.0] + [0.0] * (self.dim - 1)


@pytest.fixture(scope="module")
def practice_items():
    return [json.loads(l) for l in PRACTICE_ITEMS.read_text(
        encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def mock_store():
    """연습 문항이 가리키는 문서의 실제 청크로 만든 모의 인덱스(가짜 벡터)."""
    import numpy as np
    from vector_store import VectorStore, ChunkMetadata
    wanted = {"RFP-000001", "RFP-000038", "RFP-000043"}
    if True:
        by_doc: dict[str, list] = {d: [] for d in wanted}
        with open(CHUNKS_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                if c["document_id"] in wanted and c.get("retrieval_eligible", True):
                    if len(by_doc[c["document_id"]]) < 40:
                        by_doc[c["document_id"]].append(c)
        store = VectorStore()
        rng = np.random.RandomState(0)
        for doc_id, chunks in by_doc.items():
            metas, pairs = [], []
            for c in chunks:
                metas.append(ChunkMetadata(
                    chunk_id=c["chunk_id"], document_id=c["document_id"],
                    document_version=c["document_version"],
                    processed_sha256=c["processed_sha256"],
                    sidecar_sha256=c["sidecar_sha256"],
                    corpus_version=c["corpus_version"], text=c["search_text"],
                    document_name=c.get("source_document_title", ""),
                    chapter=" > ".join(str(x) for x in (c.get("section_path") or [])),
                    chunk_type=c["block_type"], table_idx=c.get("table_idx"),
                    part=c.get("part"), of=c.get("of"),
                    table_degraded=bool(c.get("table_degraded")),
                    location_label=c.get("location_label", ""),
                    section_path=c.get("section_path") or [],
                    section_paths=c.get("section_paths") or [],
                    md_line_start=c.get("md_line_start"),
                    md_line_end=c.get("md_line_end"),
                ))
                pairs.append((c["search_text"], rng.rand(MOCK_DIM).tolist()))
            if metas:
                store.upsert_document(doc_id, pairs, metas)
        return store


@pytest.mark.skipif(not PRACTICE_ITEMS.exists(), reason="연습 문항 파일 없음")
class TestPractice8Mocked:
    def test_all_eight_items_run_without_error(self, practice_items, mock_store,
                                               official_table, official_identity):
        from answer_pipeline import answer, answer_to_response, SessionState, ALL_ROUTES
        from conftest import FakeGen

        assert len(practice_items) == 8
        required = {"id", "answer", "structured_answer", "contexts", "retrieved",
                    "citations", "selected_document_ids", "abstained", "route",
                    "failure", "latency_ms", "cost_usd"}
        results = {}
        for item in practice_items:
            session = SessionState(active_document_id=item.get("active_document_id"))
            e, g = _FakeEmbedDim(), FakeGen()
            r = answer(item["question"], mock_store, lambda: e, lambda: g,
                       official_table, OFFICIAL_CFG, identity=official_identity,
                       session=session)
            payload = answer_to_response(item["id"], r)
            assert set(payload) == required, item["id"]
            assert r.error_stage is None, f"{item['id']}: {r.error_detail}"
            assert r.route in ALL_ROUTES, f"{item['id']}: route={r.route}"
            results[item["id"]] = r

        # 문항별 기대 동작
        assert "248,796" in results["PRAC-EXT-001"].text
        assert results["PRAC-EXT-001"].selected_document_ids == ["RFP-000043"]
        assert results["PRAC-EXT-002"].selected_document_ids == ["RFP-000038"]
        assert results["PRAC-EXT-003"].selected_document_ids == ["RFP-000038"]
        assert "2024-06-24 16:00" in results["PRAC-EXT-003"].text
        assert results["PRAC-EXT-004"].selected_document_ids == ["RFP-000001"]
        assert results["PRAC-QA-001"].task_type == "qa"
        assert "2024-06-24 16:00" in results["PRAC-QA-001"].text
        assert "참여 가능" in results["PRAC-QA-001"].text
        assert results["PRAC-QA-002"].selected_document_ids == ["RFP-000001"]
        assert "352,000,000" in results["PRAC-QA-002"].text
        assert set(results["PRAC-QA-003"].selected_document_ids) == {
            "RFP-000038", "RFP-000043"}
        assert results["PRAC-QA-004"].abstained is True
        assert "찾을 수 없습니다" in results["PRAC-QA-004"].text

        # 근거는 "실제로 있는 위치"만 남긴다(결함 1-4) — 없으면 지어내지 않는다.
        # RFP-000001/지역제한 은 추출표가 field_absent 이고 representative_location 이
        # 아예 없다. [2026-09-04 §8] 이때도 근거 자체는 있다(추출표의 그 행) —
        # 좌표를 지어내는 대신 Evidence 형으로 낸다. 좌표 키가 하나라도 있으면 실패.
        no_location_expected = {"PRAC-EXT-004"}
        for qid, r in results.items():
            if r.abstained:
                continue
            if qid in no_location_expected:
                assert len(r.citations) == 1, r.citations
                cite = r.citations[0]
                assert cite["kind"] == "extraction_table", cite
                assert cite["status"] == "field_absent", cite
                assert cite["document"] == "RFP-000001" and cite["field"] == "지역제한"
                assert not any(k in cite for k in
                               ("location", "line", "ref_no", "section", "block_index")), (
                    f"{qid}: 위치가 없는 항목인데 좌표를 지어냈다 — {r.citations}")
                continue
            assert r.citations, f"{qid}: citations 비어 있음"
            assert all(c.get("document") for c in r.citations)
