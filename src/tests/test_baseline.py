"""
1단계 자동 테스트 — 비용 없이(OpenAI 실호출 0건) 코드 오류를 먼저 잡는다.
픽스처는 conftest.py에 있고, 실제 공식 파일과 같은 스키마를 쓴다.

실행:
  source /srv/rfp/venv/bin/activate
  cd src && python -m pytest tests -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# router.py — 분기
# ---------------------------------------------------------------------------

class TestRouting:
    _CFG = {"routing_method": "rule_based", "routing_fallback": "qa"}

    @pytest.mark.parametrize("question", [
        "5억 이상인 사업 알려줘",
        "예산 3억 이하인 사업 목록 보여줘",
        "지역제한 없는 사업 추천해줘",
        "1억 넘는 사업 다 보여줘",
        "예산 10억 이하인 사업 리스트",
        "지역제한 있는 사업 전부 보여줘",
        "예산 49,500만원 이상, 지역제한 없음",
    ])
    def test_select(self, question):
        from router import route
        assert route(question, self._CFG).task_type == "select"

    @pytest.mark.parametrize("question", [
        "RFP-000001의 예산이 얼마야",
        "RFP-000002 지역제한 알려줘",
        "RFP-000003의 마감일이 언제야",
        "서울시청 사업 예산 얼마야",
        "이 사업 참가자격이 뭐야",
        "사업기간이 어떻게 돼",
        "필요한 서류가 뭐야",
        "평가배점 알려줘",
        "이 사업의 예산을 어떻게 사용하나요?",
    ])
    def test_extract(self, question):
        from router import route
        assert route(question, self._CFG).task_type == "extract"

    @pytest.mark.parametrize("question", [
        "이 사업의 추진배경이 뭐야",
        "왜 이 사업을 하는 거야",
        "이 프로젝트가 어떤 내용인지 요약해줘",
        "제안요청서 내용 정리해줘",
    ])
    def test_qa_fallback(self, question):
        from router import route
        assert route(question, self._CFG).task_type == "qa"

    @pytest.mark.parametrize("question", [
        "RFP-000001이랑 RFP-000002 예산 비교해줘",
        "두 사업 지역제한 차이가 뭐야",
        "어느 쪽 사업기간이 더 긴지 비교해줘",
    ])
    def test_compare(self, question):
        from router import route
        assert route(question, self._CFG).task_type == "compare"

    def test_system_help_vs_rfp_content(self):
        """5-2 — 시스템 사용법과 RFP 내용 질문을 구분해야 한다."""
        from router import route, NO_SEARCH_SYSTEM_HELP
        r1 = route("이 시스템은 어떻게 사용하나요?", self._CFG)
        assert r1.task_type == "no_search_needed"
        assert r1.no_search_kind == NO_SEARCH_SYSTEM_HELP
        r2 = route("이 사업의 예산을 어떻게 사용하나요?", self._CFG)
        assert r2.task_type == "extract"


# ---------------------------------------------------------------------------
# table_query.py — 금액 파서 회귀
# ---------------------------------------------------------------------------

class TestKoreanAmountParser:
    def test_plain_comma_number(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount("352,000,000원(부가가치세 포함)") == 352_000_000.0

    def test_mixed_unit_expression(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount("1억 5천만 원(부가가치세 포함)") == 150_000_000.0

    def test_thousand_won_unit_not_shortcut(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount("49,500천원(부가가치세 포함)") == 49_500_000.0

    def test_parenthetical_exact_figure_preferred(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount(
            "일금 일억구천오백삼만원정(￦195,030,000, VAT포함)"
        ) == 195_030_000.0

    def test_unparseable_returns_none(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount("") is None
        assert _parse_korean_amount(None) is None


# ---------------------------------------------------------------------------
# table_query.py — 로더·조건 질의
# ---------------------------------------------------------------------------

class TestExtractionTableLoader:
    def test_load_and_validate(self, table_rows):
        assert len(table_rows) == 36  # 3문서 x 12필드

    def test_field_absent_excluded_from_clean_match(self, table_rows):
        from table_query import parse_conditions, run_conditions_query
        conds, full = parse_conditions("지역제한 없는 사업 알려줘")
        assert full
        results, total, warnings = run_conditions_query(table_rows, conds)
        doc_ids = [r.document_id for r in results]
        assert "RFP-000001" not in doc_ids
        assert any("RFP-000001" in w for w in warnings)

    def test_warning_includes_document_id(self, table_rows):
        from table_query import parse_conditions, run_conditions_query
        conds, _ = parse_conditions("지역제한 없는 사업 알려줘")
        _, _, warnings = run_conditions_query(table_rows, conds)
        assert len(warnings) >= 1
        assert all(w.startswith("RFP-") for w in warnings)

    def test_and_compound_condition(self, table_rows):
        from table_query import parse_conditions
        conds, full = parse_conditions("1억 이상이고 지역제한 없는 사업")
        assert full and len(conds) == 2

    def test_unmatched_segment_not_silently_dropped(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("1억 이상이고 발주기관이 서울인 사업")
        assert full is False

    def test_duplicate_document_field_combo_rejected(self, tmp_path, extraction_table_path):
        from table_query import load_extraction_table, ExtractionTableFormatError
        doc = json.loads(extraction_table_path.read_text(encoding="utf-8"))
        dupe = dict(doc["rows"][0])
        doc["rows"].append(dupe)
        doc["row_count"] = len(doc["rows"])
        p = tmp_path / "dupe.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ExtractionTableFormatError):
            load_extraction_table(p)


class TestFieldAndDocumentDetection:
    def test_detect_field(self):
        from table_query import detect_field
        assert detect_field("예산이 얼마야") == "예산"
        assert detect_field("지역제한 있어?") == "지역제한"
        assert detect_field("아무 상관없는 말") is None

    def test_detect_document_id(self):
        from doc_resolver import detect_document_id
        assert detect_document_id("RFP-000001의 예산") == "RFP-000001"
        assert detect_document_id("문서 얘기") is None

    def test_detect_deadline_question(self):
        from table_query import detect_deadline_question
        assert detect_deadline_question("마감일이 언제야?")
        assert not detect_deadline_question("예산이 얼마야")

    def test_lookup_field(self, table_rows):
        from table_query import lookup_field
        row = lookup_field(table_rows, "RFP-000001", "예산")
        assert row is not None and row["status"] == "value_present"


# ---------------------------------------------------------------------------
# identity_metadata.py — 마감일·기관명
# ---------------------------------------------------------------------------

class TestIdentityMetadata:
    def test_load_and_deadline(self, identity_index):
        from datetime import datetime
        assert len(identity_index.records) == 4
        assert identity_index.deadline("RFP-000001") == datetime(2025, 1, 1, 17, 0, 0)
        assert identity_index.deadline("RFP-000003") is None  # 미상

    def test_is_before_deadline_missing_shown_not_excluded(self, identity_index):
        from datetime import datetime
        from identity_metadata import is_before_deadline
        ref = datetime(2024, 6, 1)
        assert is_before_deadline("RFP-000001", identity_index, ref)[0] is True
        assert is_before_deadline("RFP-000002", identity_index, ref)[0] is False
        ok, note = is_before_deadline("RFP-000003", identity_index, ref)
        assert ok is None and "미상" in note

    def test_missing_required_column_raises(self, tmp_path):
        from identity_metadata import load_identity, IdentityFormatError
        p = tmp_path / "bad_identity.csv"
        p.write_text('"document_id","buyer_org"\n"RFP-000001","x"\n', encoding="utf-8")
        with pytest.raises(IdentityFormatError):
            load_identity(p)

    def test_reference_datetime_never_uses_now(self, base_cfg):
        from identity_metadata import reference_datetime_from_config
        from datetime import datetime
        assert reference_datetime_from_config(base_cfg) == datetime(2024, 6, 1)
        bad = dict(base_cfg)
        bad["reference_datetime"] = None
        with pytest.raises(RuntimeError):
            reference_datetime_from_config(bad)


class TestUnknownOrgDetection:
    def test_detect_unknown_org(self, identity_index):
        from doc_resolver import detect_unknown_orgs
        assert detect_unknown_orgs(
            "은하수정보진흥재단에서 발주한 사업 요약해줘", identity_index
        ) == ["은하수정보진흥재단"]
        assert detect_unknown_orgs("벤처기업협회 사업 요약해줘", identity_index) == []
        assert detect_unknown_orgs("이 사업 추진목표가 뭐야", identity_index) == []

    def test_known_and_unknown_together(self, identity_index):
        """회귀 — 알려진 기관이 하나라도 있으면 미확인 기관을 놓치던 버그."""
        from doc_resolver import detect_unknown_orgs
        got = detect_unknown_orgs(
            "벤처기업협회 사업이랑 가짜미래재단 사업을 비교해줘", identity_index)
        assert "가짜미래재단" in got


# ---------------------------------------------------------------------------
# vector_store.py
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_upsert_and_search_roundtrip(self):
        from vector_store import VectorStore, ChunkMetadata
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="D1-0001", document_id="D1", document_version="1",
                              processed_sha256="x", sidecar_sha256="y",
                              corpus_version="v2", text="본문")]
        store.upsert_document("D1", [("본문", [1.0, 0.0])], meta)
        results = store.search([1.0, 0.0], top_k=1)
        assert len(results) == 1 and results[0][0].chunk_id == "D1-0001"

    def test_duplicate_chunk_id_rejected(self):
        from vector_store import VectorStore, ChunkMetadata, VectorStoreError
        store = VectorStore()
        meta = [
            ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                          processed_sha256="x", sidecar_sha256="y", corpus_version="v2"),
            ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                          processed_sha256="x", sidecar_sha256="y", corpus_version="v2"),
        ]
        with pytest.raises(VectorStoreError):
            store.upsert_document("D1", [("a", [1.0]), ("b", [2.0])], meta)

    def test_dimension_mismatch_rejected(self):
        from vector_store import VectorStore, ChunkMetadata, VectorStoreError
        store = VectorStore()
        m1 = [ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                            processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        store.upsert_document("D1", [("a", [1.0, 2.0])], m1)
        m2 = [ChunkMetadata(chunk_id="c2", document_id="D2", document_version="1",
                            processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        with pytest.raises(VectorStoreError):
            store.upsert_document("D2", [("b", [1.0, 2.0, 3.0])], m2)

    def test_top_k_must_be_positive(self):
        from vector_store import VectorStore, ChunkMetadata, VectorStoreError
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                              processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        store.upsert_document("D1", [("a", [1.0])], meta)
        with pytest.raises(VectorStoreError):
            store.search([1.0], top_k=0)

    def test_save_load_roundtrip_atomic(self, tmp_path):
        from vector_store import VectorStore, ChunkMetadata, IndexTag
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                              processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        store.upsert_document("D1", [("a", [1.0, 2.0])], meta)
        tag = IndexTag(chunk_size=1500, chunk_overlap=150, embedding_model="m",
                       embedding_provider="openai", preprocess_version="v2",
                       corpus_version="v2", build_timestamp="t")
        store.save(tmp_path / "idx", tag)
        loaded = VectorStore.load(tmp_path / "idx")
        assert len(loaded.metadata) == 1
        assert (tmp_path / "idx" / "vectors.npy").exists()
        assert (tmp_path / "idx" / "metadata.jsonl").exists()
        assert (tmp_path / "idx" / "index_tag.json").exists()

    def test_config_mismatch_detects_corpus_version(self, tmp_path):
        from dataclasses import asdict
        from vector_store import config_mismatch_check, ConfigMismatchError, IndexTag
        idx_dir = tmp_path / "idx"
        idx_dir.mkdir()
        tag = IndexTag(chunk_size=1500, chunk_overlap=150, embedding_model="m",
                       embedding_provider="openai", preprocess_version="v2",
                       corpus_version="v2", build_timestamp="t")
        (idx_dir / "index_tag.json").write_text(json.dumps(asdict(tag)), encoding="utf-8")
        cfg = {"chunk_size": 1500, "chunk_overlap": 150, "embedding_model": "m",
               "embedding_provider": "openai", "corpus": "v3", "preprocess": "v2",
               "index": "v1"}
        with pytest.raises(ConfigMismatchError):
            config_mismatch_check(idx_dir, cfg)


# ---------------------------------------------------------------------------
# build_index.py — 조용한 손실 방지
# ---------------------------------------------------------------------------

class TestBuildIndexStrictness:
    def test_map_chunk_requires_version_and_hash(self):
        from build_index import map_chunk
        good = {"chunk_id": "c1", "document_id": "D1", "search_text": "본문",
                "block_type": "text", "document_version": "1",
                "processed_sha256": "a" * 64, "sidecar_sha256": "b" * 64}
        assert map_chunk(good)["document_version"] == "1"
        missing = dict(good)
        missing.pop("document_version")
        with pytest.raises(KeyError):
            map_chunk(missing)
        empty = dict(good)
        empty["processed_sha256"] = ""
        with pytest.raises(KeyError):
            map_chunk(empty)

    def test_load_chunks_truncated_file_raises(self, tmp_path):
        from build_index import load_chunks
        p = tmp_path / "chunks.jsonl"
        p.write_text('{"chunk_id": "c1"}\n{"chunk_id": "c2', encoding="utf-8")
        with pytest.raises(ValueError):
            load_chunks(p)

    def test_load_chunks_complete_file_ok(self, chunks_jsonl_path):
        from build_index import load_chunks
        assert len(load_chunks(chunks_jsonl_path)) == 5

    def test_mixed_version_within_document_rejected(self, chunks_jsonl_path):
        from build_index import load_chunks, map_chunk, validate_chunk_consistency
        chunks = [map_chunk(c) for c in load_chunks(chunks_jsonl_path)]
        chunks[1]["document_version"] = "2"
        with pytest.raises(ValueError):
            validate_chunk_consistency(chunks)


# ---------------------------------------------------------------------------
# answer_pipeline — 경로 분리(LLM을 언제 태우는가)
# ---------------------------------------------------------------------------

class TestRouteExecution:
    def _answer(self, question, table, store, cfg, identity=None, session=None,
                clients=None):
        from answer_pipeline import answer, SessionState
        from conftest import forbidden
        if clients is None:
            get_e, get_g = forbidden("embed"), forbidden("gen")
        else:
            _, _, get_e, get_g = clients
        return answer(question, store, get_e, get_g, table, cfg,
                      identity=identity, session=session or SessionState())

    @pytest.mark.parametrize("question", [
        "1억 이상인 사업 알려줘",
        "지역제한 없는 사업 추천해줘",
    ])
    def test_select_never_calls_llm(self, question, table_rows, small_store,
                                    base_cfg, identity_index):
        r = self._answer(question, table_rows, small_store, base_cfg, identity_index)
        assert r.task_type == "select"

    @pytest.mark.parametrize("question", [
        "RFP-000001의 예산이 얼마야",
        "RFP-000002의 지역제한 알려줘",
        "RFP-000001의 마감일이 언제야",
    ])
    def test_simple_extract_never_calls_llm(self, question, table_rows, small_store,
                                            base_cfg, identity_index):
        r = self._answer(question, table_rows, small_store, base_cfg, identity_index)
        assert r.task_type == "extract" and r.abstained is False

    def test_compare_never_calls_llm(self, table_rows, small_store, base_cfg,
                                     identity_index):
        r = self._answer("RFP-000001이랑 RFP-000002 예산 비교해줘", table_rows,
                         small_store, base_cfg, identity_index)
        assert r.task_type == "compare" and r.abstained is False

    def test_qa_does_call_llm(self, table_rows, small_store, base_cfg,
                              identity_index, fake_clients):
        e, g, _, _ = fake_clients
        r = self._answer("RFP-000001의 추진배경이 뭐야", table_rows, small_store,
                         base_cfg, identity_index, clients=fake_clients)
        assert r.task_type == "qa"
        assert e.calls >= 1 and g.calls >= 1
        assert len(r.contexts) >= 1 and len(r.retrieved) >= 1 and len(r.citations) >= 1
        assert all(isinstance(c["section"], str) for c in r.citations)

    def test_extract_with_explanation_does_call_llm(self, table_rows, small_store,
                                                    base_cfg, identity_index,
                                                    fake_clients):
        e, g, _, _ = fake_clients
        r = self._answer("RFP-000001의 예산이 얼마인지 자세히 설명해줘", table_rows,
                         small_store, base_cfg, identity_index, clients=fake_clients)
        assert r.task_type == "extract" and g.calls >= 1

    def test_select_fills_structured_answer(self, table_rows, small_store, base_cfg,
                                            identity_index):
        r = self._answer("1억 이상인 사업 알려줘", table_rows, small_store, base_cfg,
                         identity_index)
        assert isinstance(r.structured_answer, list)
        assert r.selected_document_ids == r.structured_answer

    def test_compare_fills_nested_structured_answer(self, table_rows, small_store,
                                                    base_cfg, identity_index):
        r = self._answer("RFP-000001이랑 RFP-000002 예산 비교해줘", table_rows,
                         small_store, base_cfg, identity_index)
        assert isinstance(r.structured_answer, dict)
        assert "RFP-000001" in r.structured_answer
        assert "예산" in r.structured_answer["RFP-000001"]

    def test_field_absent_not_treated_as_no_restriction(self, table_rows, small_store,
                                                        base_cfg, identity_index):
        r = self._answer("RFP-000001의 지역제한 알려줘", table_rows, small_store,
                         base_cfg, identity_index)
        assert "항목 자체가 없" in r.text
        assert "제한이 없습니다" not in r.text

    def test_missing_document_asks_instead_of_guessing(self, table_rows, small_store,
                                                       base_cfg):
        r = self._answer("예산이 얼마야", table_rows, small_store, base_cfg, None)
        assert r.abstained is True
        assert r.route == "애매_되묻기"

    def test_unknown_org_not_searched(self, table_rows, small_store, base_cfg,
                                      identity_index):
        r = self._answer("은하수정보진흥재단에서 발주한 사업 요약해줘", table_rows,
                         small_store, base_cfg, identity_index)
        assert r.abstained is True and "찾을 수 없습니다" in r.text


class TestDegradedTableExclusion:
    def test_degraded_chunk_excluded_from_generation_context(self, base_cfg):
        from vector_store import VectorStore, ChunkMetadata
        from answer_pipeline import answer_qa_or_extract_by_search
        from conftest import FakeEmbed, FakeGen

        store = VectorStore()
        metas = [
            ChunkMetadata(chunk_id="D1-0001", document_id="D1", document_version="1",
                          processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                          text="정상 본문", table_degraded=False),
            ChunkMetadata(chunk_id="D1-0002", document_id="D1", document_version="1",
                          processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                          text="손상된 표 내용", chunk_type="table", table_idx=1,
                          table_degraded=True),
        ]
        store.upsert_document("D1", [("정상 본문", [1.0, 0.0]),
                                     ("손상된 표 내용", [0.9, 0.1])], metas)

        captured = {}

        class CaptureGen(FakeGen):
            def generate(self, question, context_chunks, format_instruction="...",
                         structured_context=None):
                captured["context"] = context_chunks
                return super().generate(question, context_chunks, format_instruction,
                                        structured_context)

        result = answer_qa_or_extract_by_search(
            "질문", store, FakeEmbed(), CaptureGen(), base_cfg)
        # 손상된 표는 LLM 컨텍스트에 들어가지 않는다(값을 지어내지 못하게)
        assert not any("손상된 표" in c for c in captured["context"])
        # 대신 원문 위치는 sources 로 안내하고, citations 에는 넣지 않는다
        # (실제 답의 근거가 아니므로 — 결함 1-4)
        assert any("표 1" in src for src in result.sources)
        assert all("표 1" not in (c.get("ref_no") or "") for c in result.citations)
