"""
PR #10 2차 수정분 회귀 테스트 (2026-09-02) — 실제 API 호출 0건.

  1-1 낡거나 잘못된 인덱스를 실행 전에 차단 (차단 시 API 호출 0회)
  1-2 문서 후보가 여러 개인 QA 는 임의 검색 없이 되묻기
  1-3 쉼표·접속사 없는 복합 조건도 전부 판독
  1-4 citations 에는 실제 근거만 (contexts/retrieved 와 분리)
  2-1 배치 임베딩 사용량 전체 누적
"""
from __future__ import annotations

import json
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
# 1-1 인덱스 태그 검증
# ===========================================================================

class TestIndexTagValidation:
    def _idx(self, tmp_path, small_store, **over):
        from conftest import write_valid_index
        return write_valid_index(tmp_path, small_store, **over)

    def test_valid_tag_passes(self, tmp_path, small_store, base_cfg):
        from vector_store import validate_index_tag
        out = validate_index_tag(self._idx(tmp_path, small_store), base_cfg)
        assert out["vector_dimension"] == 2
        assert out["tag"]["chunking_version"] == "v3"

    @pytest.mark.parametrize("over,needle", [
        ({"chunking_version": "v1"}, "chunking_version"),
        ({"extraction_version": "v2"}, "extraction_version"),
        ({"corpus_version": "v1"}, "corpus_version"),
        ({"registry_version": "v1"}, "registry_version"),
        ({"preprocess_version": "v1"}, "preprocess_version"),
        ({"embedding_model": "text-embedding-3-large"}, "embedding_model"),
        ({"embedding_provider": "local"}, "embedding_provider"),
        ({"chunk_size": 700}, "chunk_size"),
        ({"chunk_overlap": 50}, "chunk_overlap"),
    ])
    def test_stale_or_wrong_tag_blocked(self, tmp_path, small_store, base_cfg,
                                        over, needle):
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store, **over)
        with pytest.raises(IndexTagError) as ei:
            validate_index_tag(idx, base_cfg)
        assert needle in str(ei.value)

    def test_missing_tag_file_blocked(self, tmp_path, small_store, base_cfg):
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store)
        (idx / "index_tag.json").unlink()
        with pytest.raises(IndexTagError, match="꼬리표가 없습니다"):
            validate_index_tag(idx, base_cfg)

    def test_broken_json_blocked(self, tmp_path, small_store, base_cfg):
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store)
        (idx / "index_tag.json").write_text("{ this is not json", encoding="utf-8")
        with pytest.raises(IndexTagError, match="JSON"):
            validate_index_tag(idx, base_cfg)

    def test_tag_not_an_object_blocked(self, tmp_path, small_store, base_cfg):
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store)
        (idx / "index_tag.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(IndexTagError, match="객체가 아닙니다"):
            validate_index_tag(idx, base_cfg)

    @pytest.mark.parametrize("drop", [
        "chunking_version", "corpus_version", "embedding_model",
        "vector_dimension", "registry_version", "extraction_version",
        "build_timestamp", "chunk_size",
    ])
    def test_required_key_missing_blocked(self, tmp_path, small_store, base_cfg, drop):
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store)
        tag = json.loads((idx / "index_tag.json").read_text(encoding="utf-8"))
        tag.pop(drop)
        (idx / "index_tag.json").write_text(json.dumps(tag), encoding="utf-8")
        with pytest.raises(IndexTagError, match="필수 항목이 없습니다") as ei:
            validate_index_tag(idx, base_cfg)
        assert drop in str(ei.value)

    def test_vector_dimension_mismatch_blocked(self, tmp_path, small_store, base_cfg):
        """꼬리표가 말하는 차원과 실제 vectors.npy 가 다르면 막는다."""
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store, vector_dimension=1536)
        with pytest.raises(IndexTagError, match="vector_dimension"):
            validate_index_tag(idx, base_cfg)

    def test_missing_vectors_file_blocked(self, tmp_path, small_store, base_cfg):
        from vector_store import validate_index_tag, IndexTagError
        idx = self._idx(tmp_path, small_store)
        (idx / "vectors.npy").unlink()
        with pytest.raises(IndexTagError, match="vectors.npy"):
            validate_index_tag(idx, base_cfg)

    def test_official_stale_index_would_be_blocked(self, base_cfg):
        """공식 index_v1(chunking v1 / extraction v2)은 현재 설정에서 차단돼야 한다."""
        from vector_store import validate_index_tag, IndexTagError
        official = Path("/srv/rfp/shared_data/processed/index_v1")
        if not (official / "index_tag.json").exists():
            pytest.skip("공식 index_v1 없음")
        with pytest.raises(IndexTagError) as ei:
            validate_index_tag(official, base_cfg)
        assert "chunking_version" in str(ei.value)

    def test_blocked_index_makes_zero_api_calls(self, tmp_path, small_store,
                                                base_yaml_path, extraction_table_path,
                                                identity_csv_path, monkeypatch):
        """인덱스 검증 실패 시 임베딩·생성 클라이언트를 만들지도, 호출하지도 않는다."""
        import importlib
        import answer_pipeline
        from conftest import write_valid_index
        from vector_store import IndexTagError

        idx = write_valid_index(tmp_path, small_store, chunking_version="v1")
        created = {"embed": 0, "gen": 0}

        def _boom_embed(cfg):
            created["embed"] += 1
            raise AssertionError("인덱스 검증 실패 시 임베딩 클라이언트를 만들면 안 됨")

        def _boom_gen(cfg):
            created["gen"] += 1
            raise AssertionError("인덱스 검증 실패 시 생성 클라이언트를 만들면 안 됨")

        monkeypatch.setenv("RAG_CONFIG_PATH", str(base_yaml_path))
        importlib.reload(answer_pipeline)
        monkeypatch.setattr(answer_pipeline, "EmbeddingClient", _boom_embed)
        monkeypatch.setattr(answer_pipeline, "GenerationClient", _boom_gen)
        monkeypatch.setattr(sys, "argv", [
            "answer_pipeline.py", "--question", "RFP-000001의 추진배경이 뭐야",
            "--index", str(idx), "--extraction-table", str(extraction_table_path),
            "--identity", str(identity_csv_path), "--allow-unofficial-table",
            "--allow-no-chunks"])
        with pytest.raises(IndexTagError):
            answer_pipeline.main()
        assert created == {"embed": 0, "gen": 0}

    def test_build_index_and_runtime_share_one_validator(self):
        """기준 이중화 금지 — 빌드와 실행이 같은 함수를 쓴다."""
        import build_index
        import answer_pipeline
        from vector_store import validate_index_tag
        assert build_index.validate_index_tag is validate_index_tag
        assert answer_pipeline.validate_index_tag is validate_index_tag


# ===========================================================================
# 1-2 문서 후보가 여러 개인 QA
# ===========================================================================

@pytest.fixture
def multi_candidate_identity(tmp_path):
    from identity_metadata import load_identity
    header = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')
    rows = [
        '"RFP-000001","a.md","","","","1","0","한국수자원공사","A 통합정보시스템 고도화",'
        '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
        '"RFP-000002","b.md","","","","2","0","한국수자원공사","B 데이터 플랫폼 구축",'
        '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
        '"RFP-000003","c.md","","","","3","0","한국수자원공사","C 관제시스템 유지보수",'
        '"2024-01-01 10:00:00","2025-01-01 17:00:00","true"',
    ]
    p = tmp_path / "identity_multi_qa.csv"
    p.write_text("﻿" + "\n".join([header] + rows) + "\n", encoding="utf-8")
    return load_identity(p)


class TestMultiCandidateQa:
    def _run(self, question, store, table, cfg, identity, clients, session=None):
        from answer_pipeline import answer, SessionState
        e, g, get_e, get_g = clients
        return answer(question, store, get_e, get_g, table, cfg,
                      identity=identity, session=session or SessionState())

    def test_multi_candidate_qa_asks_back_without_api(
        self, small_store, table_rows, base_cfg, multi_candidate_identity, fake_clients,
    ):
        from answer_pipeline import ROUTE_CLARIFY
        e, g, _, _ = fake_clients
        r = self._run("한국수자원공사 사업 추진배경이 뭐야", small_store, table_rows,
                      base_cfg, multi_candidate_identity, fake_clients)
        assert r.abstained is True
        assert r.route == ROUTE_CLARIFY
        # ⭐ 검색·LLM 호출 0회
        assert e.calls == 0 and g.calls == 0

    def test_candidate_ids_present_in_response(
        self, small_store, table_rows, base_cfg, multi_candidate_identity, fake_clients,
    ):
        r = self._run("한국수자원공사 사업 추진배경이 뭐야", small_store, table_rows,
                      base_cfg, multi_candidate_identity, fake_clients)
        for doc in ("RFP-000001", "RFP-000002", "RFP-000003"):
            assert doc in r.text
            assert doc in r.selected_document_ids
        # 후보 정보는 구조화 결과에도 보존
        assert r.structured_answer["clarification_needed"] is True
        assert set(r.structured_answer["candidates"]) == {
            "RFP-000001", "RFP-000002", "RFP-000003"}
        assert any("한국수자원공사" in lab
                   for lab in r.structured_answer["candidate_labels"])

    def test_never_falls_back_to_whole_corpus_search(
        self, small_store, table_rows, base_cfg, multi_candidate_identity, fake_clients,
    ):
        """임의로 첫 문서를 고르지도, 코퍼스 전체 검색으로 넘어가지도 않는다."""
        r = self._run("한국수자원공사 사업 추진배경이 뭐야", small_store, table_rows,
                      base_cfg, multi_candidate_identity, fake_clients)
        assert r.contexts == [] and r.retrieved == []
        assert r.citations == []

    def test_active_document_id_resolves_and_proceeds(
        self, small_store, table_rows, base_cfg, multi_candidate_identity, fake_clients,
    ):
        from answer_pipeline import SessionState
        e, g, _, _ = fake_clients
        session = SessionState(active_document_id="RFP-000002")
        r = self._run("거기 추진배경이 뭐야", small_store, table_rows, base_cfg,
                      multi_candidate_identity, fake_clients, session=session)
        assert r.abstained is False
        assert e.calls >= 1 and g.calls >= 1
        assert r.selected_document_ids == ["RFP-000002"]

    def test_explicit_document_id_resolves_and_proceeds(
        self, small_store, table_rows, base_cfg, multi_candidate_identity, fake_clients,
    ):
        e, g, _, _ = fake_clients
        r = self._run("RFP-000001의 추진배경이 뭐야", small_store, table_rows,
                      base_cfg, multi_candidate_identity, fake_clients)
        assert r.abstained is False
        assert e.calls >= 1 and g.calls >= 1

    def test_unknown_org_is_distinguished_from_many_candidates(
        self, small_store, table_rows, base_cfg, multi_candidate_identity, fake_clients,
    ):
        """존재하지 않는 기관과 '후보가 여러 개'는 다른 상태다."""
        e, g, _, _ = fake_clients
        unknown = self._run("가짜미래재단에서 발주한 사업 요약해줘", small_store,
                            table_rows, base_cfg, multi_candidate_identity, fake_clients)
        assert unknown.abstained is True
        assert "찾을 수 없습니다" in unknown.text
        assert unknown.resolution["unknown_orgs"] == ["가짜미래재단"]
        assert not unknown.selected_document_ids

        many = self._run("한국수자원공사 사업 추진배경이 뭐야", small_store, table_rows,
                         base_cfg, multi_candidate_identity, fake_clients)
        assert many.abstained is True
        assert len(many.selected_document_ids) == 3
        assert not many.resolution.get("unknown_orgs")
        assert e.calls == 0 and g.calls == 0

    def test_single_candidate_still_runs(
        self, small_store, table_rows, base_cfg, identity_index, fake_clients,
    ):
        e, g, _, _ = fake_clients
        r = self._run("벤처기업협회 사업 추진배경이 뭐야", small_store, table_rows,
                      base_cfg, identity_index, fake_clients)
        assert r.abstained is False
        assert r.selected_document_ids == ["RFP-000001"]
        assert e.calls >= 1 and g.calls >= 1


# ===========================================================================
# 1-3 구분자 없는 복합 조건
# ===========================================================================

class TestSeparatorlessCompoundConditions:
    @pytest.mark.parametrize("question", [
        "예산 5억 이상이고 지역제한 없는 사업",
        "예산 5억 이상, 지역제한 없는 사업",
        "예산 5억 이상 지역제한 없는 사업",
    ])
    def test_two_conditions_any_separator(self, question):
        from table_query import parse_conditions
        conds, full = parse_conditions(question)
        assert full is True
        assert len(conds) == 2
        assert {c.field for c in conds} == {"예산", "지역제한"}
        budget = next(c for c in conds if c.field == "예산")
        assert (budget.operator, budget.value) == (">=", 500_000_000.0)
        region = next(c for c in conds if c.field == "지역제한")
        assert region.operator == "no_restriction"

    def test_comma_number_with_separatorless_second_condition(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("예산 49,500만원 이상 지역제한 있는 사업")
        assert full is True and len(conds) == 2
        assert conds[0].value == 495_000_000.0        # 49,500 이 안 쪼개짐
        assert conds[1].operator == "has_restriction"

    def test_three_conditions_in_one_sentence(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("예산 3억 이상 5억 이하 지역제한 없는 사업")
        assert full is True and len(conds) == 3
        ops = sorted((c.operator, c.value) for c in conds if c.field == "예산")
        assert ops == [("<=", 500_000_000.0), (">=", 300_000_000.0)]

    @pytest.mark.parametrize("question", [
        "1억 이상이고 발주기관이 서울인 사업",
        "예산 5억 이상 담당자가 김철수인 사업",
        "예산 5억 이상 사업기간이 6개월인 사업",
    ])
    def test_unknown_condition_is_not_silently_dropped(self, question):
        """일부만 읽고 fully_matched=True 로 성공 처리하지 않는다."""
        from table_query import parse_conditions
        conds, full = parse_conditions(question)
        assert full is False
        assert len(conds) >= 1     # 읽은 것은 남기되, 전부 읽었다고 하지 않는다

    def test_partially_read_conditions_trigger_clarify(self, table_rows, base_cfg,
                                                       identity_index):
        from answer_pipeline import answer_select_by_table, ROUTE_CLARIFY
        r = answer_select_by_table("예산 5억 이상 담당자가 김철수인 사업", table_rows,
                                   base_cfg, identity=identity_index)
        assert r.abstained is True
        assert r.route == ROUTE_CLARIFY
        assert "인식하지 못" in r.text

    def test_existing_single_condition_behaviour_kept(self):
        from table_query import parse_conditions
        for q, op, val in [("5억 이상인 사업", ">=", 500_000_000.0),
                           ("5억 초과인 사업", ">", 500_000_000.0),
                           ("5억 넘는 사업", ">", 500_000_000.0),
                           ("5억 이하인 사업", "<=", 500_000_000.0),
                           ("5억 미만인 사업", "<", 500_000_000.0),
                           ("1억 5천만원 이상인 사업", ">=", 150_000_000.0)]:
            conds, full = parse_conditions(q)
            assert full and len(conds) == 1 and conds[0].operator == op
            assert conds[0].value == val

    def test_number_comma_still_protected(self):
        from table_query import split_conditions, _parse_korean_amount
        assert split_conditions("49,500") == ["49,500"]
        assert _parse_korean_amount("49,500") == 49_500.0
        assert _parse_korean_amount("49,500,000") == 49_500_000.0
        assert _parse_korean_amount("1,234.5") == 1_234.5

    def test_separatorless_query_returns_expected_documents(self, table_rows):
        from table_query import parse_conditions, run_conditions_query
        conds, full = parse_conditions("예산 4,900만원 이상 지역제한 있는 사업")
        assert full and len(conds) == 2
        results, _, _ = run_conditions_query(table_rows, conds)
        assert [r.document_id for r in results] == ["RFP-000002"]


# ===========================================================================
# 1-4 citations = 실제 근거만
# ===========================================================================

@pytest.fixture
def five_chunk_store():
    from vector_store import VectorStore, ChunkMetadata
    store = VectorStore()
    metas, pairs = [], []
    titles = ["표지", "일반현황 및 연혁", "2. 사업목표", "사업지원 요구사항", "붙임"]
    for i, title in enumerate(titles, 1):
        metas.append(ChunkMetadata(
            chunk_id=f"RFP-000001-000{i}", document_id="RFP-000001",
            document_version="1", processed_sha256="x", sidecar_sha256="y",
            corpus_version="v2", text=f"{title} 본문", document_name="문서",
            chapter=title, section_path=[title], section_paths=[[title]],
            location_label=f"{title} · 문단 1-{i}",
            md_line_start=i * 10, md_line_end=i * 10 + 5,
        ))
        pairs.append((f"{title} 본문", [1.0 - i * 0.01, 0.0]))
    store.upsert_document("RFP-000001", pairs, metas)
    return store


class TestCitationsAreRealEvidenceOnly:
    def _search(self, store, cfg, gen, structured=None, doc="RFP-000001"):
        from answer_pipeline import answer_qa_or_extract_by_search
        from conftest import FakeEmbed
        return answer_qa_or_extract_by_search(
            "질문", store, FakeEmbed(), gen, cfg, document_id=doc,
            structured=structured or [])

    def test_only_declared_evidence_is_cited(self, five_chunk_store, base_cfg):
        """검색 후보 5개 중 답에 실제로 쓴 1개만 인용한다."""
        from conftest import FakeGen
        cfg = dict(base_cfg); cfg["top_k"] = 5
        gen = FakeGen(use_evidence=(3,))          # 3번(2. 사업목표)만 사용
        r = self._search(five_chunk_store, cfg, gen)
        assert len(r.retrieved) == 5              # 후보는 그대로 보존
        assert len(r.contexts) == 5               # LLM 입력도 그대로 보존
        assert len(r.citations) == 1              # 인용은 실제 쓴 것만
        assert r.citations[0]["section"] == "2. 사업목표"
        # 표지·일반현황·사업지원 요구사항 같은 무관한 청크는 인용되지 않는다
        cited = {c["section"] for c in r.citations}
        assert not ({"표지", "일반현황 및 연혁", "사업지원 요구사항"} & cited)

    def test_contexts_and_retrieved_not_auto_copied(self, five_chunk_store, base_cfg):
        from conftest import FakeGen
        cfg = dict(base_cfg); cfg["top_k"] = 5
        gen = FakeGen(use_evidence=())            # 원문 근거 미사용 선언
        r = self._search(five_chunk_store, cfg, gen)
        assert len(r.contexts) == 5 and len(r.retrieved) == 5
        assert r.citations == []
        assert r.citation_diagnostics["protocol"] == "none_declared"

    def test_protocol_violation_yields_no_citations(self, five_chunk_store, base_cfg):
        """모델이 규약을 안 지키면 근거를 추측해 붙이지 않는다."""
        from conftest import FakeGen
        cfg = dict(base_cfg); cfg["top_k"] = 5
        gen = FakeGen(use_evidence=None)
        r = self._search(five_chunk_store, cfg, gen)
        assert r.citations == []
        assert r.citation_diagnostics["protocol"] == "missing"

    def test_unknown_evidence_id_is_rejected(self, five_chunk_store, base_cfg):
        """존재하지 않는 근거 ID는 인용으로 인정하지 않고 진단에만 남긴다."""
        from conftest import FakeGen
        cfg = dict(base_cfg); cfg["top_k"] = 5
        gen = FakeGen(use_evidence=(2, 99), clamp=False)
        r = self._search(five_chunk_store, cfg, gen)
        assert len(r.citations) == 1
        assert r.citations[0]["section"] == "일반현황 및 연혁"   # E2 만 인정
        assert r.citation_diagnostics["unknown_evidence_ids"] == [99]

    def test_structured_only_answer_cites_structured_source_only(
        self, base_cfg, identity_index,
    ):
        """구조화 값만으로 답하면 그 구조화 출처만 인용한다."""
        from vector_store import VectorStore
        from answer_pipeline import (answer_qa_or_extract_by_search,
                                     build_deadline_evidence)
        from conftest import FakeEmbed, FakeGen
        ev = build_deadline_evidence("RFP-000001", identity_index, base_cfg)
        gen = FakeGen()
        r = answer_qa_or_extract_by_search("마감일 언제야", VectorStore(), FakeEmbed(),
                                           gen, base_cfg, structured=[ev])
        assert gen.calls == 0
        assert len(r.citations) == 1
        assert_identity_deadline_citation(r.citations[0])

    def test_deadline_qa_does_not_cite_unrelated_chunks(
        self, five_chunk_store, base_cfg, identity_index,
    ):
        """PRAC-QA-001 형: 마감일 답의 인용에 표지·일반현황이 들어가면 안 된다."""
        from answer_pipeline import build_deadline_evidence
        from conftest import FakeGen
        cfg = dict(base_cfg); cfg["top_k"] = 5
        ev = build_deadline_evidence("RFP-000001", identity_index, cfg, eligibility=True)
        gen = FakeGen(use_evidence=())     # 설명에 원문을 쓰지 않았다고 선언
        r = self._search(five_chunk_store, cfg, gen, structured=[ev])
        assert len(r.citations) == 1
        assert_identity_deadline_citation(r.citations[0])
        cited = {c.get("section") for c in r.citations}
        assert not ({"표지", "일반현황 및 연혁", "사업지원 요구사항"} & cited)

    def test_compare_cites_only_compared_doc_field_pairs(
        self, table_rows, base_cfg, identity_index,
    ):
        from answer_pipeline import answer_compare_by_table
        r = answer_compare_by_table("RFP-000001이랑 RFP-000002 예산 비교해줘",
                                    table_rows, base_cfg, identity=identity_index)
        assert {c["document"] for c in r.citations} == {"RFP-000001", "RFP-000002"}
        assert all(c.get("field") == "예산" for c in r.citations)
        # 비교하지 않은 필드의 근거는 들어가지 않는다
        assert len(r.citations) == 2

    def test_missing_evidence_and_wrong_evidence_are_distinguishable(
        self, table_rows, base_cfg,
    ):
        """하루님 채점 기준(3-4-3)에서 '근거 누락'과 '잘못된 근거'가 갈리도록,
        **없는 좌표를 지어내지 않는다**. [2026-09-04 §8] 위치가 없는 미기재 행은
        빈 인용 대신 Evidence 형(kind/document/field/status)으로 낸다 — 좌표 키는 없다."""
        from answer_pipeline import build_field_evidence
        absent = build_field_evidence(table_rows, "RFP-000001", "지역제한", base_cfg)
        assert len(absent.citations) == 1
        cite = absent.citations[0]
        assert cite["kind"] == "extraction_table" and cite["status"] == "field_absent"
        assert cite["document"] == "RFP-000001" and cite["field"] == "지역제한"
        assert not any(k in cite for k in
                       ("location", "line", "ref_no", "section", "block_index"))
        present = build_field_evidence(table_rows, "RFP-000001", "예산", base_cfg)
        assert len(present.citations) == 1     # 실제 위치 하나만 → 오인용 없음
        assert present.citations[0]["line"] == 61

    def test_conflict_keeps_every_real_location(self, table_rows, base_cfg):
        from answer_pipeline import build_field_evidence
        ev = build_field_evidence(table_rows, "RFP-000002", "컨소시엄 요건", base_cfg)
        assert {c["line"] for c in ev.citations} == {385, 669}


class TestChunkLocatorCoordinates:
    def test_locator_uses_team_coordinate_convention(self):
        from chunk_locator import ChunkLocator
        loc = ChunkLocator.from_records([
            {"document_id": "RFP-000043", "section_path": ["4. 제안 요청내용"],
             "location_label": "4. 제안 요청내용 · 문단 1-57",
             "md_line_start": 247, "md_line_end": 362, "chunk_id": "c1"},
        ])
        hit = loc.locate("RFP-000043", 252)
        assert hit is not None
        assert hit.section == "4. 제안 요청내용"
        assert hit.ref_no == "4. 제안 요청내용 · 문단 1-57"
        assert (hit.line, hit.line_end) == (247, 362)

    def test_locator_returns_none_outside_any_chunk(self):
        from chunk_locator import ChunkLocator
        loc = ChunkLocator.from_records([
            {"document_id": "D1", "section_path": ["s"], "location_label": "s · 문단 1",
             "md_line_start": 10, "md_line_end": 20, "chunk_id": "c1"}])
        assert loc.locate("D1", 999) is None
        assert loc.locate("D1", None) is None
        assert loc.locate("없는문서", 15) is None

    def test_locator_picks_narrowest_containing_chunk(self):
        from chunk_locator import ChunkLocator
        loc = ChunkLocator.from_records([
            {"document_id": "D1", "section_path": ["큰절"], "location_label": "큰절 · 문단 1-99",
             "md_line_start": 1, "md_line_end": 100, "chunk_id": "wide"},
            {"document_id": "D1", "section_path": ["작은절"], "location_label": "작은절 · 문단 1-2",
             "md_line_start": 40, "md_line_end": 45, "chunk_id": "narrow"}])
        assert loc.locate("D1", 42).chunk_id == "narrow"


# ===========================================================================
# 2-1 배치 임베딩 사용량 누적
# ===========================================================================

class TestEmbeddingUsageAccumulationRound2:
    def test_every_batch_counted(self, base_cfg):
        import embedding_client as ec
        from pricing import Usage, compute_cost

        class FakeResp:
            def __init__(self, n):
                self.data = [type("D", (), {"embedding": [0.0, 1.0]})() for _ in range(n)]
                self.usage = type("U", (), {"total_tokens": 1000 * n,
                                            "prompt_tokens": 1000 * n})()

        client = ec.EmbeddingClient.__new__(ec.EmbeddingClient)
        client.client = type("C", (), {"embeddings": type(
            "E", (), {"create": lambda self, model, input: FakeResp(len(input))})()})()
        client.model = "text-embedding-3-small"
        client.usage = Usage()
        client.embed_batch([f"t{i}" for i in range(1000)], batch_size=100)
        assert client.usage.embedding_requests == 10
        assert client.usage.embedding_tokens == 1_000_000
        total, detail = compute_cost(base_cfg, client.usage)
        assert detail["embedding_cost_usd"] == pytest.approx(0.02)


# ===========================================================================
# 3차: 손상된 표만 검색된 경로의 출처 처리
# ===========================================================================

class TestDegradedOnlyPathCitations:
    """정상 청크 0개 + 손상 표만 검색된 경로.

    값을 만들지 않았으므로 `citations` 는 비어야 한다. 확인 위치는 `sources`,
    검색 기록은 `retrieved` 에 남는다(역할 분리 — 결함 1-4와 같은 원칙)."""

    @pytest.fixture
    def degraded_only_store(self):
        from vector_store import VectorStore, ChunkMetadata
        store = VectorStore()
        metas, pairs = [], []
        for i in (1, 2):
            metas.append(ChunkMetadata(
                chunk_id=f"RFP-000001-000{i}", document_id="RFP-000001",
                document_version="1", processed_sha256="x", sidecar_sha256="y",
                corpus_version="v2", text=f"손상된 표 {i} 내용",
                document_name="문서", chapter="2. 표",
                chunk_type="table", table_idx=i, table_degraded=True,
                section_path=["2. 표"], section_paths=[["2. 표"]],
                location_label=f"2. 표 · 표 {i}",
                md_line_start=i * 10, md_line_end=i * 10 + 3,
            ))
            pairs.append((f"손상된 표 {i} 내용", [1.0 - i * 0.01, 0.0]))
        store.upsert_document("RFP-000001", pairs, metas)
        return store

    def _run(self, store, cfg, gen):
        from answer_pipeline import answer_qa_or_extract_by_search
        from conftest import FakeEmbed
        return answer_qa_or_extract_by_search("질문", store, FakeEmbed(), gen, cfg)

    def test_degraded_only_yields_no_citations(self, degraded_only_store, base_cfg):
        from conftest import FakeGen
        gen = FakeGen()
        r = self._run(degraded_only_store, base_cfg, gen)

        assert gen.calls == 0, "손상 표만 있으면 생성 API 를 호출하지 않는다"
        assert r.abstained is True
        assert r.citations == [], "값을 만들지 않았으므로 인용할 근거가 없다"
        assert r.contexts == [], "손상 표는 LLM 컨텍스트에 넣지 않는다"
        assert len(r.retrieved) == 2, "검색 기록은 retrieved 에 남는다"
        assert all("표" in s for s in r.sources) and len(r.sources) == 2, \
            "원문 확인 위치는 sources 에 남는다"
        assert "손상" in r.text
        assert r.citation_diagnostics["protocol"] == "no_generation_degraded_only"
        assert r.citation_diagnostics["degraded_chunks"] == 2

    def test_degraded_chunk_ids_recorded_in_retrieved(self, degraded_only_store, base_cfg):
        from conftest import FakeGen
        r = self._run(degraded_only_store, base_cfg, FakeGen())
        assert set(r.retrieved_chunk_ids) == {"RFP-000001-0001", "RFP-000001-0002"}
        assert all(rec["block_type"] == "table" for rec in r.retrieved)

    def test_normal_chunk_citation_behaviour_unchanged(self, five_chunk_store, base_cfg):
        """기존 정상 청크 인용 동작은 그대로 유지된다."""
        from conftest import FakeGen
        cfg = dict(base_cfg); cfg["top_k"] = 5
        gen = FakeGen(use_evidence=(3,))
        r = self._run(five_chunk_store, cfg, gen)
        assert gen.calls == 1
        assert len(r.citations) == 1 and r.citations[0]["section"] == "2. 사업목표"
        assert len(r.contexts) == 5 and len(r.retrieved) == 5

    def test_structured_citation_behaviour_unchanged(self, base_cfg, identity_index):
        """구조화 자료 인용 동작도 그대로 유지된다."""
        from vector_store import VectorStore
        from answer_pipeline import (answer_qa_or_extract_by_search,
                                     build_deadline_evidence)
        from conftest import FakeEmbed, FakeGen
        ev = build_deadline_evidence("RFP-000001", identity_index, base_cfg)
        gen = FakeGen()
        r = answer_qa_or_extract_by_search("마감일", VectorStore(), FakeEmbed(), gen,
                                           base_cfg, structured=[ev])
        assert gen.calls == 0
        assert len(r.citations) == 1
        assert_identity_deadline_citation(r.citations[0])

    def test_structured_plus_degraded_still_cites_structured_only(
        self, degraded_only_store, base_cfg, identity_index,
    ):
        """구조화 값 + 손상 표가 함께면 구조화 출처만 인용한다."""
        from answer_pipeline import build_deadline_evidence, answer_qa_or_extract_by_search
        from conftest import FakeEmbed, FakeGen
        ev = build_deadline_evidence("RFP-000001", identity_index, base_cfg)
        gen = FakeGen()
        r = answer_qa_or_extract_by_search("마감일", degraded_only_store, FakeEmbed(),
                                           gen, base_cfg, structured=[ev])
        assert gen.calls == 0
        assert len(r.citations) == 1
        assert_identity_deadline_citation(r.citations[0])
        assert all(c.get("source") != "chunks_v3" for c in r.citations)

    def test_response_envelope_still_uniform(self, degraded_only_store, base_cfg):
        from answer_pipeline import answer_to_response
        from conftest import FakeGen
        r = self._run(degraded_only_store, base_cfg, FakeGen())
        required = {"id", "answer", "structured_answer", "contexts", "retrieved",
                    "citations", "selected_document_ids", "abstained", "route",
                    "failure", "latency_ms", "cost_usd"}
        assert set(answer_to_response("q", r)) == required


class TestUsedEvidenceMarkerParsing:
    """실제 gpt-5-mini 가 표식을 문장 중간에 여러 번 내보낸 사례 회귀.

    표식을 못 찾으면 인용이 0건이 되고 표식 문자열이 답변 본문에 새어 나온다."""

    @pytest.mark.parametrize("text,ids,body", [
        ("답변.\nUSED_EVIDENCE: E1, E3", [1, 3], "답변."),
        ("본문\nUSED_EVIDENCE: NONE", [], "본문"),
        ("본문만 있음", None, "본문만 있음"),
        ("본문\nused_evidence : e2", [2], "본문"),
        ("본문 (USED_EVIDENCE: E3)", [3], "본문"),
        ("앞 USED_EVIDENCE: E2 뒤 문장이 이어짐", [2], "앞 뒤 문장이 이어짐"),
    ])
    def test_marker_variants(self, text, ids, body):
        from generation_client import split_used_evidence
        got_body, got_ids = split_used_evidence(text)
        assert got_ids == ids
        assert got_body == body
        assert "USED_EVIDENCE" not in got_body.upper()

    def test_inline_markers_multiple_occurrences_union(self):
        """실제 관측된 형태 — 문장 끝마다 표식이 붙는 경우."""
        from generation_client import split_used_evidence
        real = ("목적\n- 본 사업은 …을 수행한다. (근거: 요구사항 목록) USED_EVIDENCE: E5\n\n"
                "필요성\n- 재해복구시스템 설치가 필요하다. USED_EVIDENCE: E1")
        body, ids = split_used_evidence(real)
        assert ids == [1, 5]
        assert "USED_EVIDENCE" not in body.upper()
        assert "재해복구시스템" in body and "목적" in body

    def test_marker_never_leaks_into_answer(self, five_chunk_store, base_cfg):
        """파서를 거친 답변 본문에 표식이 남지 않고, 선언한 근거만 인용된다."""
        from answer_pipeline import answer_qa_or_extract_by_search
        from conftest import FakeEmbed, FakeGen

        class InlineMarkerGen(FakeGen):
            def generate(self, question, context_chunks, format_instruction="...",
                         structured_context=None):
                from generation_client import split_used_evidence
                raw = "본문입니다. (근거) USED_EVIDENCE: E2\n추가 문장 USED_EVIDENCE: E4"
                body, ids = split_used_evidence(raw)
                self.calls += 1
                self.last_used_evidence = ids
                self.usage.add_generation(50, 10)
                return body

        cfg = dict(base_cfg); cfg["top_k"] = 5
        r = answer_qa_or_extract_by_search("질문", five_chunk_store, FakeEmbed(),
                                           InlineMarkerGen(), cfg)
        assert "USED_EVIDENCE" not in r.text.upper()
        assert len(r.citations) == 2
        assert {c["section"] for c in r.citations} == {"일반현황 및 연혁", "사업지원 요구사항"}
