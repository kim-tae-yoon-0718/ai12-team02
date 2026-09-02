"""
1단계 자동 테스트 — 비용 없이(OpenAI 실호출 0건) 코드 오류를 먼저 잡는다.
실제 공식 데이터 스키마를 그대로 따르는 작은 픽스처로 검증한다 —
합성 스키마가 아니라 실제 컬럼명·상태값·필드명 그대로.

실행:
  pip install pytest --break-system-packages
  pytest tests/test_baseline.py -v

⚠️ 이 테스트는 OpenAI API를 호출하지 않는다(embed_batch/generate를 모의
처리). 실제 API 종단 검증은 2단계(scripts/real_api_smoke_test 안내 참고)에서
별도로 한다 — 이 파일의 책임이 아니다.
"""
from __future__ import annotations
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 픽스처 — 실제 공식 파일과 동일한 스키마의 작은 데이터
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_path(tmp_path) -> Path:
    doc = {
        "active_document_count": 4,
        "corpus_version": "v2",
        "document_count": 4,
        "preprocess_version": "v2",
        "registry_format": "test",
        "documents": [
            {"document_id": "RFP-000001", "output_filename": "문서A.md",
             "retrieval_eligible": True, "relation_status": "independent",
             "duplicate_of_document_id": None},
            {"document_id": "RFP-000002", "output_filename": "문서B.md",
             "retrieval_eligible": True, "relation_status": "independent",
             "duplicate_of_document_id": None},
            {"document_id": "RFP-000003", "output_filename": "문서C.md",
             "retrieval_eligible": True, "relation_status": "independent",
             "duplicate_of_document_id": None},
            {"document_id": "RFP-000099", "output_filename": "중복문서.md",
             "retrieval_eligible": False, "relation_status": "duplicate_confirmed",
             "duplicate_of_document_id": "RFP-000001"},
        ],
    }
    p = tmp_path / "document_registry_v2.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def extraction_table_path(tmp_path) -> Path:
    fields = [
        "사업 개요", "사업분야", "공고일", "사업기간", "예산",
        "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건",
        "평가 배점", "제출 방식", "필수 제출 서류", "과업 범위",
    ]
    rows = []
    # RFP-000001: 예산 있음(한글 단위 혼합), 지역제한 field_absent
    for f in fields:
        if f == "예산":
            rows.append({"document_id": "RFP-000001", "field_name": f,
                         "status": "value_present", "answer_raw": "1억 5천만 원(부가가치세 포함)",
                         "answer_normalized": "1억 5천만 원(부가가치세 포함)", "active": "true"})
        elif f == "지역제한":
            rows.append({"document_id": "RFP-000001", "field_name": f,
                         "status": "field_absent", "answer_raw": "", "answer_normalized": "",
                         "active": "true"})
        else:
            rows.append({"document_id": "RFP-000001", "field_name": f,
                         "status": "field_absent", "answer_raw": "", "answer_normalized": "",
                         "active": "true"})
    # RFP-000002: 예산 낮음(천원 단위 오변환 회귀 검증용), 지역제한 명시
    for f in fields:
        if f == "예산":
            rows.append({"document_id": "RFP-000002", "field_name": f,
                         "status": "value_present", "answer_raw": "49,500천원(부가세 포함)",
                         "answer_normalized": "49,500천원(부가세 포함)", "active": "true"})
        elif f == "지역제한":
            rows.append({"document_id": "RFP-000002", "field_name": f,
                         "status": "value_present", "answer_raw": "경기도 소재 업체만",
                         "answer_normalized": "경기도 소재 업체만", "active": "true"})
        else:
            rows.append({"document_id": "RFP-000002", "field_name": f,
                         "status": "field_absent", "answer_raw": "", "answer_normalized": "",
                         "active": "true"})
    # RFP-000003: 예산 external_reference(선별형 경고 케이스)
    for f in fields:
        if f == "예산":
            rows.append({"document_id": "RFP-000003", "field_name": f,
                         "status": "external_reference", "answer_raw": "", "answer_normalized": "",
                         "active": "true"})
        else:
            rows.append({"document_id": "RFP-000003", "field_name": f,
                         "status": "field_absent", "answer_raw": "", "answer_normalized": "",
                         "active": "true"})

    doc = {
        "corpus_version": "v2", "document_count": 3, "extraction_version": "v2",
        "field_count": 12, "registry_version": "v2", "row_count": len(rows),
        "schema_version": "1-12-2/v2", "rows": rows,
    }
    p = tmp_path / "extraction_table_v2.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def deadline_csv_path(tmp_path) -> Path:
    rows = [
        "공고 번호,공고 차수,사업명,사업 금액,발주 기관,공개 일자,입찰 참여 시작일,입찰 참여 마감일,사업 요약,파일형식,파일명,텍스트",
        '1,1,사업A,150000000,서울시청,2024-01-01,2024-01-10,2025-01-01 17:00:00,요약,hwp,문서A.hwp,본문',
        '2,1,사업B,49500000,인천시청,2024-01-01,2024-01-10,2023-01-01 17:00:00,요약,hwp,문서B.hwp,본문',
        '3,1,사업C,0,,2024-01-01,2024-01-10,,요약,hwp,문서C.hwp,본문',
    ]
    p = tmp_path / "data_list.csv"
    p.write_text("\ufeff" + "\n".join(rows), encoding="utf-8")
    return p


@pytest.fixture
def chunks_jsonl_path(tmp_path) -> Path:
    chunks = [
        {"chunk_id": "RFP-000001-0001", "document_id": "RFP-000001", "document_version": "1",
         "processed_sha256": "a" * 40, "sidecar_sha256": "b" * 40, "corpus_version": "v2",
         "source_document_title": "문서A", "section_path": ["1. 개요"],
         "block_type": "text", "table_idx": None, "row_start": None, "row_end": None,
         "part": None, "of": None, "table_degraded": False, "oversize": False,
         "retrieval_eligible": True, "search_text": "문서A > 1. 개요\n추진배경 내용입니다."},
        {"chunk_id": "RFP-000001-0002", "document_id": "RFP-000001", "document_version": "1",
         "processed_sha256": "a" * 40, "sidecar_sha256": "b" * 40, "corpus_version": "v2",
         "source_document_title": "문서A", "section_path": ["2. 표"],
         "block_type": "table", "table_idx": 1, "row_start": 1, "row_end": 3,
         "part": None, "of": None, "table_degraded": True, "oversize": False,
         "retrieval_eligible": True, "search_text": "문서A > 2. 표\n(손상된 표, 빈칸 다수)"},
    ]
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    return p


def _fake_embed_batch(self, texts, batch_size=100, max_retries=3):
    return [np.random.RandomState(abs(hash(t)) % (2**31)).rand(8).tolist() for t in texts]


def _fake_generate(self, question, context_chunks, format_instruction="간결하고 명확하게 답하세요."):
    return f"[모의] 근거 {len(context_chunks)}건"


# ---------------------------------------------------------------------------
# table_query.py — 금액 파서 (회귀 버그 재발 방지)
# ---------------------------------------------------------------------------

class TestTaskTypeRoutingGeneralization:
    """가장 중요한 검증 — 선별형/추출형/QA형/비교형이 실제로 올바른 경로로
    갈라지는지, 8개 연습 문항 표현에 묶이지 않고 다양한 문장으로 확인한다.
    두 층을 본다: (1) router.route()가 올바른 task_type을 고르는가
    (2) answer() 오케스트레이션이 그 task_type에 맞는 경로를 실제로 타서,
    LLM을 태워야 할 때만 태우고 안 태워야 할 때는 절대 안 태우는가."""

    # -- 1층: 규칙 기반 분기(router.py)가 다양한 표현에서 올바른 task_type을
    #    고르는지. 실제 API·데이터 없이 순수 규칙만 테스트.
    _CFG = {"routing_method": "rule_based", "routing_fallback": "qa"}

    @pytest.mark.parametrize("question", [
        "5억 이상인 사업 알려줘",
        "예산 3억 이하인 사업 목록 보여줘",
        "지역제한 없는 사업 추천해줘",
        "1억 넘는 사업 다 보여줘",
        "예산 10억 이하인 사업 리스트",
        "지역제한 있는 사업 전부 보여줘",
    ])
    def test_router_select(self, question):
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
    ])
    def test_router_extract(self, question):
        from router import route
        assert route(question, self._CFG).task_type == "extract"

    @pytest.mark.parametrize("question", [
        "이 사업의 추진배경이 뭐야",
        "왜 이 사업을 하는 거야",
        "사업 목적을 설명해줘",
        "이 프로젝트가 어떤 내용인지 요약해줘",
        "제안요청서 내용 정리해줘",
    ])
    def test_router_qa_fallback(self, question):
        from router import route
        # 명시적 QA 패턴이 없어 규칙 기반 분기가 못 잡으면 폴백으로 qa가 되는 게
        # baseline 설계다(4-10-1) — QA로 온 것 자체가 정상 동작이다.
        r = route(question, self._CFG)
        assert r.task_type == "qa"

    @pytest.mark.parametrize("question", [
        "RFP-000001이랑 RFP-000002 예산 비교해줘",
        "두 사업 지역제한 차이가 뭐야",
        "어느 쪽 사업기간이 더 긴지 비교해줘",
    ])
    def test_router_compare(self, question):
        from router import route
        assert route(question, self._CFG).task_type == "compare"

    # -- 2층: answer() 전체 오케스트레이션 — 실제로 그 경로를 타서 LLM을
    #    태워야 할 때만 태우는지까지 종단 확인.

    @pytest.fixture
    def full_env(self, extraction_table_path, deadline_csv_path, registry_path):
        """select/extract/qa/compare를 전부 시도해볼 수 있는 최소 환경."""
        from vector_store import VectorStore, ChunkMetadata
        from table_query import load_extraction_table
        from deadline_metadata import load_org_index, load_deadline_by_document_id

        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="RFP-000001", document_version="1",
                              processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                              text="추진배경 관련 본문")]
        store.upsert_document("RFP-000001", [("추진배경 관련 본문", [1.0, 0.0])], meta)

        table = load_extraction_table(extraction_table_path)
        org_index = load_org_index(deadline_csv_path, registry_path)
        cfg = {
            "deadline_filter_field": "입찰 참여 마감일", "top_k": 5,
            "routing_method": "rule_based", "routing_fallback": "qa",
            "deadline_filter_default": {"select": False},  # 이 테스트는 마감필터 검증이 목적이 아님
        }
        deadline_map = load_deadline_by_document_id(deadline_csv_path, registry_path, cfg)
        return dict(store=store, table=table, org_index=org_index,
                   deadline_map=deadline_map, cfg=cfg)

    def _forbidden_llm_client(self, kind):
        """select·추출형(단순)·비교형에서 호출되면 바로 실패시키는 가짜 클라이언트."""
        def _raise():
            raise AssertionError(f"이 경로에서는 {kind} 클라이언트가 호출되면 안 됨")
        return _raise

    @pytest.mark.parametrize("question", [
        "1억 이상인 사업 알려줘",
        "지역제한 없는 사업 추천해줘",
    ])
    def test_select_never_calls_llm(self, full_env, question):
        from answer_pipeline import answer, SessionState
        r = answer(
            question, full_env["store"],
            self._forbidden_llm_client("embed"), self._forbidden_llm_client("gen"),
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "select"

    @pytest.mark.parametrize("question", [
        "RFP-000001의 예산이 얼마야",
        "RFP-000001의 지역제한 알려줘",
        "RFP-000001의 마감일이 언제야",
    ])
    def test_simple_extract_never_calls_llm(self, full_env, question):
        from answer_pipeline import answer, SessionState
        r = answer(
            question, full_env["store"],
            self._forbidden_llm_client("embed"), self._forbidden_llm_client("gen"),
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "extract"
        assert r.abstained is False

    def test_compare_never_calls_llm(self, full_env):
        from answer_pipeline import answer, SessionState
        r = answer(
            "RFP-000001이랑 RFP-000002 예산 비교해줘", full_env["store"],
            self._forbidden_llm_client("embed"), self._forbidden_llm_client("gen"),
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "compare"
        assert r.abstained is False

    def test_qa_does_call_llm(self, full_env):
        """QA형은 반대로 — 검색+생성 경로를 실제로 타야 한다(호출 안 되면 실패)."""
        from answer_pipeline import answer, SessionState
        called = {"embed": 0, "gen": 0}

        class FakeEmbed:
            def embed_query(self, q):
                called["embed"] += 1
                return [1.0, 0.0]

        class FakeGen:
            def generate(self, question, context_chunks, format_instruction="..."):
                called["gen"] += 1
                return "모의 답변"

        r = answer(
            "RFP-000001의 추진배경이 뭐야", full_env["store"],
            lambda: FakeEmbed(), lambda: FakeGen(),
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "qa"
        assert called["embed"] >= 1 and called["gen"] >= 1
        # 2026-09-02 하루님 스키마 — QA는 contexts/retrieved/citations가 채워져야 함
        assert len(r.contexts) >= 1
        assert len(r.retrieved) >= 1
        assert len(r.citations) >= 1
        assert all(isinstance(c["section"], str) for c in r.citations)  # dict 통째로 새면 안 됨

    def test_select_fills_structured_answer(self, full_env):
        from answer_pipeline import answer, SessionState
        r = answer(
            "1억 이상인 사업 알려줘", full_env["store"],
            lambda: None, lambda: None,
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "select"
        assert isinstance(r.structured_answer, list)  # 선별형은 목록 형태
        assert r.selected_document_ids == r.structured_answer

    def test_compare_fills_nested_structured_answer(self, full_env):
        from answer_pipeline import answer, SessionState
        r = answer(
            "RFP-000001이랑 RFP-000002 예산 비교해줘", full_env["store"],
            lambda: None, lambda: None,
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "compare"
        assert isinstance(r.structured_answer, dict)  # 비교형은 {문서ID: {필드: 값}}
        assert "RFP-000001" in r.structured_answer
        assert "예산" in r.structured_answer["RFP-000001"]

    def test_extract_with_explanation_does_call_llm(self, full_env):
        """추출형이라도 '설명해줘'가 있으면 G→I→J→K로 넘어가 LLM을 태워야 한다."""
        from answer_pipeline import answer, SessionState
        called = {"gen": 0}

        class FakeEmbed:
            def embed_query(self, q):
                return [1.0, 0.0]

        class FakeGen:
            def generate(self, question, context_chunks, format_instruction="..."):
                called["gen"] += 1
                return "모의 답변"

        r = answer(
            "RFP-000001의 예산이 얼마인지 자세히 설명해줘", full_env["store"],
            lambda: FakeEmbed(), lambda: FakeGen(),
            full_env["table"], full_env["cfg"],
            deadline_map=full_env["deadline_map"], session=SessionState(),
            org_index=full_env["org_index"],
        )
        assert r.task_type == "extract"
        assert called["gen"] >= 1  # 이번엔 호출돼야 정상


class TestKoreanAmountParser:
    def test_plain_comma_number(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount("352,000,000원(부가가치세 포함)") == 352_000_000.0

    def test_mixed_unit_expression(self):
        from table_query import _parse_korean_amount
        assert _parse_korean_amount("1억 5천만 원(부가가치세 포함)") == 150_000_000.0

    def test_thousand_won_unit_not_shortcut(self):
        """리뷰에서 지적된 회귀 버그 — 쉼표 숫자 우선 반환이 단위(천)를 무시하던 문제."""
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
# table_query.py — 로더, 조건 질의, field_absent 경고 처리
# ---------------------------------------------------------------------------

class TestExtractionTableLoader:
    def test_load_and_validate(self, extraction_table_path):
        from table_query import load_extraction_table
        rows = load_extraction_table(extraction_table_path)
        assert len(rows) == 36  # 3문서 x 12필드

    def test_field_absent_excluded_from_clean_match(self, extraction_table_path):
        """리뷰 지적 — field_absent를 '지역제한 없음' 조건 충족으로 잘못 포함하던 버그."""
        from table_query import load_extraction_table, parse_conditions, run_conditions_query
        table = load_extraction_table(extraction_table_path)
        conds, full = parse_conditions("지역제한 없는 사업 알려줘")
        assert full
        results, total, warnings = run_conditions_query(table, conds)
        doc_ids = [r.document_id for r in results]
        assert "RFP-000001" not in doc_ids  # field_absent는 깨끗한 매칭에서 빠져야 함
        assert any("RFP-000001" in w for w in warnings)  # 대신 경고에 문서ID와 함께 있어야 함

    def test_warning_includes_document_id(self, extraction_table_path):
        """리뷰 반영 중 직접 찾은 버그 — 경고 문구에 문서ID 없으면 중복 제거로 뭉개짐."""
        from table_query import load_extraction_table, parse_conditions, run_conditions_query
        table = load_extraction_table(extraction_table_path)
        conds, _ = parse_conditions("지역제한 없는 사업 알려줘")
        _, _, warnings = run_conditions_query(table, conds)
        assert len(warnings) >= 1
        assert all(w.startswith("RFP-") for w in warnings)

    def test_and_compound_condition(self, extraction_table_path):
        from table_query import load_extraction_table, parse_conditions, run_conditions_query
        table = load_extraction_table(extraction_table_path)
        conds, full = parse_conditions("1억 이상이고 지역제한 없는 사업")
        assert full
        assert len(conds) == 2

    def test_unmatched_segment_not_silently_dropped(self, extraction_table_path):
        from table_query import parse_conditions
        conds, full = parse_conditions("1억 이상이고 발주기관이 서울인 사업")
        assert full is False  # 뒷부분(발주기관)을 인식 못 했으면 전부인식 False여야 함


class TestFieldAndDocumentDetection:
    def test_detect_field(self):
        from table_query import detect_field
        assert detect_field("예산이 얼마야") == "예산"
        assert detect_field("지역제한 있어?") == "지역제한"
        assert detect_field("아무 상관없는 말") is None

    def test_detect_document_id(self):
        from table_query import detect_document_id
        assert detect_document_id("RFP-000001의 예산") == "RFP-000001"
        assert detect_document_id("문서 얘기") is None

    def test_detect_deadline_question(self):
        from table_query import detect_deadline_question
        assert detect_deadline_question("마감일이 언제야?")
        assert not detect_deadline_question("예산이 얼마야")

    def test_lookup_field(self, extraction_table_path):
        from table_query import load_extraction_table, lookup_field
        table = load_extraction_table(extraction_table_path)
        row = lookup_field(table, "RFP-000001", "예산")
        assert row is not None
        assert row["status"] == "value_present"


# ---------------------------------------------------------------------------
# deadline_metadata.py — CSV↔registry 매핑, 마감 판정
# ---------------------------------------------------------------------------

class TestDeadlineMetadata:
    def test_org_index_and_deadline_map(self, deadline_csv_path, registry_path):
        from deadline_metadata import load_org_index, load_deadline_by_document_id
        cfg = {"deadline_filter_field": "입찰 참여 마감일"}
        org_index = load_org_index(deadline_csv_path, registry_path)
        assert org_index.get("서울시청") == ["RFP-000001"]

        deadline_map = load_deadline_by_document_id(deadline_csv_path, registry_path, cfg)
        assert deadline_map["RFP-000001"] == datetime(2025, 1, 1, 17, 0, 0)
        assert deadline_map["RFP-000003"] is None  # 미상

    def test_find_documents_by_org_mention(self, deadline_csv_path, registry_path):
        from deadline_metadata import load_org_index, find_documents_by_org_mention
        org_index = load_org_index(deadline_csv_path, registry_path)
        assert find_documents_by_org_mention("서울시청 예산 얼마야", org_index) == ["RFP-000001"]
        assert find_documents_by_org_mention("관계없는 질문", org_index) == []

    def test_is_before_deadline_missing_shown_not_excluded(self, deadline_csv_path, registry_path):
        from deadline_metadata import load_deadline_by_document_id, is_before_deadline
        cfg = {"deadline_filter_field": "입찰 참여 마감일"}
        deadline_map = load_deadline_by_document_id(deadline_csv_path, registry_path, cfg)
        ref = datetime(2024, 6, 1)
        ok, note = is_before_deadline("RFP-000001", deadline_map, ref)
        assert ok is True
        ok, note = is_before_deadline("RFP-000002", deadline_map, ref)
        assert ok is False  # 마감 지남
        ok, note = is_before_deadline("RFP-000003", deadline_map, ref)
        assert ok is None and "미상" in note  # 제외 아니라 미상 표시

    def test_detect_unknown_org(self, deadline_csv_path, registry_path):
        """QA-004류 — 존재하지 않는 기관명은 잡고, 실제 기관은 오탐 없어야 함."""
        from deadline_metadata import load_org_index, detect_unknown_org
        org_index = load_org_index(deadline_csv_path, registry_path)
        assert detect_unknown_org("은하수정보진흥재단에서 발주한 사업 요약해줘", org_index) == "은하수정보진흥재단"
        assert detect_unknown_org("서울시청 사업 요약해줘", org_index) is None  # 실제 존재
        assert detect_unknown_org("이 사업 추진목표가 뭐야", org_index) is None  # 기관명 언급 자체 없음


# ---------------------------------------------------------------------------
# vector_store.py — 정합성 검사
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_upsert_and_search_roundtrip(self):
        from vector_store import VectorStore, ChunkMetadata
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                               processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                               text="본문")]
        store.upsert_document("D1", [("본문", [1.0, 0.0])], meta)
        results = store.search([1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c1"

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
        meta1 = [ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                               processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        store.upsert_document("D1", [("a", [1.0, 2.0])], meta1)
        meta2 = [ChunkMetadata(chunk_id="c2", document_id="D2", document_version="1",
                               processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        with pytest.raises(VectorStoreError):
            store.upsert_document("D2", [("b", [1.0, 2.0, 3.0])], meta2)  # 차원 다름

    def test_top_k_must_be_positive(self):
        from vector_store import VectorStore, ChunkMetadata, VectorStoreError
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                               processed_sha256="x", sidecar_sha256="y", corpus_version="v2")]
        store.upsert_document("D1", [("a", [1.0])], meta)
        with pytest.raises(VectorStoreError):
            store.search([1.0], top_k=0)

    def test_save_load_roundtrip(self, tmp_path):
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

    def test_config_mismatch_detects_corpus_version(self, tmp_path):
        """리뷰 지적 — 예전엔 corpus 버전 등이 안 걸렸음."""
        from vector_store import config_mismatch_check, ConfigMismatchError, IndexTag
        import json as json_mod
        from dataclasses import asdict
        idx_dir = tmp_path / "idx"
        idx_dir.mkdir()
        tag = IndexTag(chunk_size=1500, chunk_overlap=150, embedding_model="m",
                       embedding_provider="openai", preprocess_version="v2",
                       corpus_version="v2", build_timestamp="t")
        (idx_dir / "index_tag.json").write_text(json_mod.dumps(asdict(tag)), encoding="utf-8")
        cfg = {"chunk_size": 1500, "chunk_overlap": 150, "embedding_model": "m",
               "embedding_provider": "openai", "corpus": "v3",  # 버전 바뀜
               "preprocess": "v2", "index": "v1"}
        with pytest.raises(ConfigMismatchError):
            config_mismatch_check(idx_dir, cfg)


# ---------------------------------------------------------------------------
# answer_pipeline.py — 세션/문서 해석, extract·compare 분리, 손상 표 제외
# ---------------------------------------------------------------------------

class TestDocumentResolution:
    def test_explicit_id_priority(self):
        from answer_pipeline import resolve_document_id, SessionState
        r = resolve_document_id("RFP-000001의 예산", session=SessionState())
        assert r.document_id == "RFP-000001" and r.method == "explicit"

    def test_org_single_match(self, deadline_csv_path, registry_path):
        from deadline_metadata import load_org_index
        from answer_pipeline import resolve_document_id, SessionState
        org_index = load_org_index(deadline_csv_path, registry_path)
        r = resolve_document_id("서울시청 예산 얼마야", session=SessionState(), org_index=org_index)
        assert r.document_id == "RFP-000001" and r.method == "org"

    def test_anaphora_uses_session(self):
        from answer_pipeline import resolve_document_id, SessionState
        session = SessionState(active_document_id="RFP-000001")
        r = resolve_document_id("그 사업 예산은?", session=session)
        assert r.document_id == "RFP-000001" and r.method == "anaphora"

    def test_anaphora_without_active_document(self):
        from answer_pipeline import resolve_document_id, SessionState
        r = resolve_document_id("그 사업 예산은?", session=SessionState())
        assert r.document_id is None and r.method == "none"

    def test_resolving_updates_session(self):
        from answer_pipeline import resolve_document_id, SessionState
        session = SessionState()
        resolve_document_id("RFP-000002의 예산", session=session)
        assert session.active_document_id == "RFP-000002"


class TestExtractRouting:
    def test_extract_returns_value_without_llm_call(self, extraction_table_path):
        from table_query import load_extraction_table
        from answer_pipeline import answer_extract_by_table, SessionState
        table = load_extraction_table(extraction_table_path)
        called = {"n": 0}

        def get_gen():
            called["n"] += 1
            raise AssertionError("G-2 경로에서는 생성 클라이언트가 호출되면 안 됨")

        result = answer_extract_by_table(
            "RFP-000001의 예산이 얼마야", table, None, lambda: None, get_gen, {},
            session=SessionState(),
        )
        assert result.abstained is False
        assert "150,000,000" in result.text or "1억 5천만" in result.text
        assert called["n"] == 0

    def test_citation_handles_representative_location_dict(self):
        """리뷰가 아니라 실제 데이터로 발견한 버그 — representative_location이
        문자열이 아니라 {document_id, file, heading, line} 객체였는데 확인
        없이 그대로 section에 넣어서 dict 전체가 들어가던 문제."""
        from answer_pipeline import table_row_to_citation
        row_with_loc = {
            "field_name": "예산",
            "representative_location": {
                "document_id": "RFP-000001", "file": "x.md",
                "heading": "(소요예산) 352,000,000원", "line": 61,
            },
        }
        citation = table_row_to_citation("RFP-000001", row_with_loc)
        assert isinstance(citation["section"], str)  # dict 통째로 들어가면 안 됨
        assert citation["section"] == "(소요예산) 352,000,000원"
        assert citation["ref_no"] == "line 61"

        row_without_loc = {"field_name": "지역제한", "representative_location": None}
        citation2 = table_row_to_citation("RFP-000001", row_without_loc)
        assert citation2["section"] == ""
        assert citation2["ref_no"] == "지역제한"  # 위치 없으면 필드명으로 대체

    def test_field_absent_not_treated_as_no_restriction(self, extraction_table_path):
        from table_query import load_extraction_table
        from answer_pipeline import answer_extract_by_table, SessionState
        table = load_extraction_table(extraction_table_path)
        result = answer_extract_by_table(
            "RFP-000001의 지역제한 알려줘", table, None, lambda: None, lambda: None, {},
            session=SessionState(),
        )
        assert "없음" not in result.text or "항목 자체가 없습니다" in result.text
        assert "제한이 없" not in result.text  # 제한 없다고 단정하면 안 됨

    def test_deadline_question_uses_csv_not_extraction_table(
        self, extraction_table_path, deadline_csv_path, registry_path,
    ):
        from table_query import load_extraction_table
        from deadline_metadata import load_deadline_by_document_id
        from answer_pipeline import answer_extract_by_table, SessionState
        table = load_extraction_table(extraction_table_path)
        cfg = {"deadline_filter_field": "입찰 참여 마감일"}
        deadline_map = load_deadline_by_document_id(deadline_csv_path, registry_path, cfg)
        result = answer_extract_by_table(
            "RFP-000001의 마감일이 언제야", table, None, lambda: None, lambda: None, cfg,
            session=SessionState(), deadline_map=deadline_map,
        )
        assert result.abstained is False
        assert "2025-01-01" in result.text

    def test_missing_document_asks_instead_of_guessing(self, extraction_table_path):
        from table_query import load_extraction_table
        from answer_pipeline import answer_extract_by_table, SessionState
        table = load_extraction_table(extraction_table_path)
        result = answer_extract_by_table(
            "예산이 얼마야", table, None, lambda: None, lambda: None, {},
            session=SessionState(),
        )
        assert result.abstained is True


class TestDegradedTableExclusion:
    def test_degraded_chunk_excluded_from_generation_context(self):
        """리뷰 지적 — table_degraded 청크가 그대로 LLM 컨텍스트에 들어가던 문제."""
        from vector_store import VectorStore, ChunkMetadata
        from config import load_config
        import embedding_client, generation_client
        from embedding_client import EmbeddingClient
        from generation_client import GenerationClient
        from answer_pipeline import answer_qa_or_extract_by_search

        store = VectorStore()
        metas = [
            ChunkMetadata(chunk_id="c1", document_id="D1", document_version="1",
                          processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                          text="정상 본문", table_degraded=False),
            ChunkMetadata(chunk_id="c2", document_id="D1", document_version="1",
                          processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                          text="손상된 표 내용", chunk_type="table", table_degraded=True),
        ]
        store.upsert_document("D1", [("정상 본문", [1.0, 0.0]), ("손상된 표 내용", [0.9, 0.1])], metas)

        captured = {}
        def fake_gen(self, question, context_chunks, format_instruction="..."):
            captured["context"] = context_chunks
            return "답변"

        class FakeEmbed:
            def embed_query(self, q):
                return [1.0, 0.0]

        with patch.object(generation_client.GenerationClient, "generate", fake_gen):
            gc = GenerationClient.__new__(GenerationClient)  # __init__ 우회(API 키 불필요)
            result = answer_qa_or_extract_by_search("질문", store, FakeEmbed(), gc, {"top_k": 5})

        assert not any("손상된 표" in c for c in captured["context"])
        assert "손상" in result.text  # 대신 안내 문구는 있어야 함


# ---------------------------------------------------------------------------
# run_eval.py — 오류 집계, 세션 기본값
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# build_index.py — 필수값 검증(조용한 손실 방지), 잘린 파일 하드 에러
# ---------------------------------------------------------------------------

class TestBuildIndexStrictness:
    def test_map_chunk_requires_version_and_hash(self):
        """리뷰 지적 — 버전·해시 없으면 예전엔 빈 문자열로 조용히 채웠음."""
        from build_index import map_chunk
        good = {"chunk_id": "c1", "document_id": "D1", "search_text": "본문",
                "block_type": "text", "document_version": "1",
                "processed_sha256": "a" * 40, "sidecar_sha256": "b" * 40}
        assert map_chunk(good)["document_version"] == "1"

        missing_version = dict(good)
        missing_version.pop("document_version")
        with pytest.raises(KeyError):
            map_chunk(missing_version)

        empty_hash = dict(good)
        empty_hash["processed_sha256"] = ""
        with pytest.raises(KeyError):
            map_chunk(empty_hash)

    def test_load_chunks_truncated_file_raises(self, tmp_path):
        """리뷰 지적 — 마지막 줄 손상 시 예전엔 스킵하고 조용히 진행했음.
        지금은 잘린 파일이면 무조건 에러로 막는다(조용한 데이터 손실 방지)."""
        from build_index import load_chunks
        p = tmp_path / "chunks.jsonl"
        p.write_text(
            '{"chunk_id": "c1", "document_id": "D1"}\n'
            '{"chunk_id": "c2", "document_id',  # 의도적으로 중간에서 잘림
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_chunks(p)

    def test_load_chunks_complete_file_ok(self, chunks_jsonl_path):
        from build_index import load_chunks
        chunks = load_chunks(chunks_jsonl_path)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# run_eval.py — 세션 기본값(안전한 쪽), 마감필터 하드블록
# ---------------------------------------------------------------------------

class TestRunEvalSessionDefault:
    def _make_env(self, tmp_path, items):
        from vector_store import VectorStore, ChunkMetadata, IndexTag
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="RFP-000001", document_version="1",
                              processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                              text="본문")]
        store.upsert_document("RFP-000001", [("본문", [1.0, 0.0])], meta)
        idx_dir = tmp_path / "idx"
        tag = IndexTag(chunk_size=1500, chunk_overlap=150, embedding_model="m",
                       embedding_provider="openai", preprocess_version="v2",
                       corpus_version="v2", build_timestamp="t")
        store.save(idx_dir, tag)

        evalset = tmp_path / "evalset.jsonl"
        with open(evalset, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        base_yaml = tmp_path / "config" / "base.yaml"
        base_yaml.parent.mkdir(exist_ok=True)
        base_yaml.write_text(
            "top_k: 5\ncorpus: v2\npreprocess: v2\ntable: v2\nindex: v1\nevalset: v1\nscorer: v1\n"
            "embedding_provider: openai\nembedding_model: m\ngeneration_provider: openai\n"
            "generation_model: g\ndata_egress_confirmed: true\nrouting_method: rule_based\n"
            "routing_fallback: qa\ndeadline_filter_default:\n  select: false\n",
            encoding="utf-8",
        )
        return evalset, idx_dir, base_yaml

    def test_default_independent_sessions(self, tmp_path, extraction_table_path):
        """session_id 없고 --continuous-session도 안 주면, 앞 문항의 활성
        문서가 뒷 문항으로 안 새어가야 한다(리뷰 지적 반영 — 새 기본값)."""
        import os, sys
        from unittest.mock import patch
        import embedding_client, generation_client

        items = [
            {"question_id": "q1", "question": "RFP-000001의 예산이 얼마야", "task_type": "extraction"},
            {"question_id": "q2", "question": "그 사업 지역제한 알려줘", "task_type": "extraction"},
        ]
        evalset, idx_dir, base_yaml = self._make_env(tmp_path, items)
        out_dir = tmp_path / "out"

        os.environ["RAG_CONFIG_PATH"] = str(base_yaml)
        os.environ["OPENAI_API_KEY"] = "dummy"
        with patch.object(embedding_client.EmbeddingClient, "embed_batch", _fake_embed_batch), \
             patch.object(generation_client.GenerationClient, "generate", _fake_generate):
            sys.argv = ["run_eval.py", "--evalset", str(evalset), "--index", str(idx_dir),
                       "--extraction-table", str(extraction_table_path), "--out", str(out_dir)]
            import run_eval
            run_eval.main()

        details = [json.loads(l) for l in (out_dir / "details.jsonl").read_text(encoding="utf-8").splitlines()]
        # q2("그 사업")는 세션이 리셋됐으니 문서를 못 찾고 확인 질문으로 abstain해야 함
        assert details[1]["abstained"] is True

    def test_continuous_session_flag_links_questions(self, tmp_path, extraction_table_path):
        """--continuous-session을 명시하면 anaphora가 앞 문항 문서를 이어받아야 한다."""
        import os, sys
        from unittest.mock import patch
        import embedding_client, generation_client

        items = [
            {"question_id": "q1", "question": "RFP-000001의 예산이 얼마야", "task_type": "extraction"},
            {"question_id": "q2", "question": "그 사업 지역제한 알려줘", "task_type": "extraction"},
        ]
        evalset, idx_dir, base_yaml = self._make_env(tmp_path, items)
        out_dir = tmp_path / "out2"

        os.environ["RAG_CONFIG_PATH"] = str(base_yaml)
        os.environ["OPENAI_API_KEY"] = "dummy"
        with patch.object(embedding_client.EmbeddingClient, "embed_batch", _fake_embed_batch), \
             patch.object(generation_client.GenerationClient, "generate", _fake_generate):
            sys.argv = ["run_eval.py", "--evalset", str(evalset), "--index", str(idx_dir),
                       "--extraction-table", str(extraction_table_path), "--out", str(out_dir),
                       "--continuous-session"]
            import importlib, run_eval
            importlib.reload(run_eval)
            run_eval.main()

        details = [json.loads(l) for l in (out_dir / "details.jsonl").read_text(encoding="utf-8").splitlines()]
        assert details[1]["abstained"] is False  # 이번엔 이어받아서 답이 나와야 함
        assert "RFP-000001" in details[1]["answer"]


class TestDeadlineFilterHardBlock:
    def test_missing_files_raises_when_filter_should_be_on(self, tmp_path):
        """리뷰 지적 — 예전엔 경고만 하고 필터 없이 진행했음. 이제 하드블록."""
        import os, sys
        base_yaml = tmp_path / "config" / "base.yaml"
        base_yaml.parent.mkdir()
        base_yaml.write_text(
            "top_k: 5\ncorpus: v2\npreprocess: v2\ntable: v2\nindex: v1\nevalset: v1\nscorer: v1\n"
            "embedding_provider: openai\nembedding_model: m\ngeneration_provider: openai\n"
            "generation_model: g\ndata_egress_confirmed: true\nrouting_method: rule_based\n"
            "routing_fallback: qa\ndeadline_filter_default:\n  select: true\n",  # 필터 켜짐
            encoding="utf-8",
        )
        evalset = tmp_path / "evalset.jsonl"
        evalset.write_text(
            json.dumps({"question_id": "q1", "question": "5억 이상"}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.environ["RAG_CONFIG_PATH"] = str(base_yaml)
        os.environ["OPENAI_API_KEY"] = "dummy"
        sys.argv = ["run_eval.py", "--evalset", str(evalset), "--index", str(tmp_path / "noidx"),
                   "--out", str(tmp_path / "out")]
        import importlib, run_eval
        importlib.reload(run_eval)
        with pytest.raises(RuntimeError, match="deadline_filter_default"):
            run_eval.main()


class TestRunEvalErrorAggregation:
    def test_error_stage_propagates_to_error_field(self, tmp_path, extraction_table_path):
        """리뷰 지적 — answer() 내부 오류가 error_count=0으로 조용히 기록되던 문제."""
        import sys
        from unittest.mock import patch
        import embedding_client, generation_client

        def broken_embed(self, texts, batch_size=100, max_retries=3):
            raise RuntimeError("의도적 오류 주입")

        evalset = tmp_path / "evalset.jsonl"
        evalset.write_text(
            json.dumps({"question_id": "q1", "question": "RFP-000001의 추진배경이 뭐야",
                       "task_type": "qa"}, ensure_ascii=False),
            encoding="utf-8",
        )

        from vector_store import VectorStore, ChunkMetadata, IndexTag
        store = VectorStore()
        meta = [ChunkMetadata(chunk_id="c1", document_id="RFP-000001", document_version="1",
                              processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
                              text="본문")]
        store.upsert_document("RFP-000001", [("본문", [1.0, 0.0])], meta)
        idx_dir = tmp_path / "idx"
        tag = IndexTag(chunk_size=1500, chunk_overlap=150, embedding_model="m",
                       embedding_provider="openai", preprocess_version="v2",
                       corpus_version="v2", build_timestamp="t")
        store.save(idx_dir, tag)

        out_dir = tmp_path / "out"
        base_yaml = tmp_path / "config" / "base.yaml"
        base_yaml.parent.mkdir()
        base_yaml.write_text(
            "top_k: 5\ncorpus: v2\npreprocess: v2\ntable: v2\nindex: v1\nevalset: v1\nscorer: v1\n"
            "embedding_provider: openai\nembedding_model: m\ngeneration_provider: openai\n"
            "generation_model: g\ndata_egress_confirmed: true\nrouting_method: rule_based\n"
            "routing_fallback: qa\n",
            encoding="utf-8",
        )

        import os
        env_backup = os.environ.get("RAG_CONFIG_PATH")
        os.environ["RAG_CONFIG_PATH"] = str(base_yaml)
        os.environ["OPENAI_API_KEY"] = "dummy"
        try:
            with patch.object(embedding_client.EmbeddingClient, "embed_batch", broken_embed), \
                 patch.object(generation_client.GenerationClient, "generate",
                              lambda self, *a, **kw: "안 불릴 것"):
                sys.argv = ["run_eval.py", "--evalset", str(evalset), "--index", str(idx_dir),
                           "--extraction-table", str(extraction_table_path), "--out", str(out_dir)]
                import run_eval
                # 리뷰 반영 — 오류 1건 이상이면 이제 SystemExit(1)로 끝난다
                # (문제13). summary.json 자체는 exit 전에 이미 저장돼 있다.
                with pytest.raises(SystemExit) as exc_info:
                    run_eval.main()
                assert exc_info.value.code == 1

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            assert summary["error_count"] == 1  # 이게 0이면 리뷰가 지적한 버그가 재발한 것
        finally:
            if env_backup:
                os.environ["RAG_CONFIG_PATH"] = env_backup
            else:
                os.environ.pop("RAG_CONFIG_PATH", None)
