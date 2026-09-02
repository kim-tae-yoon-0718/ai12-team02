"""pytest 공용 설정 — src/rag·src/scripts를 sys.path에 넣고, 테스트가 공식
데이터(/srv/rfp)에 의존하지 않도록 픽스처를 모아둔다.

⚠️ 이 테스트는 OpenAI API를 호출하지 않는다(embed/generate를 모의 처리).
"""
import json
import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "rag"))
sys.path.insert(0, str(_SRC / "scripts"))

OFFICIAL_FIELDS = [
    "사업 개요", "사업분야", "공고일", "사업기간", "예산",
    "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건",
    "평가 배점", "제출 방식", "필수 제출 서류", "과업 범위",
]

_H1 = "a" * 64
_H2 = "b" * 64
_H3 = "c" * 64
_S1 = "d" * 64
_S2 = "e" * 64
_S3 = "f" * 64


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """RAG_ROOT/RAG_CONFIG_PATH가 실제 서버 경로를 가리켜 테스트 결과가
    환경에 따라 달라지지 않게 한다(회귀: 공식 base.yaml을 읽어 실패하던 문제)."""
    monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)
    monkeypatch.setenv("RAG_ROOT", str(tmp_path / "_no_such_root"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    yield


def _loc(heading, line, block_type="paragraph", block_index=1, section_titles=None):
    return {
        "block_index": block_index,
        "block_type": block_type,
        "document_id": "X",
        "file": "x.md",
        "heading": heading,
        "line": line,
        "line_end": line,
        "line_start": line,
        "section_path": [{"level": 1, "line": line, "title": t}
                         for t in (section_titles or [heading])],
        "source_type": "body_sentence",
    }


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
             "document_version": 1, "processed_sha256": _H1, "sidecar_sha256": _S1,
             "retrieval_eligible": True, "relation_status": "independent",
             "duplicate_of_document_id": None},
            {"document_id": "RFP-000002", "output_filename": "문서B.md",
             "document_version": 1, "processed_sha256": _H2, "sidecar_sha256": _S2,
             "retrieval_eligible": True, "relation_status": "independent",
             "duplicate_of_document_id": None},
            {"document_id": "RFP-000003", "output_filename": "문서C.md",
             "document_version": 1, "processed_sha256": _H3, "sidecar_sha256": _S3,
             "retrieval_eligible": True, "relation_status": "independent",
             "duplicate_of_document_id": None},
            {"document_id": "RFP-000099", "output_filename": "중복문서.md",
             "document_version": 1, "processed_sha256": _H1, "sidecar_sha256": _S1,
             "retrieval_eligible": False, "relation_status": "duplicate_confirmed",
             "duplicate_of_document_id": "RFP-000001"},
        ],
    }
    p = tmp_path / "document_registry_v2.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def identity_csv_path(tmp_path) -> Path:
    """공식 identity_v2와 같은 컬럼 구성(BOM 포함)."""
    header = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')
    rows = [
        ('"RFP-000001","문서A.md","","","","2024001","0.0","(사)벤처기업협회",'
         '"2024년 벤처확인종합관리시스템 기능 고도화 용역사업","2024-01-01 10:00:00",'
         '"2025-01-01 17:00:00","true"'),
        ('"RFP-000002","문서B.md","","","","2024002","0.0","㈜테스트정보",'
         '"테스트정보 통합관리시스템 구축","2024-01-01 10:00:00",'
         '"2023-01-01 17:00:00","true"'),
        ('"RFP-000003","문서C.md","","","","2024003","0.0","재단법인 한국테스트재단",'
         '"한국테스트재단 데이터 플랫폼 고도화","2024-01-01 10:00:00","","true"'),
        ('"RFP-000099","중복문서.md","","","","2024004","0.0","서울특별시교육청",'
         '"서울특별시교육청 지능정보화전략계획 수립","2024-01-01 10:00:00",'
         '"2025-02-01 17:00:00","true"'),
    ]
    p = tmp_path / "document_identity_v2.csv"
    p.write_text("﻿" + "\n".join([header] + rows) + "\n", encoding="utf-8")
    return p


def _rows_for(doc_id, overrides):
    out = []
    for f in OFFICIAL_FIELDS:
        base = {
            "document_id": doc_id, "field_name": f, "status": "field_absent",
            "answer_raw": "", "answer_normalized": "", "active": "true",
            "representative_location": None, "additional_locations": [],
            "schema_version": "1-12-2/v3", "extraction_version": "v3",
            "corpus_version": "v2", "registry_version": "v2",
        }
        base.update(overrides.get(f, {}))
        out.append(base)
    return out


@pytest.fixture
def extraction_table_path(tmp_path) -> Path:
    rows = []
    rows += _rows_for("RFP-000001", {
        "예산": {"status": "value_present", "answer_raw": "1억 5천만 원(부가가치세 포함)",
                "answer_normalized": "1억 5천만 원(부가가치세 포함)",
                "representative_location": _loc("2. 사업개요", 61, "paragraph", 6),
                "additional_locations": [_loc("Ⅰ. 추진개요", 27, "heading", 0)]},
        "사업기간": {"status": "value_present", "answer_raw": "계약일로부터 150일",
                  "answer_normalized": "계약일로부터 150일",
                  "representative_location": _loc("2. 사업개요", 55, "paragraph", 3)},
    })
    rows += _rows_for("RFP-000002", {
        "예산": {"status": "value_present", "answer_raw": "49,500천원(부가세 포함)",
                "answer_normalized": "49,500천원(부가세 포함)",
                "representative_location": _loc("3. 예산", 120, "table", 2)},
        "지역제한": {"status": "value_present",
                  "answer_raw": "입찰공고일 전일부터 본점 소재지가 경기도인 업체",
                  "answer_normalized": "입찰공고일 전일부터 본점 소재지가 경기도인 업체",
                  "representative_location": _loc("2. 입찰 참가자격", 88, "paragraph", 4)},
        "사업기간": {"status": "value_present", "answer_raw": "계약체결일로부터 4개월 이내",
                  "answer_normalized": "계약체결일로부터 4개월 이내",
                  "representative_location": _loc("Ⅰ. 사업 개요", 81, "paragraph", 2)},
        "컨소시엄 요건": {
            "status": "conflict",
            "answer_raw": "공동수급을 불허하고 단독 형태로만 입찰에 참가할 수 있음\n---\n공동수급(컨소시엄) 및 하도급 : 허용",
            "answer_normalized": "공동수급을 불허하고 단독 형태로만 입찰에 참가할 수 있음 --- 공동수급(컨소시엄) 및 하도급 : 허용",
            "representative_location": _loc("6) 제약사항", 385, "table", 8),
            "additional_locations": [_loc("다. 입찰 참가자격", 669, "paragraph", 15)]},
    })
    rows += _rows_for("RFP-000003", {
        "예산": {"status": "external_reference", "answer_raw": "예산: 입찰공고문 참조",
                "answer_normalized": "예산: 입찰공고문 참조",
                "representative_location": _loc("라. 제안 안내", 1089, "paragraph", 1)},
        "평가 배점": {"status": "not_disclosed", "answer_raw": "비공개",
                   "answer_normalized": "비공개",
                   "representative_location": _loc("4. 평가", 300, "paragraph", 2)},
    })
    doc = {
        "corpus_version": "v2", "document_count": 3, "extraction_version": "v3",
        "field_count": 12, "fields": OFFICIAL_FIELDS, "registry_version": "v2",
        "row_count": len(rows), "schema_version": "1-12-2/v3", "rows": rows,
    }
    p = tmp_path / "extraction_table_v3.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def extraction_metadata_path(tmp_path, extraction_table_path) -> Path:
    doc = json.loads(extraction_table_path.read_text(encoding="utf-8"))
    meta = {
        "row_count": doc["row_count"], "document_count": doc["document_count"],
        "field_count": doc["field_count"], "schema_version": doc["schema_version"],
        "extraction_version": doc["extraction_version"],
        "corpus_version": doc["corpus_version"],
        "registry_version": doc["registry_version"],
    }
    p = tmp_path / "extraction_metadata.json"
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def chunks_jsonl_path(tmp_path) -> Path:
    def _c(cid, doc_id, h, s, **kw):
        base = {
            "chunk_id": cid, "document_id": doc_id, "document_version": "1",
            "processed_sha256": h, "sidecar_sha256": s, "corpus_version": "v2",
            "preprocess_version": "v2", "chunking_version": "v3",
            "source_document_title": "문서", "section_path": ["1. 개요"],
            "section_paths": [["1. 개요"]], "location_label": "1. 개요 · 문단 1",
            "block_type": "text", "table_idx": None, "row_start": None,
            "row_end": None, "part": None, "of": None, "table_degraded": False,
            "oversize": False, "retrieval_eligible": True,
            "md_line_start": 10, "md_line_end": 12,
            "search_text": "문서 > 1. 개요\n추진배경 내용입니다.",
        }
        base.update(kw)
        return base

    chunks = [
        _c("RFP-000001-0001", "RFP-000001", _H1, _S1),
        _c("RFP-000001-0002", "RFP-000001", _H1, _S1, block_type="table",
           table_idx=1, table_degraded=True, search_text="문서 > 2. 표\n(손상된 표)"),
        _c("RFP-000002-0001", "RFP-000002", _H2, _S2),
        _c("RFP-000003-0001", "RFP-000003", _H3, _S3),
        # 등록부의 검색 제외(중복) 문서도 청킹은 돼 있다 — 공식 데이터와 같은 상태.
        # 검색 대상에서 빼는 건 build_index의 retrieval_eligible 필터가 한다.
        _c("RFP-000099-0001", "RFP-000099", _H1, _S1),
    ]
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
                 encoding="utf-8")
    return p


@pytest.fixture
def chunks_version_txt(tmp_path, chunks_jsonl_path) -> Path:
    p = chunks_jsonl_path.parent / "VERSION.txt"
    p.write_text(
        "# RFP 검색용 청크\n"
        "chunking version   : v3\n"
        "corpus version     : v2\n"
        "preprocess version : v2\n"
        "table version      : v3\n"
        "문서 수       : 4\n"
        "청크 수       : 5\n",
        encoding="utf-8",
    )
    return p


BASE_YAML_TEXT = """
top_k: 5
corpus: v2
preprocess: v2
table: v3
index: v1
evalset: v1
scorer: null
document_registry_version: v2
chunking_version: v3
chunk_size: 1500
chunk_overlap: 150
extraction_version: v3
schema_version: "1-12-2/v3"
embedding_provider: openai
embedding_model: text-embedding-3-small
generation_provider: openai
generation_model: gpt-5-mini
data_egress_confirmed: true
routing_method: rule_based
routing_fallback: qa
temperature: null
top_p: null
logprobs: null
max_completion_tokens: 4096
pricing_unit: per_1m_tokens
pricing:
  gpt-5-mini:
    input_per_1m: 0.25
    cached_input_per_1m: 0.025
    output_per_1m: 2.0
  text-embedding-3-small:
    input_per_1m: 0.02
reference_datetime_source: external
reference_datetime: "2024-06-01"
deadline_filter_field: bid_deadline
deadline_missing_policy: show_as_unknown
deadline_filter_disclosure: false
deadline_filter_default:
  select: false
  extract: false
  qa: false
  compare: false
"""


@pytest.fixture
def base_cfg() -> dict:
    import yaml
    return yaml.safe_load(BASE_YAML_TEXT)


@pytest.fixture
def base_yaml_path(tmp_path) -> Path:
    p = tmp_path / "config" / "base.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(BASE_YAML_TEXT, encoding="utf-8")
    return p


def make_index_tag(**over):
    """base_cfg 와 정합한 **완전한** 인덱스 꼬리표(결함 1-1 검증 통과용)."""
    from vector_store import IndexTag
    kw = dict(chunk_size=1500, chunk_overlap=150,
              embedding_model="text-embedding-3-small", embedding_provider="openai",
              preprocess_version="v2", corpus_version="v2", build_timestamp="t",
              registry_version="v2", extraction_version="v3", chunking_version="v3",
              vector_dimension=2, document_count=2, chunk_count=2)
    kw.update(over)
    return IndexTag(**kw)


def write_valid_index(tmp_path, store, name="idx", **tag_over):
    """검증을 통과하는 인덱스를 저장하고 경로를 돌려준다."""
    idx = tmp_path / name
    kw = {"vector_dimension": store.dimension or 2}
    kw.update(tag_over)          # 호출자가 준 값이 우선(불일치 케이스 테스트용)
    store.save(idx, make_index_tag(**kw))
    return idx


@pytest.fixture
def identity_index(identity_csv_path):
    from identity_metadata import load_identity
    return load_identity(identity_csv_path)


@pytest.fixture
def table_rows(extraction_table_path):
    from table_query import load_extraction_table
    return load_extraction_table(extraction_table_path)


@pytest.fixture
def small_store():
    from vector_store import VectorStore, ChunkMetadata
    store = VectorStore()
    for doc_id, vec, text in (
        ("RFP-000001", [1.0, 0.0], "추진배경 관련 본문"),
        ("RFP-000002", [0.0, 1.0], "다른 문서 본문"),
    ):
        meta = [ChunkMetadata(
            chunk_id=f"{doc_id}-0003", document_id=doc_id, document_version="1",
            processed_sha256="x", sidecar_sha256="y", corpus_version="v2",
            text=text, document_name="문서", chapter="1. 개요",
            section_path=[{"level": 1, "line": 10, "title": "1. 개요"}],
            section_paths=[[{"level": 1, "line": 10, "title": "1. 개요"}]],
            location_label="1. 개요 · 문단 1", md_line_start=10, md_line_end=12,
        )]
        store.upsert_document(doc_id, [(text, vec)], meta)
    return store


class FakeEmbed:
    def __init__(self):
        self.calls = 0
        from pricing import Usage
        self.usage = Usage()

    def reset_usage(self):
        from pricing import Usage
        self.usage = Usage()

    def embed_query(self, q):
        self.calls += 1
        self.usage.add_embedding(11)
        return [1.0, 0.0]


class FakeGen:
    """실제 GenerationClient 와 같은 계약 — 답에 실제로 쓴 근거 번호를 선언한다.

    use_evidence: None=규약 미준수 / () = 원문 근거 미사용 / (1,..) = 그 번호만 사용."""

    def __init__(self, text="모의 생성 답변", use_evidence=(1,), clamp=True):
        self.calls = 0
        self.text = text
        self.last_structured_context = None
        self.last_context_chunks = None
        self._use_evidence = use_evidence
        self._clamp = clamp
        self.last_used_evidence = None
        from pricing import Usage
        self.usage = Usage()

    def reset_usage(self):
        from pricing import Usage
        self.usage = Usage()

    def generate(self, question, context_chunks, format_instruction="...",
                 structured_context=None):
        self.calls += 1
        self.last_structured_context = structured_context
        self.last_context_chunks = list(context_chunks)
        self.usage.add_generation(100, 20, cached_tokens=0)
        if self._use_evidence is None:
            self.last_used_evidence = None
        elif self._clamp:
            self.last_used_evidence = [e for e in self._use_evidence
                                       if e <= len(context_chunks)]
        else:
            self.last_used_evidence = list(self._use_evidence)
        return self.text


@pytest.fixture
def fake_clients():
    e, g = FakeEmbed(), FakeGen()
    return e, g, (lambda: e), (lambda: g)


def forbidden(kind):
    def _raise():
        raise AssertionError(f"이 경로에서는 {kind} 클라이언트가 호출되면 안 됨")
    return _raise
