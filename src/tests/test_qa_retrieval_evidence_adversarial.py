#@title QA 검색·생성·출처 경계의 다각도 회귀 검사
#@markdown 합성 문서와 고정 벡터, API 응답 대역으로 실제 답변 경로를 검사합니다. 유료 API는 호출하지 않습니다.
"""검색 코드·응답 계약 검사이며 실제 LLM의 정답률을 측정하지 않는다."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import FakeEmbed, FakeGen
from generation_client import GenerationClient, split_used_evidence
from pricing import Usage
from vector_store import ChunkMetadata, VectorStore


def _meta(doc, number, *, active=True, degraded=False, text=None, kind="text"):
    """실제 청크와 같은 위치·상태 필드를 가진 작은 근거를 만든다."""
    return ChunkMetadata(
        chunk_id=f"{doc}-{number:04d}", document_id=doc, document_version="1",
        processed_sha256="a" * 64, sidecar_sha256="b" * 64, corpus_version="v2",
        text=f"{doc} 근거 {number}: 추진 배경 설명" if text is None else text,
        active=active, table_degraded=degraded, chunk_type=kind,
        document_name=f"{doc} 문서", chapter="Ⅰ. 사업안내 > 2. 추진배경",
        section_path=["Ⅰ. 사업안내", "2. 추진배경"],
        section_paths=[["Ⅰ. 사업안내", "2. 추진배경"]],
        location_label=f"2. 추진배경 · {'표' if kind == 'table' else '문단'} {number}",
        table_idx=number if kind == "table" else None,
        md_line_start=number * 10, md_line_end=number * 10 + 4,
    )


@pytest.fixture
def mixed_store():
    """다른 문서·비활성 근거·손상 표가 정상 근거 사이에 있는 검색 자료."""
    store = VectorStore()
    store.metadata = [
        _meta("RFP-000001", 1),
        _meta("RFP-000001", 2, active=False),
        _meta("RFP-000001", 3, degraded=True, kind="table"),
        _meta("RFP-000001", 4),
        _meta("RFP-000002", 1),
    ]
    store.vectors = np.asarray(
        [[.99, .10], [1, 0], [.98, .20], [.97, .30], [1, 0]], dtype=np.float32)
    return store


def _sdk_client(cfg, raw="설명입니다.\nUSED_EVIDENCE: E1", *, finish="stop", refusal=None):
    """SDK 호출 경계만 대체하고 실제 GenerationClient.generate를 실행한다."""
    sent = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish,
                                 message=SimpleNamespace(content=raw, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=90, completion_tokens=12,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=20)),
    )

    def create(**kwargs):
        sent.append(kwargs)
        return response

    client = GenerationClient.__new__(GenerationClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.cfg = dict(cfg)
    client.model = cfg["generation_model"]
    client._system_prompt_template = "근거로 답변한다. {format_instruction}"
    client.usage = Usage()
    client.last_request_kwargs = None
    client.last_used_evidence = [91]
    return client, sent


def _answer(cfg, store, gen, *, question="RFP-000001 추진 배경을 설명해줘", identity=None):
    """라우팅부터 출처 기록까지 실제 answer()를 실행한다."""
    from answer_pipeline import answer
    embed = FakeEmbed()
    result = answer(question, store, lambda: embed, lambda: gen, [], cfg, identity=identity)
    return result, embed


@pytest.mark.parametrize("active_only", [True, False])
@pytest.mark.parametrize("doc", [None, "RFP-000001", "RFP-000002", "RFP-000999"])
@pytest.mark.parametrize("top_k", [1, 3, 99])
def test_search_filters_before_top_k(mixed_store, active_only, doc, top_k):
    """전체 순위의 앞부분에 다른 문서가 있어도 요청 문서의 top-k를 찾는다."""
    found = mixed_store.search([1, 0], top_k=top_k, active_only=active_only, document_id=doc)
    eligible = [(i, m) for i, m in enumerate(mixed_store.metadata)
                if (not active_only or m.active) and (doc is None or m.document_id == doc)]
    expected_scores = {m.chunk_id: float(mixed_store.vectors[i, 0]
                       / np.linalg.norm(mixed_store.vectors[i])) for i, m in eligible}
    assert len(found) == min(top_k, len(eligible))
    assert all(m.chunk_id in expected_scores for m, _ in found)
    assert [score for _, score in found] == pytest.approx(
        sorted(expected_scores.values(), reverse=True)[:top_k])


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5, "5", None])
def test_search_rejects_invalid_top_k(mixed_store, bad_k):
    """잘못된 개수 설정을 임의의 검색 결과로 처리하지 않는다."""
    from vector_store import VectorStoreError
    with pytest.raises(VectorStoreError):
        mixed_store.search([1, 0], top_k=bad_k)


@pytest.mark.parametrize("used", [(), (1,), (2,), (1, 2), (0, 2, 99)])
def test_actual_qa_context_numbering_after_filters(mixed_store, base_cfg, used):
    """손상 표를 제외한 실제 입력 두 개의 E번호와 citations가 대응한다."""
    cfg = dict(base_cfg, top_k=10)
    gen = FakeGen(use_evidence=used, clamp=False)
    result, embed = _answer(cfg, mixed_store, gen)
    assert result.error_stage is None
    assert embed.calls == gen.calls == 1
    assert [c["chunk_id"] for c in result.contexts] == ["RFP-000001-0001", "RFP-000001-0004"]
    assert [c["chunk_id"] for c in result.retrieved] == [
        "RFP-000001-0001", "RFP-000001-0003", "RFP-000001-0004"]
    assert len(gen.last_context_chunks) == 2
    assert "근거 1" in gen.last_context_chunks[0] and "근거 4" in gen.last_context_chunks[1]
    assert all("RFP-000002" not in text and "근거 3" not in text for text in gen.last_context_chunks)
    assert {c["line"] for c in result.citations} == {10 if i == 1 else 40 for i in used if i in (1, 2)}
    assert result.citation_diagnostics["unknown_evidence_ids"] == [i for i in used if i not in (1, 2)]


@pytest.mark.parametrize("kind", ["text", "table"])
def test_qa_citation_preserves_published_coordinates(base_cfg, kind):
    """사용한 근거의 ref_no·줄 범위가 청크 공개 규약과 같아야 한다."""
    store = VectorStore()
    meta = _meta("RFP-000001", 7, kind=kind)
    store.upsert_document(meta.document_id, [(meta.text, [1, 0])], [meta])
    result, _ = _answer(base_cfg, store, FakeGen())
    assert result.error_stage is None
    assert result.citations[0]["ref_no"] == meta.location_label
    assert result.citations[0]["line"] == meta.md_line_start
    assert result.citations[0]["line_end"] == meta.md_line_end
    assert result.citations[0]["section"] == "2. 추진배경"


@pytest.mark.parametrize("raw", ["", " ", "\n\t"])
def test_empty_chunk_is_not_generation_evidence(base_cfg, raw):
    """위치 이름만 있고 본문이 없으면 실제 근거로 간주하지 않는다."""
    store = VectorStore()
    meta = _meta("RFP-000001", 1, text=raw)
    store.upsert_document(meta.document_id, [(raw, [1, 0])], [meta])
    gen = FakeGen("근거 없이 만든 답")
    result, _ = _answer(base_cfg, store, gen)
    assert result.error_stage is None
    assert gen.calls == 0
    assert result.abstained is True
    assert result.contexts == [] and result.citations == []


@pytest.mark.parametrize("mode", ["empty", "inactive", "degraded", "other_document"])
def test_no_usable_evidence_does_not_call_generator(base_cfg, mode):
    store = VectorStore()
    if mode != "empty":
        doc = "RFP-000002" if mode == "other_document" else "RFP-000001"
        meta = _meta(doc, 1, active=mode != "inactive", degraded=mode == "degraded")
        store.upsert_document(doc, [(meta.text, [1, 0])], [meta])
    gen = FakeGen()
    result, _ = _answer(base_cfg, store, gen)
    assert gen.calls == 0
    assert result.abstained is True
    assert result.contexts == [] and result.citations == []


@pytest.mark.parametrize("raw, ids, body", [
    ("본문\nUSED_EVIDENCE: E1, 2", [1, 2], "본문"),
    ("본문\nUSED_EVIDENCE: E1, E2", [1, 2], "본문"),
    ("본문\nUSED_EVIDENCE: E1 / 2 / E3", [1, 2, 3], "본문"),
    ("본문\nused evidence: e1、2·e3", [1, 2, 3], "본문"),
    ("본문\nUSED_EVIDENCE: E1000", [1000], "본문"),
    ("본문\nUSED_EVIDENCE: E0, E1000", [0, 1000], "본문"),
    ("본문\nUSED_EVIDENCE: E1, E1, 1", [1], "본문"),
    ("앞 USED_EVIDENCE: E1 뒤 USED_EVIDENCE: E2", [1, 2], "앞 뒤"),
    ("본문\nUSED_EVIDENCE: NONE", [], "본문"),
    ("본문\nUSED_EVIDENCE: 없음", [], "본문"),
    ("본문만", None, "본문만"),
])
def test_evidence_marker_parse_is_lossless(raw, ids, body):
    assert split_used_evidence(raw) == (body, ids)


@pytest.mark.parametrize("raw,finish,refusal", [
    ("일부만 생성된 설명.\nUSED_EVIDENCE: E1", "length", None),
    ("중단된 설명.\nUSED_EVIDENCE: E1", "content_filter", None),
    ("도구 호출 중간 설명", "tool_calls", None),
    ("함수 호출 중간 설명", "function_call", None),
    ("종료가 확인되지 않은 설명", None, None),
    ("", "stop", None),
    (None, "stop", None),
    ("   \n", "stop", None),
    ("USED_EVIDENCE: E1", "stop", None),
    ("표시할 수 없는 설명", "stop", "요청을 거부했습니다"),
])
def test_invalid_generation_cannot_appear_as_success(mixed_store, base_cfg, raw, finish, refusal):
    """빈 출력·토큰 한도 중단·API 거부를 정상 답변으로 보고하지 않는다."""
    gen, sent = _sdk_client(base_cfg, raw, finish=finish, refusal=refusal)
    result, _ = _answer(base_cfg, mixed_store, gen)
    assert len(sent) == 1
    assert result.abstained is True and result.error_stage is not None
    assert result.failure
    assert result.citations == []
    assert gen.last_used_evidence is None
    assert gen.usage.generation_requests == 1
    assert gen.usage.generation_input_tokens == 90
    assert gen.usage.generation_output_tokens == 12


@pytest.mark.parametrize("raw", ["확인할 수 없습니다.", "확인할 수 없습니다"])
def test_plain_model_abstention_is_recorded(mixed_store, base_cfg, raw):
    """모델이 명시적으로 답을 못 했다고 말하면 기권 칸도 그 사실을 기록한다."""
    gen, _ = _sdk_client(base_cfg, raw + "\nUSED_EVIDENCE: NONE")
    result, _ = _answer(base_cfg, mixed_store, gen)
    assert result.error_stage is None
    assert result.abstained is True


def test_partial_uncertainty_does_not_erase_answer(mixed_store, base_cfg):
    """정상 설명 안의 같은 표현을 발견했다고 전체 답을 기권으로 바꾸지 않는다."""
    text = "세부 일정은 확인할 수 없습니다. 다만 추진 배경은 시스템 노후화입니다."
    gen, _ = _sdk_client(base_cfg, text + "\nUSED_EVIDENCE: E1")
    result, _ = _answer(base_cfg, mixed_store, gen)
    assert result.error_stage is None and result.abstained is False
    assert result.text == text


@pytest.mark.parametrize("raw", ["", " \n"])
def test_pipeline_guards_empty_custom_generator(mixed_store, base_cfg, raw):
    """시험 대역이나 다른 클라이언트가 빈 값을 내도 응답 계약을 지킨다."""
    result, _ = _answer(base_cfg, mixed_store, FakeGen(raw))
    assert result.abstained is True and result.failure
    assert result.citations == []


def test_real_generation_boundary_preserves_context_and_billing(mixed_store, base_cfg):
    gen, sent = _sdk_client(base_cfg, "추진 배경입니다.\nUSED_EVIDENCE: E1, 2")
    result, _ = _answer(dict(base_cfg, top_k=10), mixed_store, gen)
    assert result.error_stage is None and not result.abstained
    assert len(sent) == 1
    user_message = sent[0]["messages"][1]["content"]
    assert "[E1]" in user_message and "[E2]" in user_message and "[E3]" not in user_message
    assert "근거 1" in user_message and "근거 4" in user_message
    assert "근거 2" not in user_message and "근거 3" not in user_message
    assert len(result.contexts) == len(result.citations) == 2
    assert result.text == "추진 배경입니다."
    assert gen.usage.generation_cached_input_tokens == 20
    assert gen.last_used_evidence == [1, 2]


def test_failed_request_clears_previous_evidence(mixed_store, base_cfg):
    gen, _ = _sdk_client(base_cfg)

    def fail(**kwargs):
        raise RuntimeError("합성 연결 실패")

    gen.client.chat.completions.create = fail
    result, _ = _answer(base_cfg, mixed_store, gen)
    assert result.abstained and result.failure
    assert gen.last_used_evidence is None
    assert gen.usage.generation_requests == 0


def test_empty_choice_response_is_failed_and_counted(mixed_store, base_cfg):
    """답 후보가 없는 SDK 응답도 실패로 기록하고 이미 사용한 토큰은 보존한다."""
    gen, _ = _sdk_client(base_cfg)
    gen.client.chat.completions.create = lambda **kw: SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5))
    result, _ = _answer(base_cfg, mixed_store, gen)
    assert result.abstained and result.failure
    assert gen.last_used_evidence is None
    assert gen.usage.generation_requests == 1
    assert gen.usage.generation_output_tokens == 5


def test_legacy_sdk_double_without_finish_reason_still_works(base_cfg):
    """예전 시험 대역에만 없는 종료 필드 때문에 기존 검사가 깨지지 않는다."""
    gen, _ = _sdk_client(base_cfg)
    gen.client.chat.completions.create = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="답입니다.\nUSED_EVIDENCE: E1"))],
        usage=None)
    assert gen.generate("질문", ["근거"]) == "답입니다."
    assert gen.last_used_evidence == [1]


@pytest.mark.parametrize("doc,line,expected", [
    ("RFP-000001", 9, None), ("RFP-000001", 10, "wide"),
    ("RFP-000001", 19, "wide"), ("RFP-000001", 20, "narrow"),
    ("RFP-000001", 24, "narrow"), ("RFP-000001", 25, "wide"),
    ("RFP-000001", 30, "wide"), ("RFP-000001", 31, None),
    ("RFP-000002", 20, "other"), ("RFP-000999", 20, None),
    ("RFP-000001", None, None),
])
def test_locator_range_boundaries_and_document_isolation(doc, line, expected):
    from chunk_locator import ChunkLocator
    records = [
        {"document_id": "RFP-000001", "chunk_id": "wide", "md_line_start": 10,
         "md_line_end": 30, "section_path": ["큰절"], "location_label": "큰절 · 문단 1-8"},
        {"document_id": "RFP-000001", "chunk_id": "narrow", "md_line_start": 20,
         "md_line_end": 24, "section_path": ["큰절", "작은절"], "location_label": "작은절 · 문단 2"},
        {"document_id": "RFP-000002", "chunk_id": "other", "md_line_start": 20,
         "md_line_end": 24, "section_path": ["다른절"], "location_label": "다른절 · 문단 2"},
    ]
    found = ChunkLocator.from_records(records).locate(doc, line)
    assert (found.chunk_id if found else None) == expected
