from __future__ import annotations

from answer_pipeline import Answer, SessionState, _stage3_fast_path
from doc_resolver import DocumentResolution, RESOLVE_EXPLICIT
from router import RouteResult


def _callbacks_that_must_not_run():
    def blocked():
        raise AssertionError("accepted fast path must not request an API client")
    return blocked, blocked


def _call(question, monkeypatch, *, routed, result=None, resolution=None):
    monkeypatch.setattr("answer_pipeline.route", lambda _q, _cfg: routed)
    if resolution is not None:
        monkeypatch.setattr(
            "answer_pipeline.resolve_document", lambda *_a, **_k: resolution)
    if result is not None and routed.task_type == "select":
        monkeypatch.setattr(
            "answer_pipeline.answer_select_by_table", lambda *_a, **_k: result)
    if result is not None and routed.task_type == "extract":
        monkeypatch.setattr(
            "answer_pipeline.answer_extract_by_table", lambda *_a, **_k: result)
    get_embed, get_gen = _callbacks_that_must_not_run()
    return _stage3_fast_path(
        question, object(), get_embed, get_gen, [], {},
        session=SessionState(),
    )


def test_complete_selection_contract_uses_api_free_fast_path(monkeypatch):
    expected = Answer(
        text="결과", task_type="select", route="추출테이블_문서선별",
        structured_answer=["RFP-000001"], selected_document_ids=["RFP-000001"],
    )
    answer, diagnostic = _call(
        "예산이 명시된 공고를 모두 찾아줘.", monkeypatch,
        routed=RouteResult("select", "selection:예산/status_value_present"),
        result=expected,
    )
    assert answer is expected
    assert diagnostic["accepted"] is True
    assert diagnostic["reason"] == "complete_selection_contract"


def test_selection_rejects_when_detected_field_was_not_parsed(monkeypatch):
    called = False

    def should_not_run(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(
        "answer_pipeline.route",
        lambda _q, _cfg: RouteResult("select", "selection:partial"),
    )
    monkeypatch.setattr("answer_pipeline.answer_select_by_table", should_not_run)
    get_embed, get_gen = _callbacks_that_must_not_run()
    answer, diagnostic = _stage3_fast_path(
        "제출 방식 항목과 필수 제출 서류 항목이 둘 다 공고문에 적혀 있지 않은 공고를 뽑아줘.",
        object(), get_embed, get_gen, [], {}, session=SessionState(),
    )
    assert answer is None
    assert diagnostic["reason"] == "detected_and_parsed_fields_differ"
    assert called is False


def test_confirmed_scalar_extract_uses_api_free_fast_path(monkeypatch):
    expected = Answer(
        text="1억원", task_type="extract", route="추출테이블_값조회",
        structured_answer={"예산": {"status": "value_present"}},
        condition_query=[{"field": "예산", "status": "value_present"}],
    )
    answer, diagnostic = _call(
        "RFP-000001의 예산 알려줘", monkeypatch,
        routed=RouteResult("extract", "field:예산"),
        resolution=DocumentResolution("RFP-000001", RESOLVE_EXPLICIT),
        result=expected,
    )
    assert answer is expected
    assert diagnostic["accepted"] is True
    assert diagnostic["reason"] == "confirmed_scalar_extract_contract"


def test_list_extract_is_left_to_stage2(monkeypatch):
    expected = Answer(
        text="항목 1", task_type="extract", route="추출테이블_값조회",
        structured_answer=["항목 1"],
        condition_query=[{"field": "필수 제출 서류", "status": "value_present"}],
    )
    answer, diagnostic = _call(
        "RFP-000001의 필수 제출 서류 알려줘", monkeypatch,
        routed=RouteResult("extract", "field:필수 제출 서류"),
        resolution=DocumentResolution("RFP-000001", RESOLVE_EXPLICIT),
        result=expected,
    )
    assert answer is None
    assert diagnostic["reason"] == "list_shape_requires_stage2"


def test_explanation_request_is_left_to_stage2(monkeypatch):
    monkeypatch.setattr(
        "answer_pipeline.route", lambda _q, _cfg: RouteResult("extract", "field:예산"))
    get_embed, get_gen = _callbacks_that_must_not_run()
    answer, diagnostic = _stage3_fast_path(
        "RFP-000001의 예산이 왜 이렇게 정해졌는지 설명해줘",
        object(), get_embed, get_gen, [], {}, session=SessionState(),
    )
    assert answer is None
    assert diagnostic["reason"] == "explanation_requires_llm"


def test_ambiguous_document_is_left_to_stage2(monkeypatch):
    answer, diagnostic = _call(
        "이 사업의 예산 알려줘", monkeypatch,
        routed=RouteResult("extract", "field:예산"),
        resolution=DocumentResolution(None, "none", candidates=["RFP-1", "RFP-2"]),
    )
    assert answer is None
    assert diagnostic["reason"] == "document_not_uniquely_resolved"


def test_qa_always_uses_stage2_language_path(monkeypatch):
    answer, diagnostic = _call(
        "이 사업의 추진 배경을 설명해줘", monkeypatch,
        routed=RouteResult("qa", None, is_fallback=True),
    )
    assert answer is None
    assert diagnostic["reason"] == "rule_path_requires_language_generation"


def test_fast_path_error_is_recorded_before_stage2_fallback(monkeypatch):
    from answer_pipeline import answer

    monkeypatch.setattr(
        "answer_pipeline._stage3_fast_path",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("broken")),
    )
    fallback = Answer(text="LLM 답", task_type="qa", route="chunks검색_LLM답변")
    monkeypatch.setattr(
        "answer_pipeline.answer_stage2_structgpt", lambda *_a, **_k: fallback)
    result = answer(
        "질문", object(), lambda: object(), lambda: object(), [],
        {"routing_method": "llm_structgpt", "stage3_rule_fast_path_enabled": True},
        get_stage2_agent=lambda: object(),
    )
    assert result is fallback
    assert result.stage3_fast_path["accepted"] is False
    assert result.stage3_fast_path["reason"] == "fast_path_internal_error"
