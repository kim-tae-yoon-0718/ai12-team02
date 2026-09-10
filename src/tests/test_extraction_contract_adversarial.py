#@title 추출·비교 응답의 값·상태·근거 보존 검사
#@markdown 공식 정답을 외우지 않는 합성 문서로 실제 answer 경로를 검사하며 유료 API는 호출하지 않습니다.
"""12필드 교차 검사와 여러 항목·마감일 조립의 독립 회귀 검사."""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import pytest

# 같은 검사를 수정 전 코드에도 실행할 수 있게 검토 전용 경로를 선택한다.
if os.environ.get("QA_EXTRACTION_REVIEW_SOURCE_ROOT"):
    _root = Path(os.environ["QA_EXTRACTION_REVIEW_SOURCE_ROOT"])
    sys.path[:0] = [str(_root / "src" / part) for part in ("rag", "scripts", "tests")]

import test_selection_fix as h
from answer_pipeline import SessionState, answer_to_response


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



DOC_A = "RFP-000971"
DOC_B = "RFP-000972"
FIELDS = tuple(h.OFFICIAL_FIELDS)
DEADLINE = "입찰 참여 마감일"
RESPONSE_KEYS = {
    "id", "answer", "structured_answer", "contexts", "retrieved", "citations",
    "selected_document_ids", "abstained", "route", "failure", "latency_ms", "cost_usd",
}


def _world(tmp_path, overrides=None, *, missing_deadline=False):
    """문서·필드마다 서로 다른 값과 위치를 줘 값/근거 혼입을 잡는다."""
    specifications = {
        document: {
            field: h.present(f"자료{document[-3:]}항목{index}값", 100 + index * 10)
            for index, field in enumerate(FIELDS)
        }
        for document in (DOC_A, DOC_B)
    }
    for document, fields in (overrides or {}).items():
        specifications[document].update(fields)
    table = h.write_table(tmp_path, specifications)
    identity = h.write_identity(tmp_path, [
        (DOC_A, "새봄연구원", "합성 자료 분석 사업", "2025-01-02 17:30:00"),
        (DOC_B, "가람연구원", "합성 자료 구축 사업",
         "" if missing_deadline else "2025-01-03 14:00:00"),
    ])
    return table, identity


def _answer(question, table, identity, *, session=None):
    """실제 문서 해석·라우터·응답 생성기를 실행하고 호출 금지 API를 연결한다."""
    result = h.run_answer(question, table, identity=identity, session=session)
    response = answer_to_response("SYNTHETIC", result)
    assert set(response) == RESPONSE_KEYS
    assert response["failure"] is None, response
    assert result.error_stage is None, response
    assert isinstance(response["abstained"], bool)
    assert response["contexts"] == [] and response["retrieved"] == []
    assert len(response["selected_document_ids"]) == len(set(response["selected_document_ids"]))
    # 응답의 파일 저장 형식이 모든 경로에서 정상 JSON으로 유지되는지 확인한다.
    assert json.loads(json.dumps(response, ensure_ascii=False)) == response
    return result


def _status_row(status, field_index):
    if status == "value_present":
        return h.present(f"확정항목{field_index}값", 410)
    if status == "conflict":
        return h.conflict("첫번째상충내용", "두번째상충내용")
    if status == "external_reference":
        return h.external()
    if status == "not_disclosed":
        return h.not_disclosed()
    return {"status": status, "answer_raw": "", "answer_normalized": "",
            "representative_location": None, "additional_locations": []}


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("status", [
    "value_present", "field_absent", "external_reference", "not_disclosed",
    "conflict", "extraction_failed", "review_required",
])
def test_all_twelve_fields_preserve_each_state(tmp_path, field, status):
    table, identity = _world(tmp_path, {DOC_A: {field: _status_row(status, FIELDS.index(field))}})
    result = _answer(f"{DOC_A}의 {field}을 알려줘", table, identity)
    assert result.selected_document_ids == [DOC_A]
    assert result.structured_answer[field]["status"] == status
    assert result.abstained is (status in {"conflict", "extraction_failed", "review_required"})
    assert all(c["document"] == DOC_A for c in result.citations)
    if status == "value_present":
        assert f"확정항목{FIELDS.index(field)}값" in result.text
    elif status == "conflict":
        assert "첫번째상충내용" in result.text and "두번째상충내용" in result.text
        assert {c["line"] for c in result.citations} == {40, 50}
    elif status == "field_absent":
        # [2026-09-04 §8 계약 변경] 미기재도 **근거는 있다** — 추출표의 그 행이다.
        #   예전에는 좌표가 없다는 이유로 인용을 통째로 비웠다(출처 확인 불가).
        #   이제 Evidence 형으로 내되, 없는 좌표를 지어내지 않는지 더 세게 본다.
        assert "단정" in result.text
        assert len(result.citations) == 1
        cite = result.citations[0]
        assert cite["kind"] == "extraction_table"
        assert cite["document"] == DOC_A and cite["field"] == field
        assert cite["status"] == "field_absent"
        assert str(cite.get("source", "")).startswith("extraction_table")
        # 가짜 좌표 금지 — 위치를 나타내는 키가 하나도 없어야 한다
        for forbidden in ("location", "line", "line_end", "ref_no", "section",
                          "block_type", "block_index"):
            assert forbidden not in cite, forbidden
    elif status == "external_reference":
        assert "직접 확인" in result.text
    elif status == "not_disclosed":
        assert "비공개" in result.text


@pytest.mark.parametrize("fields", list(itertools.combinations(FIELDS, 2)))
def test_two_requested_fields_never_silently_drop_one(tmp_path, fields):
    table, identity = _world(tmp_path)
    result = _answer(f"{DOC_A}의 {'와 '.join(fields)}을 알려줘", table, identity)
    assert result.abstained is False
    assert result.selected_document_ids == [DOC_A]
    assert set(result.structured_answer) == set(fields)
    assert {c["field"] for c in result.citations} == set(fields)
    for field in fields:
        assert f"자료971항목{FIELDS.index(field)}값" in result.text


def test_all_twelve_requested_fields_preserved_without_llm(tmp_path):
    table, identity = _world(tmp_path)
    result = _answer(f"{DOC_A}의 {', '.join(FIELDS)}을 알려줘", table, identity)
    assert result.abstained is False
    assert set(result.structured_answer) == set(FIELDS)
    assert {c["field"] for c in result.citations} == set(FIELDS)


@pytest.mark.parametrize("status", ["field_absent", "external_reference", "not_disclosed",
                                    "conflict", "extraction_failed", "review_required"])
def test_mixed_known_and_unresolved_fields_keep_both(tmp_path, status):
    table, identity = _world(tmp_path, {DOC_A: {"예산": _status_row(status, 4)}})
    result = _answer(f"{DOC_A}의 예산과 사업기간을 알려줘", table, identity)
    assert set(result.structured_answer) == {"예산", "사업기간"}
    assert result.structured_answer["예산"]["status"] == status
    assert result.structured_answer["사업기간"]["status"] == "value_present"
    assert result.abstained is (status in {"conflict", "extraction_failed", "review_required"})
    assert "자료971항목3값" in result.text
    if status == "conflict":
        assert {40, 50}.issubset({c["line"] for c in result.citations})


@pytest.mark.parametrize("field", FIELDS)
def test_field_and_deadline_are_both_answered(tmp_path, field):
    table, identity = _world(tmp_path)
    result = _answer(f"{DOC_A}의 {field}과 마감일을 알려줘", table, identity)
    assert result.abstained is False
    assert set(result.structured_answer) == {field, DEADLINE}
    assert f"자료971항목{FIELDS.index(field)}값" in result.text
    assert "2025-01-02 17:30" in result.text
    assert any(c.get("field") == field for c in result.citations)
    deadline_cites = [c for c in result.citations if c.get("kind") == "identity"]
    assert len(deadline_cites) == 1
    assert_identity_deadline_citation(deadline_cites[0])


def test_deadline_single_field_keeps_existing_flat_structure(tmp_path):
    table, identity = _world(tmp_path)
    result = _answer(f"{DOC_A}의 마감일을 알려줘", table, identity)
    assert result.text == "2025-01-02 17:30"
    assert result.structured_answer["column"] == "bid_deadline"
    assert result.structured_answer["value_datetime"] == "2025-01-02 17:30"
    assert len(result.citations) == 1


@pytest.mark.parametrize("field", FIELDS)
def test_comparison_field_and_deadline_are_both_answered(tmp_path, field):
    table, identity = _world(tmp_path)
    result = _answer(f"{DOC_A}와 {DOC_B}의 {field}과 마감일을 비교해줘", table, identity)
    assert result.abstained is False
    assert set(result.structured_answer) == {DOC_A, DOC_B}
    for document in (DOC_A, DOC_B):
        assert set(result.structured_answer[document]) == {field, DEADLINE}
        assert f"자료{document[-3:]}항목{FIELDS.index(field)}값" in result.structured_answer[document][field]
    assert result.structured_answer[DOC_A][DEADLINE] == "2025-01-02 17:30"
    assert result.structured_answer[DOC_B][DEADLINE] == "2025-01-03 14:00"
    assert len(result.citations) == 4


@pytest.mark.parametrize("missing_deadline", [False, True])
def test_comparison_only_deadline_uses_identity(tmp_path, missing_deadline):
    table, identity = _world(tmp_path, missing_deadline=missing_deadline)
    result = _answer(f"{DOC_A}와 {DOC_B}의 마감일을 비교해줘", table, identity)
    assert set(result.structured_answer) == {DOC_A, DOC_B}
    assert result.abstained is missing_deadline
    assert result.structured_answer[DOC_A][DEADLINE] == "2025-01-02 17:30"
    assert result.citations
    for c in result.citations:
        assert_identity_deadline_citation(c)
    if missing_deadline:
        assert "미상" in result.structured_answer[DOC_B][DEADLINE]
        assert "2025-01-02" not in result.structured_answer[DOC_B][DEADLINE]


@pytest.mark.parametrize("status", ["value_present", "field_absent", "external_reference",
                                    "not_disclosed", "conflict", "extraction_failed", "review_required"])
def test_comparison_mixed_status_never_converts_it_to_value(tmp_path, status):
    table, identity = _world(tmp_path, {DOC_B: {"예산": _status_row(status, 4)}})
    result = _answer(f"{DOC_A}와 {DOC_B}의 예산과 사업기간을 비교해줘", table, identity)
    assert result.abstained is (status in {"conflict", "extraction_failed", "review_required"})
    assert set(result.structured_answer) == {DOC_A, DOC_B}
    assert set(result.structured_answer[DOC_B]) == {"예산", "사업기간"}
    if status == "conflict":
        assert "첫번째상충내용" in result.structured_answer[DOC_B]["예산"]
        assert "두번째상충내용" in result.structured_answer[DOC_B]["예산"]
    elif status != "value_present":
        assert "확정항목4값" not in result.structured_answer[DOC_B]["예산"]


@pytest.mark.parametrize("raw,norm,expected", [
    (0, "0", "0"), ("0", "0", "0"), ("0원", "0원", "0원"),
    (None, "1억원", "1억원"), ("", "1억원", "1억원"),
    ("49,500천원", "", "49,500천원"),
    ("다. 예산소요액 : 금 248,796천원(부가세 포함)", "", "248,796천원(부가세 포함)"),
])
def test_scalar_zero_and_existing_value_fallback(tmp_path, raw, norm, expected):
    row = h.present(raw)
    row["answer_normalized"] = norm
    table, identity = _world(tmp_path, {DOC_A: {"예산": row}})
    result = _answer(f"{DOC_A}의 예산을 알려줘", table, identity)
    assert result.text == expected
    assert result.abstained is False


@pytest.mark.parametrize("norm", [
    ["사업자등록증", "제안서", "제안요약서"],
    '["사업자등록증", "제안서", "제안요약서"]',
])
def test_single_list_keeps_grader_array_contract(tmp_path, norm):
    row = h.present("사업자등록증\n제안서\n제안요약서")
    row["answer_normalized"] = norm
    table, identity = _world(tmp_path, {DOC_A: {"필수 제출 서류": row}})
    result = _answer(f"{DOC_A}의 필수 제출 서류를 알려줘", table, identity)
    assert result.structured_answer == ["사업자등록증", "제안서", "제안요약서"]
    assert result.abstained is False


def test_multifield_list_keeps_all_items_and_scalar(tmp_path):
    row = h.present("사업자등록증\n제안서\n제안요약서")
    row["answer_normalized"] = '["사업자등록증", "제안서", "제안요약서"]'
    table, identity = _world(tmp_path, {DOC_A: {"필수 제출 서류": row}})
    result = _answer(f"{DOC_A}의 예산과 필수 제출 서류를 알려줘", table, identity)
    assert set(result.structured_answer) == {"예산", "필수 제출 서류"}
    assert result.structured_answer["필수 제출 서류"]["items"] == ["사업자등록증", "제안서", "제안요약서"]
    assert "자료971항목4값" in result.text


@pytest.mark.parametrize("field,value", [
    ("평가 배점", "기술평가: 90점, 가격평가: 10점"),
    ("참가 자격(면허·실적)", "기술인력: 3명 이상, 사업실적: 2건 이상"),
    ("과업 범위", "데이터 수집: 100건, 품질 검사: 100건"),
])
def test_semantic_sub_labels_are_not_stripped_as_decoration(tmp_path, field, value):
    table, identity = _world(tmp_path, {DOC_A: {field: h.present(value)}})
    result = _answer(f"{DOC_A}의 {field}을 알려줘", table, identity)
    assert result.text == value


def test_session_switch_then_multiple_fields_uses_latest_document(tmp_path):
    table, identity = _world(tmp_path)
    session = SessionState()
    first = _answer(f"{DOC_A}의 예산을 알려줘", table, identity, session=session)
    assert first.selected_document_ids == [DOC_A]
    second = _answer(f"{DOC_B}의 사업기간을 알려줘", table, identity, session=session)
    assert second.selected_document_ids == [DOC_B]
    third = _answer("그 사업의 예산과 사업기간을 알려줘", table, identity, session=session)
    assert third.selected_document_ids == [DOC_B]
    assert set(third.structured_answer) == {"예산", "사업기간"}
    assert "자료971" not in third.text
    assert all(c["document"] == DOC_B for c in third.citations)


@pytest.mark.parametrize("value", [
    "사업자등록증\n제안서", "본사 | 지사", "서울\r\n경기", "서울 | 경기\n부산",
])
def test_comparison_escapes_display_but_preserves_machine_value(tmp_path, value):
    table, identity = _world(tmp_path, {DOC_A: {"필수 제출 서류": h.present(value)}})
    result = _answer(f"{DOC_A}와 {DOC_B}의 필수 제출 서류를 비교해줘", table, identity)
    # 기계가 채점할 값은 원래 값 그대로이고 화면용 표에서만 문법을 정리한다.
    assert result.structured_answer[DOC_A]["필수 제출 서류"] == value
    display = value.replace("\r\n", "\n").replace("\r", "\n").replace("|", "\\|").replace("\n", "<br>")
    assert f"| {DOC_A} | {display} |" in result.text
    rows = [line for line in result.text.splitlines() if line.startswith("| RFP-")]
    assert len(rows) == 2 and all(row.endswith(" |") for row in rows)
