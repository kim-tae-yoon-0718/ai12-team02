#@title QA 연결부의 값 보존·문서 범위·실패 처리 조합 검사
#@markdown 합성 문서와 시험용 생성기를 사용해 누락과 거짓 성공을 검사하며 외부 API는 호출하지 않습니다.
"""정답 내용은 시험 자료에만 두고 실제 answer()/answer_qa() 연결을 검사한다."""
from itertools import combinations

import pytest

from conftest import FakeEmbed, FakeGen, OFFICIAL_FIELDS, _rows_for
from answer_pipeline import answer, answer_qa, answer_to_response, build_field_evidence


@pytest.mark.parametrize("fields", list(combinations(OFFICIAL_FIELDS, 2)))
def test_qa_keeps_every_requested_official_field(fields, small_store, base_cfg, identity_index):
    # 질문에 명시한 두 필드의 값을 구별해서 한 값만 넘기는 실수를 잡는다.
    values = {f: {"status": "value_present", "answer_raw": f"확정값_{i}_보존",
                  "answer_normalized": f"확정값_{i}_보존"} for i, f in enumerate(fields)}
    table = _rows_for("RFP-000001", values)
    generator = FakeGen("원문을 바탕으로 보충한 설명입니다.")
    result = answer_qa(f"RFP-000001의 {fields[0]} 및 {fields[1]}를 설명해줘",
                       small_store, table, lambda: FakeEmbed(), lambda: generator,
                       base_cfg, identity=identity_index)
    assert result.failure is None
    assert result.selected_document_ids == ["RFP-000001"]
    for field in fields:
        assert field in result.structured_answer
        assert values[field]["answer_raw"] in result.text
        assert values[field]["answer_raw"] in generator.last_structured_context
    assert len(answer_to_response("synthetic", result)) == 12


@pytest.mark.parametrize("raw", [None, "", " ", "\n\t"])
def test_blank_raw_uses_nonblank_normalized(raw, base_cfg):
    row = {"document_id": "RFP-000001", "field_name": "예산", "status": "value_present",
           "answer_raw": raw, "answer_normalized": "5억 원"}
    evidence = build_field_evidence([row], "RFP-000001", "예산", base_cfg)
    assert evidence.answer_text == "5억 원"
    assert not evidence.abstained


@pytest.mark.parametrize("raw,norm", [
    (None, None), ("", ""), (" ", "\n"), (None, []), (None, {}),
    ("", "[]"), ("", "[ ]"), ([], ""), ({}, ""), ("{}", None),
])
def test_value_present_without_any_value_is_not_success(raw, norm, small_store, base_cfg):
    table = _rows_for("RFP-000001", {"예산": {"status": "value_present", "answer_raw": raw,
                                                "answer_normalized": norm}})
    result = answer("RFP-000001의 예산 알려줘", small_store, lambda: FakeEmbed(),
                    lambda: FakeGen(), table, base_cfg)
    assert result.abstained
    assert result.failure is None
    assert "확인" in result.text


@pytest.mark.parametrize("raw", [0, "0", "0원"])
def test_zero_value_is_never_replaced_by_other_normalized_value(raw, base_cfg):
    row = {"document_id": "RFP-000001", "field_name": "예산", "status": "value_present",
           "answer_raw": raw, "answer_normalized": "9억 원"}
    evidence = build_field_evidence([row], "RFP-000001", "예산", base_cfg)
    assert evidence.answer_text == str(raw)
    assert not evidence.abstained


@pytest.mark.parametrize("text,refused", [
    ("확인할 수 없습니다.", True), ("알 수 없습니다", True), ("모르겠습니다.", True),
    ("제공된 자료만으로는 확인할 수 없습니다.", True),
    ("문서에는 '확인할 수 없습니다'라는 안내와 문의처가 있습니다.", False),
    ("가격은 확인할 수 없습니다. 추진 목적은 노후 시설 개선입니다.", False),
    ("제공된 자료에서는 위험관리 절차를 확인할 수 있습니다.", False),
])
def test_only_unambiguous_whole_answer_refusal_sets_abstained(
        text, refused, small_store, table_rows, base_cfg, identity_index):
    result = answer("RFP-000001의 추진배경을 설명해줘", small_store, lambda: FakeEmbed(),
                    lambda: FakeGen(text), table_rows, base_cfg, identity=identity_index)
    assert result.abstained is refused
    assert result.failure is None
    if refused:
        assert result.citations == []  # '모른다'는 답에 근거를 쓴 척하지 않는다.


@pytest.mark.parametrize("question", [
    "우주관측플랫폼구축사업의 추진배경을 설명해줘",
    "RFP-0000019의 추진배경을 설명해줘",
    "RFP-999999의 추진배경을 설명해줘",
    "그 사업의 추진배경을 설명해줘",
])
def test_unresolved_specific_qa_does_not_search_unrelated_documents(
        question, small_store, table_rows, base_cfg, identity_index):
    e, g = FakeEmbed(), FakeGen("엉뚱한 사업 설명")
    result = answer(question, small_store, lambda: e, lambda: g, table_rows,
                    base_cfg, identity=identity_index)
    assert result.abstained and result.failure is None
    assert e.calls == g.calls == 0
    assert result.contexts == result.retrieved == result.citations == []


@pytest.mark.parametrize("stage", ["embedding", "generation"])
def test_external_failure_is_not_counted_as_success(stage, small_store, table_rows, base_cfg, identity_index):
    class BrokenEmbed(FakeEmbed):
        def embed_query(self, question):
            raise RuntimeError("synthetic embedding error")

    class BrokenGen(FakeGen):
        def generate(self, *args, **kwargs):
            raise RuntimeError("synthetic generation error")

    e = BrokenEmbed() if stage == "embedding" else FakeEmbed()
    g = BrokenGen() if stage == "generation" else FakeGen()
    result = answer("RFP-000001의 추진배경은?", small_store, lambda: e, lambda: g,
                    table_rows, base_cfg, identity=identity_index)
    assert result.abstained and result.failure and result.error_stage
    assert result.citations == []
    assert answer_to_response("synthetic", result)["failure"]


def test_qa_missing_identity_cannot_invent_deadline(small_store, table_rows, base_cfg):
    result = answer_qa("RFP-000001의 마감일과 예산을 설명해줘", small_store, table_rows,
                       lambda: FakeEmbed(), lambda: FakeGen("마감일을 추정하면 안 됩니다."),
                       base_cfg, identity=None)
    assert result.abstained
    assert result.structured_answer["입찰 참여 마감일"]["status"] == "identity_missing"
    assert "예산" in result.structured_answer
