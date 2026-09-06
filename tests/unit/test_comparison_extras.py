"""결함 C — 비교형에서 관계없는 추가 내용이 감점되지 않던 문제 (2026-09-04).

★예전에는 **정답에 있는 칸만** 확인했다. 그래서 정답 칸을 다 맞힌 뒤 묻지도 않은
  문서나 필드를 아무리 덧붙여도 1.0 이 나왔다. 비교표에서 '없는 문서를 끌어와
  붙이는 것'은 대표적인 환각인데 감점이 없었다.
"""
import pytest

from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import grade_comparison

GOLD = [
    {"항목": "예산", "RFP-000014": "90,000,000원", "RFP-000020": "1,400,000,000원"},
    {"항목": "사업기간", "RFP-000014": "90일", "RFP-000020": "2024년 12월 31일까지"},
]
PERFECT = {
    "RFP-000014": {"예산": "90,000,000원", "사업기간": "90일"},
    "RFP-000020": {"예산": "1,400,000,000원", "사업기간": "2024년 12월 31일까지"},
}


def _item():
    return EvaluationItem.model_validate(dict(
        id="QA-005", question="q", task_type="qa", answer_type="comparison",
        answer_raw=GOLD))


def _resp(structured):
    return ModelResponse(id="QA-005", answer="| 항목 |", abstained=False,
                         structured_answer=structured)


def grade(structured):
    return grade_comparison(_item(), _resp(structured))


# ── 1. 정답과 정확히 같으면 만점 ──────────────────────────────────────

def test_exact_match_is_full_score():
    s = grade(PERFECT)
    assert s.score == 1.0
    assert s.detail["n_extra"] == 0
    assert s.detail["missing_cells"] == [] and s.detail["wrong_cells"] == []


# ── 2·3. 누락과 값 오류 ───────────────────────────────────────────────

def test_missing_cell_is_below_full_score():
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    del tbl["RFP-000020"]["사업기간"]
    s = grade(tbl)
    assert s.score < 1.0
    assert len(s.detail["missing_cells"]) == 1
    assert s.detail["missing_cells"][0]["document_id"] == "RFP-000020"


def test_wrong_value_is_below_full_score_and_not_counted_as_matched():
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    tbl["RFP-000014"]["예산" ] = "99,999,999원"
    s = grade(tbl)
    assert s.score < 1.0
    assert len(s.detail["wrong_cells"]) == 1
    assert len(s.detail["matched_cells"]) == 3      # 값 틀린 칸은 맞은 칸이 아니다


# ── 4·5·6. 추가 문서·필드는 반드시 감점 ───────────────────────────────

def test_extra_document_is_penalized():
    """★정답을 모두 맞혔더라도 묻지 않은 문서를 붙이면 만점이 아니다."""
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    tbl["RFP-000099"] = {"예산": "5억원", "사업기간": "6개월"}
    s = grade(tbl)
    assert s.score < 1.0
    assert len(s.detail["extra_document_cells"]) == 2
    assert s.detail["denominator"] == 4 + 2
    assert s.score == pytest.approx(4 / 6)


def test_extra_field_on_a_gold_document_is_penalized():
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    tbl["RFP-000014"]["평가 배점"] = "100점"
    s = grade(tbl)
    assert s.score < 1.0
    assert len(s.detail["extra_field_cells"]) == 1
    assert s.detail["extra_document_cells"] == []
    assert s.score == pytest.approx(4 / 5)


def test_well_formed_but_unasked_document_id_is_still_penalized():
    """형식이 정상인 RFP ID 라도 정답에 없으면 추가다."""
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    tbl["RFP-000001"] = {"예산": "3억원"}
    s = grade(tbl)
    assert s.score < 1.0
    assert s.detail["extra_document_cells"][0]["document_id"] == "RFP-000001"


# ── 7. 문서 ID 형식 오류 ──────────────────────────────────────────────

def test_malformed_document_id_is_recorded_and_blocks_full_score():
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    tbl["그 문서"] = {"예산": "3억원"}
    s = grade(tbl)
    assert s.score < 1.0
    assert any("형식 아님" in e for e in s.detail["pred_errors"])


def test_gold_document_ids_are_not_flagged_as_malformed():
    """정답 자체가 쓰는 표기는 형식 오류가 아니다 — 검사 대상은 '정답에 없는' ID 다."""
    it = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="qa", answer_type="comparison",
        answer_normalized=["A|예산|5억원", "B|예산|7억원"]))
    r = ModelResponse(id="X", answer="| A | B |", abstained=False,
                      structured_answer={"A": {"예산": "5억원"}, "B": {"예산": "7억원"}})
    s = grade_comparison(it, r)
    assert s.detail.get("pred_errors", []) == []
    assert s.score == 1.0


# ── 8. 구조 오류는 실행을 멈추지 않고 그 문항만 0점 ───────────────────

@pytest.mark.parametrize("bad", [
    {"RFP-000014": "예산 9천만원"},              # 중첩이 dict 아님
    {"RFP-000014": ["예산", "9천만원"]},          # 중첩이 리스트
])
def test_non_dict_nesting_scores_zero_without_raising(bad):
    s = grade(bad)
    assert s.score == 0.0
    assert any("dict 아님" in e for e in s.detail["structure_errors"])


def test_top_level_not_a_dict_scores_zero_without_raising():
    s = grade(["RFP-000014", "RFP-000020"])
    assert s.score == 0.0
    assert any("structured_answer 가 dict 아님" in e for e in s.detail["structure_errors"])


# ── 9. 상세 결과에서 정답 칸과 추가 칸을 구분할 수 있다 ────────────────

def test_detail_separates_gold_cells_from_extras():
    tbl = {k: dict(v) for k, v in PERFECT.items()}
    tbl["RFP-000014"]["평가 배점"] = "100점"      # 추가 필드
    tbl["RFP-000099"] = {"예산": "5억원"}          # 추가 문서
    del tbl["RFP-000020"]["사업기간"]              # 누락
    d = grade(tbl).detail
    assert {"matched_cells", "missing_cells", "wrong_cells",
            "extra_document_cells", "extra_field_cells", "n_extra",
            "denominator"} <= set(d)
    assert len(d["matched_cells"]) == 3
    assert len(d["missing_cells"]) == 1
    assert [x["document_id"] for x in d["extra_document_cells"]] == ["RFP-000099"]
    assert [x["field"] for x in d["extra_field_cells"]] == ["평가 배점"]
    assert d["n_extra"] == 2 and d["denominator"] == 6
    assert d["conclusion_at_risk"] is True
