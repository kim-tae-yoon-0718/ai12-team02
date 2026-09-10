"""채점기 입력·집계 계약 회귀 테스트 (2026-09-04).

§7 에서 고친 구멍을 못박는다 — 응답 완전성, 실패 응답 처리, 경로 독립 채점,
출처 누락과 오인용 구분, 재정렬 미적용, 심판 표시.

★최종 50문항의 질문·정답은 이 파일에 넣지 않는다. 모두 합성 입력이다.
"""
import pytest
from pydantic import ValidationError

from grader.models import EvaluationItem, Location, ModelResponse
from grader.retrieval import grade_citation, grade_retrieval
from grader.task_scoring import check_format, score_item
from grader.validation import check_response_contract

CFG = {"residual_limit": 20}


def _it(**k):
    b = dict(id="X", question="q", task_type="extraction", answer_type="value", answer_raw="x")
    b.update(k)
    return EvaluationItem.model_validate(b)


def _r(**k):
    b = dict(id="X", answer="", abstained=False)
    b.update(k)
    return ModelResponse(**b)


# ── 8. 응답 누락·중복·추가 id ──────────────────────────────────────────

def test_missing_response_is_a_contract_violation_not_a_smaller_denominator():
    """응답을 안 낸 문항을 건너뛰면 '절반만 제출하면 점수가 오르는' 구멍이 생긴다."""
    items = [_it(id="A"), _it(id="B"), _it(id="C")]
    out = check_response_contract(items, [_r(id="A"), _r(id="B")])
    assert out["ok"] is False
    assert out["missing"] == ["C"]
    assert any("누락" in p for p in out["problems"])


def test_duplicate_response_ids_are_a_contract_violation():
    items = [_it(id="A")]
    out = check_response_contract(items, [_r(id="A"), _r(id="A")])
    assert out["ok"] is False and out["duplicate"] == ["A"]


def test_extra_response_ids_fail_only_when_exact_match_required():
    items = [_it(id="A")]
    resp = [_r(id="A"), _r(id="Z")]
    assert check_response_contract(items, resp, require_exact=True)["ok"] is False
    # 부분집합 모드(CI/개발)는 응답이 더 많은 게 정상 — 누락·중복만 실패다.
    loose = check_response_contract(items, resp, require_exact=False)
    assert loose["ok"] is True and loose["extra"] == ["Z"]


def test_complete_response_set_passes():
    items = [_it(id="A"), _it(id="B")]
    out = check_response_contract(items, [_r(id="B"), _r(id="A")])
    assert out["ok"] is True and out["problems"] == []


# ── 9. answer=null + failure 는 로딩을 중단시키지 않는다 ────────────────

def test_null_answer_with_failure_loads():
    r = ModelResponse(id="A", answer=None, abstained=False, failure="timeout")
    assert r.answer is None and r.answer_text == ""


def test_null_answer_without_failure_is_rejected():
    """사유 없는 null 은 실패인지 빈 답인지 구분이 안 된다 — 통과시키면 안 된다."""
    with pytest.raises(ValidationError):
        ModelResponse(id="A", answer=None, abstained=False)


def test_null_answer_is_scored_without_crashing():
    it = _it(answer_type="value", answer_raw="5억원")
    out = score_item(it, ModelResponse(id="X", answer=None, abstained=False,
                                       failure="generation_error"), CFG)
    assert out["task_score"].score == 0.0


# ── 10. 시스템 실패는 오답과 구분되지만 분모에는 남는다 ─────────────────

def test_system_failure_and_wrong_answer_are_both_zero_but_distinguishable():
    from grader.runner import _missing_row
    row = _missing_row(_it(id="A"))
    assert row["score"] == 0.0
    assert row["final_status"] == "FAIL-system"
    assert row["failure"] == "MISSING_RESPONSE"
    # 분모에 남는 행이므로 집계에 필요한 키가 모두 있어야 한다.
    for key in ("id", "task_type", "answer_type", "field_tag", "answer_source"):
        assert key in row


def test_system_failure_counted_in_report_denominator():
    from grader.diagnostics.report import full_report
    from grader.runner import _missing_row
    rows = [
        {"id": "A", "task_type": "qa", "answer_type": "value", "field_tag": None,
         "answer_source": "chunk", "score": 1.0, "final_status": "PASS",
         "abstention": {}, "failure": None, "route": None},
        _missing_row(_it(id="B", task_type="qa")),
    ]
    rep = full_report(rows)
    assert rep["n_items"] == 2                 # 누락 문항이 분모에서 빠지지 않는다
    assert rep["overall_score"] == 0.5
    assert rep["failures"]["system_error_zero"] == 1


# ── 11. 경로(route)가 달라도 같은 답이면 같은 점수 ──────────────────────

@pytest.mark.parametrize("route", ["chunk_search", "추출테이블_값조회",
                                   "identity_v2_값조회", None])
def test_task_score_is_independent_of_route(route):
    """경로는 진단 축이지 점수 축이 아니다 — 경로별로 점수가 달라지면 안 된다."""
    it = _it(answer_type="value", answer_raw="5억원")
    out = score_item(it, _r(answer="5억원", route=route), CFG)
    assert out["task_score"].score == 1.0
    assert out["final_status"] == "PASS"


# ── 12. 출처 누락과 오인용은 다른 실패다 ───────────────────────────────

def _loc(line=10):
    return Location(document="A", section="S", ref_no="S · 문단 1", line=line)


def test_missing_citation_and_wrong_citation_are_distinguished():
    it = _it(task_type="qa", location=_loc())
    none = grade_citation(it, _r(answer="x"))
    assert none["applicable"] and none["n_citations"] == 0 and none["score"] == 0.0

    wrong = grade_citation(it, _r(answer="x", citations=[_loc(line=999)]))
    assert wrong["score"] == 0.0
    assert wrong["n_citations"] == 1                    # 붙이긴 했다
    assert wrong["n_wrong_citations"] == 1              # 그런데 틀렸다
    assert wrong["n_missing_evidence"] == 1
    assert none["n_citations"] != wrong["n_citations"]  # 두 실패가 같은 숫자로 안 섞인다


def test_citation_graded_even_on_non_search_routes():
    """추출표·identity 로 답해도 근거 좌표는 채점 대상이다 — 여기서 끄면 무감점 구멍."""
    it = _it(task_type="qa", location=_loc())
    nsr = frozenset({"추출테이블_값조회"})
    resp = _r(answer="x", route="추출테이블_값조회", citations=[_loc()])
    cit = grade_citation(it, resp, non_search_routes=nsr)
    assert cit["applicable"] is True and cit["matched"] is True
    assert cit["search_used"] is False
    # 검색 단계 점수는 반대로 N/A 여야 한다.
    assert all(s["applicable"] is False
               for s in grade_retrieval(it, resp, 5, None, 5, non_search_routes=nsr))


# ── 13. 재정렬을 안 썼으면 '미적용'이지 0점이 아니다 ────────────────────

def test_reranker_not_applied_is_na_not_zero():
    it = _it(task_type="qa", location=_loc())
    resp = _r(answer="x", route="chunk_search",
              retrieved=[], reranked=[])          # 재정렬 단계 없음
    stages = {s["stage"]: s for s in grade_retrieval(it, resp, 5, None, 5)}
    rr = stages["reranker_k"]
    assert rr["applicable"] is False
    assert "재정렬" in rr["reason"]
    assert "score" not in rr or rr.get("score") is None   # 0점으로 적히지 않는다


# ── 14. StubJudge 와 실심판은 결과에서 구분된다 ─────────────────────────

def test_stub_judge_is_labelled_and_faithfulness_is_na():
    from grader.runner import faithfulness_axis
    stub = {"kind": "stub", "usable_for_final": False}
    axis = faithfulness_axis([], stub)
    assert axis["applicable"] is False
    assert "Stub" in axis["reason"] or "stub" in axis["reason"]


def test_faithfulness_axis_is_separate_and_na_when_not_run():
    from grader.runner import faithfulness_axis
    real = {"kind": "real", "usable_for_final": True}
    axis = faithfulness_axis([], real)
    assert axis["applicable"] is False and axis["n"] == 0
    assert "mean_score" not in axis          # 안 돌렸으면 숫자를 만들지 않는다


# ── 15. abstained 키 자체가 없으면 형식 오류 ───────────────────────────

def test_missing_abstained_key_is_a_format_error():
    it = _it(answer_type="value", answer_raw="x")
    r = ModelResponse(id="X", answer="x")        # abstained 를 안 냄
    fs = check_format(it, r, CFG)
    assert fs.passed is False
    assert any("abstained" in v for v in fs.violations)
