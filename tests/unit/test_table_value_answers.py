"""B. 추출표 기반 value 문항 — 공식 추출표 값을 그대로 답해도 정답 (2026-09-04 최종).

수정 전 실측(EXT-05): 표 원문 "6,758,571,493원($5,198,901/1$=1,300원, 2024년 기준환율)" → 0
(amount_match_but_verbose), "6,758,571,493달러" → 1(통화 미검사),
"6,758,571,493원이며 별도 예산 3억원이 추가됩니다" → 1(근거 없는 추가 금액 미검사).
문항 ID 하드코딩이 아니라 answer_source=table + 추출표 근거 + answer_normalized 목록의 일반 규칙이다.
"""
import json
import os
from pathlib import Path

import pytest

from grader.models import EvaluationItem, ModelResponse
from grader.normalize import match_short
from grader.task_scoring import grade_short_answer, score_item

CFG = {"residual_limit": 20, "accept_natural_absence_phrasing": True}
GOLD = "6,758,571,493원(부가가치세 등 제반 비용 포함)"
TABLE = "6,758,571,493원($5,198,901/1$=1,300원, 2024년 기준환율)"


def _item(**over):
    base = dict(id="X", question="예산이 얼마야?", task_type="extraction", answer_type="value",
                answer_raw=GOLD, answer_normalized=[TABLE], answer_source="table",
                evidence=[{"kind": "extraction_table", "document": "RFP-000007", "field": "예산",
                           "status": "value_present", "source": "extraction_table_v3"}])
    base.update(over)
    return EvaluationItem.model_validate(base)


def _r(a):
    return ModelResponse(id="X", answer=a, abstained=False, route="추출테이블_값조회")


ACCEPT = ["6,758,571,493원", "해당 사업의 예산은 6,758,571,493원입니다.", GOLD, TABLE]
REJECT = ["6,758,571,493달러", "5,198,901원", "6,758,571,494원",
          "6,758,571,493원이며 별도 예산 3억원이 추가됩니다", "확인할 수 없습니다"]


@pytest.mark.parametrize("a", ACCEPT)
def test_accepted_answers_score_full_in_function_and_item(a):
    assert grade_short_answer(_item(), _r(a), CFG).score == 1.0, a
    assert score_item(_item(), _r(a), CFG)["task_score"].score == 1.0, a


@pytest.mark.parametrize("a", REJECT)
def test_rejected_answers_score_zero_in_function_and_item(a):
    assert grade_short_answer(_item(), _r(a), CFG).score == 0.0, a
    assert score_item(_item(), _r(a), CFG)["task_score"].score == 0.0, a


def test_currency_mismatch_is_caught_even_when_accept_list_mentions_dollars():
    """허용 답에 환율 설명($…)이 있어도 핵심 숫자에 외화 단위가 붙으면 오답."""
    ok, why = match_short(GOLD, "6,758,571,493달러", accept=[TABLE])
    assert ok is False and "currency" in why


def test_unsourced_extra_amount_is_rejected_but_sourced_explanation_is_allowed():
    ok, why = match_short(GOLD, "6,758,571,493원이며 별도 예산 3억원이 추가됩니다", accept=[TABLE])
    assert ok is False and "extra_unsourced" in why
    assert match_short(GOLD, TABLE, accept=[TABLE])[0] is True       # 표 값 그대로
    assert match_short(GOLD, "6,758,571,493원", accept=[TABLE])[0] is True


def test_hangul_syllable_in_words_is_not_counted_as_an_extra_amount():
    """'사업'의 '사'(=4)를 추가 금액으로 세면 정상 답이 오답이 된다."""
    assert match_short("5억원", "해당 사업의 예산은 5억원입니다.")[0] is True


def test_no_table_value_in_accept_list_means_table_verbatim_is_still_verbose():
    """규칙은 일반적이다: 허용 답 목록에 표 값이 없으면 표 원문 그대로도 군더더기다."""
    it = _item(answer_normalized=None)
    assert grade_short_answer(it, _r(TABLE), CFG).score == 0.0


def test_checker_flags_gold_incompatible_with_table_value(tmp_path):
    from checks.check_evalset import check_table_value_compatibility
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"rows": [
        {"document_id": "RFP-000007", "field_name": "예산", "status": "value_present",
         "answer_raw": TABLE, "active": "true"}]}), encoding="utf-8")
    good = {"id": "X", "answer_type": "value", "answer_source": "table", "answer_raw": GOLD,
            "answer_normalized": [TABLE],
            "evidence": [{"kind": "extraction_table", "document": "RFP-000007", "field": "예산", "status": "value_present"}]}
    assert check_table_value_compatibility([good], table) == []
    wrong_amount = dict(good, answer_raw="6,758,571,494원")
    assert any("금액" in e for e in check_table_value_compatibility([wrong_amount], table))
    missing_accept = dict(good, answer_normalized=None)
    assert any("허용 답 목록" in e for e in check_table_value_compatibility([missing_accept], table))


def test_candidate_ext05_from_evalset_file():
    p = os.environ.get("RFP_CANDIDATE_EVALSET")
    if not p or not Path(p).exists():
        pytest.skip("RFP_CANDIDATE_EVALSET 미지정")
    items = {json.loads(l)["id"]: json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}
    it = EvaluationItem.model_validate(items["EXT-05"])
    for a in ACCEPT:
        assert score_item(it, _r(a), CFG)["task_score"].score == 1.0, a
    for a in REJECT:
        assert score_item(it, _r(a), CFG)["task_score"].score == 0.0, a
