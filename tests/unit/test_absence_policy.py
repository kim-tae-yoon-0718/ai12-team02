"""§3 자연어 '없음' 채점 — 확정 정책 (2026-09-04).

문서에 항목이 없다는 사실을 **단정한** 답은 같은 뜻으로 인정한다.
정보를 **확인하지 못했다**는 답은 '없음'이 아니다. 기권·빈 답·오류도 아니다.
문장 안에 '없음/없습니다'가 들어 있다는 이유만으로 정답 처리하지 않는다.
"""
import pytest

from grader.models import EvaluationItem, ModelResponse
from grader.normalize import (
    ABSENCE_CONFIRMED, ABSENCE_NOT_VERIFIED, ABSENCE_OTHER, classify_absence,
)
from grader.task_scoring import grade_short_answer, score_item

ON = {"residual_limit": 20, "accept_natural_absence_phrasing": True}
OFF = {"residual_limit": 20, "accept_natural_absence_phrasing": False}


def _it(gold="없음"):
    return EvaluationItem.model_validate(dict(
        id="X", question="지역 제한 조건이 있어?", task_type="extraction",
        answer_type="value", answer_raw=gold, answer_source="table"))


def _r(answer, abstained=False):
    return ModelResponse(id="X", answer=answer, abstained=abstained)


ACCEPT = ["없음", "해당 없음", "지역 제한 없음", "공동수급 조건 없음",
          "문서에 명시되어 있지 않음", "관련 내용이 기재되어 있지 않음",
          "지역 제한 조건은 없습니다.", "지역 제한이 없다고 확인했습니다",
          "해당 사항 없음", "관련 규정이 없습니다"]
REJECT = ["확인할 수 없음", "찾지 못함", "알 수 없음", "자료가 부족함", "현재 확인이 어려움",
          "관련 내용이 없다고 확인할 수 없음", "문서를 찾지 못해 답할 수 없음",
          "관련 내용이 없는지는 알 수 없습니다", "지역 제한이 없다고 확인할 수 없습니다",
          "문서에서 확인할 수 없습니다.", "정보가 부족하여 판단할 수 없습니다",
          "오류: 문서를 불러오지 못했습니다"]


# ── 분류기 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ACCEPT)
def test_confirmed_absence_is_classified_confirmed(text):
    assert classify_absence(text) == ABSENCE_CONFIRMED, text


@pytest.mark.parametrize("text", REJECT)
def test_not_verified_is_classified_not_verified(text):
    assert classify_absence(text) == ABSENCE_NOT_VERIFIED, text


@pytest.mark.parametrize("text", ["", "   ", "5억원", "2024년 6월 1일", "공동이행방식만 허용"])
def test_non_absence_text_is_other(text):
    assert classify_absence(text) == ABSENCE_OTHER, text


def test_real_model_sentence_with_interpretive_caveat_is_confirmed_absence():
    """실제 모델 문장(EXT-09/11). 본문이 항목 부재를 **단정**하고 괄호는 값 해석 유보다 —
    부재 주장을 철회한 것이 아니므로 확정 부재. 단, 이 판정은 분류기가 내려야 하고
    어절 '없음' 포함으로 새어 들어가면 안 된다(수정 전 실측: content_equivalent 로 1.0)."""
    t = "원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)."
    assert classify_absence(t) == ABSENCE_CONFIRMED
    s = grade_short_answer(_it(), _r(t), ON)
    assert s.score == 1.0 and "absence_confirmed" in s.detail["why"]


def test_negated_absence_claim_is_not_verified():
    for t in ("관련 내용이 없다고 단정할 수 없음", "없다고 판단할 수 없습니다",
              "값이 없음을 확인할 수 없습니다"):
        assert classify_absence(t) == ABSENCE_NOT_VERIFIED, t
        assert grade_short_answer(_it(), _r(t), ON).score == 0.0, t


def test_absence_gold_never_uses_token_containment():
    """'없음' 어절이 들어 있을 뿐인 문장은 분류기가 OTHER 이면 0 — 어절 포함 경로 금지."""
    s = grade_short_answer(_it(), _r("예산 항목의 값은 5억원이며, 그 외 사항은 없음 처리 대상"), ON)
    assert s.score == 0.0 and "absence_unclear" in s.detail["why"]
    assert grade_short_answer(_it(), _r("지역제한: 없음"), ON).score == 1.0


def test_double_negation_is_resolved_by_checking_failure_first():
    """★부정이 겹치면 '확인 실패' 신호가 우선한다 — 순서가 정책이다."""
    assert classify_absence("지역 제한이 없다고 확인했습니다") == ABSENCE_CONFIRMED
    assert classify_absence("지역 제한이 없다고 확인할 수 없습니다") == ABSENCE_NOT_VERIFIED
    assert classify_absence("관련 내용이 없는지는 알 수 없습니다") == ABSENCE_NOT_VERIFIED
    assert classify_absence("관련 내용이 없다고 확인할 수 없음") == ABSENCE_NOT_VERIFIED


# ── 채점 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ACCEPT)
def test_confirmed_absence_scores_full_by_default(text):
    assert score_item(_it(), _r(text), ON)["task_score"].score == 1.0, text


@pytest.mark.parametrize("text", REJECT)
def test_not_verified_never_scores(text):
    """★토글 값과 무관하게 0 — '없음' 토큰이 들어 있어도 어절 포함 경로로 새지 않는다."""
    assert score_item(_it(), _r(text), ON)["task_score"].score == 0.0, text
    assert score_item(_it(), _r(text), OFF)["task_score"].score == 0.0, text


def test_containment_hole_is_closed():
    """수정 전 실측: '확인할 수 없음'·'알 수 없음'이 어절 '없음' 포함으로 1.0 이었다."""
    for t in ("확인할 수 없음", "알 수 없음", "문서를 찾지 못해 답할 수 없음"):
        s = grade_short_answer(_it(), _r(t), ON)
        assert s.score == 0.0, t
        assert "not_verified" in s.detail["why"]


def test_abstained_empty_and_error_are_not_absence():
    assert score_item(_it(), _r("없음", abstained=True), ON)["task_score"].score == 0.0
    assert score_item(_it(), _r("", abstained=False), ON)["task_score"].score == 0.0
    assert score_item(_it(), _r("오류: 문서를 불러오지 못했습니다"), ON)["task_score"].score == 0.0


@pytest.mark.parametrize("gold", ["없음", "해당 없음", "지역제한 없음"])
def test_gold_absence_variants_accept_confirmed_phrasing(gold):
    assert score_item(_it(gold), _r("문서에 명시되어 있지 않음"), ON)["task_score"].score == 1.0
    assert score_item(_it(gold), _r("확인할 수 없음"), ON)["task_score"].score == 0.0


def test_absence_gate_does_not_touch_non_absence_golds():
    """정답이 부재가 아니면 게이트는 개입하지 않는다 — 기존 채점 그대로."""
    it = _it("5억원")
    assert score_item(it, _r("5억원"), ON)["task_score"].score == 1.0
    assert score_item(it, _r("없음"), ON)["task_score"].score == 0.0
    assert score_item(it, _r("확인할 수 없음"), ON)["task_score"].score == 0.0


def test_default_config_enables_the_confirmed_policy():
    from pathlib import Path
    from grader.config import load_config
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(str(repo / "config/grader.yaml"))
    assert cfg.grading.accept_natural_absence_phrasing is True


# ── A. 검토에서 재현된 오판정 (2026-09-04 최종) ─────────────────────────
# 수정 전: 괄호를 먼저 지우고 바깥의 '없음'으로 확정 → "없음(확인할 수 없음)" 이 부재로 둔갑.
#          '없지는 않다'·'없을 수도 있다'·'없다는 의미는 아니다'도 광역 패턴에 걸려 부재로 확정.

A_REJECT = ["없음(확인할 수 없음)", "없음(문서를 찾지 못함)", "없음(자료 부족)",
            "지역 제한 없음(문서를 찾지 못함)", "해당 항목은 없습니다(자료 부족으로 확인할 수 없음)",
            "지역 제한이 없지는 않습니다", "지역 제한이 없는 것은 아닙니다", "지역 제한이 없을 수도 있습니다",
            "지역 제한이 없다고 볼 수는 없습니다", "지역 제한이 없다는 의미는 아닙니다",
            "관련 조건이 없다는 의미는 아닙니다", "관련 조건이 없다고는 할 수 없습니다",
            "확인할 수 없습니다", "알 수 없습니다", "검색 결과가 없습니다", "응답이 없습니다",
            "답변을 생성하지 못했습니다", "지역 제한 여부는 불확실합니다"]
A_ACCEPT = ["원문에 해당 항목이 없습니다", "문서에 지역제한 항목이 기재되어 있지 않습니다",
            "해당 필드는 미기재입니다", "원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)"]


@pytest.mark.parametrize("text", A_REJECT)
def test_uncertain_or_reversed_absence_is_not_confirmed(text):
    """판정 함수와 실제 채점 경로(토글 ON/OFF) 모두에서 0 이어야 한다."""
    assert classify_absence(text) != ABSENCE_CONFIRMED, text
    for cfg in (ON, OFF):
        assert score_item(_it(), _r(text), cfg)["task_score"].score == 0.0, (text, cfg)


@pytest.mark.parametrize("text", A_ACCEPT)
def test_clear_document_absence_is_confirmed(text):
    assert classify_absence(text) == ABSENCE_CONFIRMED, text
    assert score_item(_it(), _r(text), ON)["task_score"].score == 1.0, text


def test_parenthetical_is_not_blindly_dropped():
    """괄호 안 확인 실패는 바깥 '없음'을 뒤집는다 — 괄호 선삭제 금지."""
    assert classify_absence("없음(확인할 수 없음)") == ABSENCE_NOT_VERIFIED
    assert classify_absence("없음(검색 실패)") != ABSENCE_CONFIRMED


def test_field_absent_caveat_exception_is_narrow():
    """예외는 '원문에 항목 자체가 없다' 본문 + 해석 유보 괄호에만 열린다."""
    assert classify_absence("원문에 해당 항목 자체가 없습니다(값이 없다는 뜻으로 단정할 수 없음)") == ABSENCE_CONFIRMED
    # 같은 본문이라도 괄호가 검색·확인 실패면 예외 아님
    assert classify_absence("원문에 해당 항목 자체가 없습니다(문서를 찾지 못함)") == ABSENCE_NOT_VERIFIED
    assert classify_absence("원문에 해당 항목 자체가 없습니다(확인할 수 없음)") == ABSENCE_NOT_VERIFIED
    # 본문이 넓은 '없음'이면 해석 유보 괄호라도 예외 아님
    assert classify_absence("없음(값이 없다는 뜻으로 단정할 수 없음)") != ABSENCE_CONFIRMED


def test_toggle_off_does_not_leak_through_token_containment():
    for text in ("없음(확인할 수 없음)", "지역 제한이 없지는 않습니다"):
        assert grade_short_answer(_it(), _r(text), OFF).score == 0.0, text
