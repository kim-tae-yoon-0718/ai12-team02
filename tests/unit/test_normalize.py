from grader.normalize import match_short, normalize_text, parse_amount, parse_date


def test_unicode_and_space():
    assert normalize_text(" 조달청 　클라우드 ") == "조달청 클라우드"


def test_parse_date_partial():
    assert parse_date("2026년 9월") == "2026-09"


def test_match_reports_reason():
    ok, why = match_short("5억원", "500,000,000원")
    assert ok
    assert why == "amount"


def test_amount_unit_confusion_is_not_equal():
    """그럴듯하지만 금액 단위만 틀린 답 — 이 도메인에서 가장 비싼 오답 중 하나."""
    assert parse_amount("5억원") == 500_000_000
    assert parse_amount("5천만원") == 50_000_000
    assert parse_amount("500백만원") == 500_000_000
    ok, _ = match_short("5억원", "5천만원")
    assert not ok


def test_short_answer_notation_variants():
    for txt in ["5억원", "500,000,000원", "500000000", "5억", "오억원"]:
        ok, why = match_short("5억원", txt, accept=["500,000,000원"])
        assert ok, f"{txt} 를 오답 처리함 ({why})"


def test_date_variants():
    for txt in ["2026-09-15", "2026.09.15", "2026년 9월 15일"]:
        assert match_short("2026-09-15", txt)[0]


def test_correct_plus_hallucination_is_rejected_by_default():
    """정답 + 근거에 없는 내용 덧붙임 — 기본값(allow_partial=False)은 인정하지 않는다."""
    pred = "5억원이며 지역제한은 없고 컨소시엄은 필수입니다"
    assert match_short("5억원", pred, allow_partial=False)[0] is False
    assert match_short("5억원", pred, allow_partial=True)[0] is True


def test_korean_spacing_is_a_notation_variant():
    """한국어 띄어쓰기는 표기 변형 — 공백만 다른 답은 정답으로 인정한다(3-7).
    (현진 PRAC-EXT-004: 정답 '지역 제한 없음')"""
    assert match_short("지역 제한 없음", "지역제한 없음")[0] is True
    assert match_short("지역제한 없음", "지역 제한 없음")[0] is True
    assert match_short("계약 체결일로부터 4개월", "계약체결일로부터 4개월")[0] is True
    # 공백 무시가 덧붙임까지 통과시키지는 않는다
    assert match_short("지역 제한 없음", "지역제한없음 가점 있음")[0] is False
