from answer_pipeline import as_list_value


def test_list_value_drops_standalone_enumeration_markers():
    row = {
        "answer_raw": "가. 첫 번째 조건\n나. 두 번째 조건\n사. 세 번째 조건",
    }
    assert as_list_value(row) == ["첫 번째 조건", "두 번째 조건", "세 번째 조건"]


def test_list_value_joins_wrapped_continuation_lines():
    row = {
        "answer_raw": "인감증명서 1부.\n(법인 인감 사용)\n계약서 사본 1부.\n청렴서약서 1부.",
    }
    assert as_list_value(row) == [
        "인감증명서 1부. (법인 인감 사용)",
        "계약서 사본 1부.",
        "청렴서약서 1부.",
    ]


def test_list_value_does_not_swallow_standalone_item_matching_join_keyword():
    """특정 낱말로 시작한다는 이유만으로 병합하면 그 낱말로 시작하는 진짜
    별도 서류 항목이 앞 항목에 흡수돼 사라진다(회귀: 코드리뷰 지적)."""
    row = {
        "answer_raw": "인감증명서 1부.\n계약서 사본 1부.\n청렴서약서 1부.",
    }
    assert as_list_value(row) == [
        "인감증명서 1부.",
        "계약서 사본 1부.",
        "청렴서약서 1부.",
    ]
