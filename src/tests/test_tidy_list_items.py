"""EXT_LIST_TIDY — 추출표 목록형 값의 서식 흔적 정리 회귀 검사.

추출표 v4 answer_normalized 가 "가."·"(원본대조필)" 같은 조각을 별도 배열 원소로
담아 채점기가 extra 로 세는 문제(작업일지 §6, ext_improve 실험) — as_list_value 에서
정리한다. 기본값 off 는 기존 동작 그대로.
"""
from __future__ import annotations

from answer_pipeline import _tidy_list_items


def test_off_is_identity():
    items = ["가.", "값 A", "(부연)"]
    assert _tidy_list_items(items, "off") == items


def test_basic_drops_bare_enum_tokens():
    items = ["가.", "지방계약법 제92조 해당 안됨", "나.", "본점이 부산", "다.", "직접생산확인증명서 소지"]
    out = _tidy_list_items(items, "basic")
    assert out == ["지방계약법 제92조 해당 안됨", "본점이 부산", "직접생산확인증명서 소지"]


def test_basic_merges_dangling_parenthetical_into_previous():
    items = ["사업자등록증 사본 1부.", "(원본대조필)", "법인등기부등본 1부.", "(개인사업자인 경우 주민등록등본)"]
    out = _tidy_list_items(items, "basic")
    assert out == ["사업자등록증 사본 1부. (원본대조필)", "법인등기부등본 1부. (개인사업자인 경우 주민등록등본)"]


def test_basic_joins_continuation_ending_with_ttoneun():
    items = ["용역실적증명서 원본 또는", "계약서 사본 1부."]
    out = _tidy_list_items(items, "basic")
    assert out == ["용역실적증명서 원본 또는 계약서 사본 1부."]


def test_full_absorbs_subenumerated_items_into_parent():
    items = ["1) 정보시스템 운영 및 유지관리", "가) 인포21 시스템", "나) 공통관리시스템",
             "2) 연계시스템 데이터 처리", "3) 개발/운영 제반 업무"]
    out_basic = _tidy_list_items(items, "basic")
    out_full = _tidy_list_items(items, "full")
    assert len(out_basic) == 5      # basic 은 세부항목을 상위로 올릴 뿐
    assert len(out_full) == 3       # full 은 상위에 흡수
    assert out_full[0].startswith("정보시스템 운영 및 유지관리")


def test_tidy_does_not_touch_clean_lists():
    items = ["전자조달시스템(나라장터)을 통하여 전자적 방식으로 제출",
             "전자조달시스템 이용이 곤란한 경우 전자우편이나 우편으로 제출"]
    assert _tidy_list_items(items, "basic") == items
    assert _tidy_list_items(items, "full") == items
