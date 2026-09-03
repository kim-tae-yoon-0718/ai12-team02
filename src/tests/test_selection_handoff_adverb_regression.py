"""'따로' 회귀 수정이 남긴 한 글자 값 누락 결함의 회귀 테스트 (2026-09-03).

배경 — "따로"를 "따 + 로"로 잘못 읽던 문제를 고치면서 "실제 값은 두 글자 이상"
이라는 글자 수 규칙이 들어갔다. 그 규칙 때문에 한 글자 실제 값(웹·앱·0)까지
값 조건에서 사라져, 조건을 조용히 버린 채 모든 문서를 확정 답변으로 돌려줬다.

여기서 지키는 원칙은 두 가지다.
  A. 부사(따로·별도로·새로…)는 값이 아니라 '어떻게 기재됐는가'를 꾸미므로
     정상 상태 조회로 남아야 한다.
  B. 지원하지 않는 값 조건은 **버리지 않고 되묻는다** — 한 글자여도 마찬가지다.

⚠️ 자료는 전부 이 파일 안에서 만든 합성 자료이고, 문서 ID도 평가셋과 겹치지 않는
   대역(RFP-0009xx)을 쓴다. 검사는 파서만 따로 부르지 않고 실제 진입점 answer()
   를 그대로 탄다. 생성·임베딩 클라이언트는 "호출되면 실패"하는 감시자를 재사용해,
   선별 경로가 유료 API를 쓰지 않는다는 사실도 함께 검증한다.
"""
from __future__ import annotations

import pytest

from test_selection_fix import (
    absent,
    present,
    run_answer,
    write_identity,
    write_registry,
    write_table,
)


# ---------------------------------------------------------------------------
# 합성 자료 — 사업분야·예산·제출 방식이 서로 다른 문서 3개
# ---------------------------------------------------------------------------
#   RFP-000981  사업분야 웹        예산 0원      제출 방식 방문접수
#   RFP-000982  사업분야 앱        예산 1억 원   제출 방식 전자접수
#   RFP-000983  사업분야 건물 청소  예산 6억 원   제출 방식 전자접수
# 세 문서 모두 활성·검색 대상이고 마감일도 기준시각 이후여서 마감 필터를 통과한다.

_DOC_SPECS = {
    "RFP-000981": {"사업분야": "웹", "예산": "0원", "제출 방식": "방문접수"},
    "RFP-000982": {"사업분야": "앱", "예산": "1억 원", "제출 방식": "전자접수"},
    "RFP-000983": {"사업분야": "건물 청소", "예산": "6억 원", "제출 방식": "전자접수"},
}
ALL_DOCS = sorted(_DOC_SPECS)


@pytest.fixture
def value_world(tmp_path):
    """한 글자 값과 여러 글자 값이 섞인 작은 세계."""
    specs = {}
    for doc_id, values in _DOC_SPECS.items():
        over = {field: present(value) for field, value in values.items()}
        over["사업 개요"] = absent()          # 값 없는 필드도 하나 섞어 둔다
        specs[doc_id] = over
    table = write_table(tmp_path, specs)
    scope = write_registry(tmp_path, [
        {"document_id": d, "active": True, "retrieval_eligible": True,
         "duplicate_of_document_id": None} for d in ALL_DOCS])
    identity = write_identity(tmp_path, [
        (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00") for d in ALL_DOCS])
    return table, scope, identity


# ===========================================================================
# A. 부사 5개 — 값이 아니라 수식어다. 정상 상태 조회로 남아야 한다.
# ===========================================================================

class TestStatedValueAdverbsStayStatusQueries:
    """"사업분야가 {부사} 분류·명시된 공고" 는 '사업분야 기재됨' 질의다."""

    @pytest.mark.parametrize("adverb", ["따로", "별도로", "새로", "구체적으로", "명확히"])
    def test_adverb_is_not_read_as_a_value(self, adverb, value_world):
        from table_query import parse_selection

        table, scope, identity = value_world
        question = f"사업분야가 {adverb} 분류·명시된 공고를 알려줘"

        parse = parse_selection(question)
        assert parse.unresolved == [], question
        assert parse.fully_matched is True, question

        r = run_answer(question, table, identity=identity, registry_scope=scope)

        assert r.task_type == "select", question
        assert r.route == "추출테이블_문서선별", question
        assert r.abstained is False, question
        assert r.error_stage is None, question
        assert r.failure is None, question
        # 세 문서 모두 사업분야가 기재돼 있으므로 전부 나와야 한다.
        assert r.selected_document_ids == ALL_DOCS, question


# ===========================================================================
# B. 실제 값 5개 — 지원하지 않는 조건이라도 **버리지 않고 되묻는다**
# ===========================================================================

class TestUnsupportedValueConditionsAreNotDropped:
    """값 비교는 이번 범위 밖이다 — 조용히 버리면 다른 값의 문서가 답이 된다."""

    # (질문, 조건으로 남아야 하는 값) — 한 글자 값 3개가 이번 결함의 핵심이다.
    @pytest.mark.parametrize("question,value", [
        ("사업분야가 웹으로 명시된 공고를 찾아줘", "웹"),
        ("사업분야가 앱으로 명시된 공고를 찾아줘", "앱"),
        ("예산이 0으로 명시된 공고를 찾아줘", "0"),
        ("사업분야가 인공지능으로 명시된 공고를 찾아줘", "인공지능"),
        ("제출 방식이 방문접수로 기재된 공고를 찾아줘", "방문접수"),
    ])
    def test_value_condition_is_preserved_and_asked_back(
            self, question, value, value_world):
        from table_query import parse_selection

        table, scope, identity = value_world

        # 값 조건이 미해석 조건으로 남는다 — 어떤 값 때문인지도 알려준다.
        parse = parse_selection(question)
        assert parse.fully_matched is False, question
        assert any(f"'{value}'" in u for u in parse.unresolved), \
            f"{question} → {parse.unresolved}"

        r = run_answer(question, table, identity=identity, registry_scope=scope)

        assert r.task_type == "select", question
        assert r.abstained is True, question
        assert r.error_stage is None, question
        assert r.failure is None, question
        assert r.selected_document_ids == [], question
        # 지원하지 않는 조건이라는 사실을 사용자에게 알린다.
        assert r.structured_answer.get("clarification_needed") is True, question
        assert any(f"'{value}'" in u
                   for u in r.structured_answer.get("unrecognized", [])), \
            f"{question} → {r.structured_answer}"
