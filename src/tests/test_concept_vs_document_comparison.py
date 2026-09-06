#@title 개념 차이 설명과 문서 비교를 가른다 (§10)
#@markdown "차이"라는 낱말만으로 비교표 경로에 보내지 않는다. 문서를 두 건 이상
#@markdown 가리킨 질문의 기존 비교 경로는 그대로 유지한다.
"""수정 전에는 "공동수급과 하도급의 차이가 뭐야?"가 문서 비교로 가서
"비교하려면 문서가 두 건 이상 필요합니다"로 끝났다(실제 재현)."""
from __future__ import annotations

import pytest

from router import is_concept_comparison, route
from test_selection_fix import cfg, present, run_answer, write_identity, write_table

CFG = {"routing_method": "rule_based", "routing_fallback": "qa"}

CONCEPT = [
    "공동수급과 하도급의 차이가 뭐야?",
    "기술평가와 가격평가의 차이를 설명해줘.",
    "이 문서에서 공동이행방식과 분담이행방식의 차이를 설명해줘.",
    "이 사업의 기존 방식과 개선 방식의 차이가 뭐야?",
    "적격심사와 협상에 의한 계약의 차이를 알려줘.",
    "지역제한과 참가자격의 차이가 뭐야?",
    "일반경쟁과 제한경쟁 차이 알려줘",
]
DOCUMENT = [
    "RFP-000001과 RFP-000002의 예산을 비교해줘.",
    "RFP-000014와 RFP-000020의 사업기간을 비교해줘.",
    "두 사업의 사업기간을 비교해줘.",
    "서울시 사업과 부산시 사업의 예산 차이가 뭐야?",
    "가온재단 사업과 별빛공사 사업의 예산을 비교해줘.",
    "둘 중 예산이 큰 건 뭐야?",
    "우리 사업과 저쪽 사업 중 어느 쪽 예산이 커?",
    "예산을 비교해줘",
]


@pytest.mark.parametrize("question", CONCEPT)
def test_concept_difference_goes_to_qa(question):
    r = route(question, CFG)
    assert r.task_type == "qa", r
    assert r.matched_rule in ("concept_difference_explanation",
                              "single_document_concept_explanation")


@pytest.mark.parametrize("question", DOCUMENT)
def test_document_comparison_keeps_the_compare_route(question):
    assert route(question, CFG).task_type == "compare", question


@pytest.mark.parametrize("question", CONCEPT)
def test_helper_agrees_for_concepts(question):
    assert is_concept_comparison(question) or "이 문서에서" in question


@pytest.mark.parametrize("question", DOCUMENT)
def test_helper_rejects_document_pairs(question):
    assert is_concept_comparison(question) is False, question


@pytest.fixture
def two_docs(tmp_path):
    table = write_table(tmp_path, {
        "RFP-000901": {"예산": present("1억원"), "컨소시엄 요건": present("공동수급 허용")},
        "RFP-000902": {"예산": present("2억원")},
    })
    identity = write_identity(tmp_path, [
        ("RFP-000901", "가온재단", "하늘 관측망 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000902", "별빛공사", "바다 관측망 구축 사업", "2025-01-01 17:00:00"),
    ])
    return table, identity


def test_concept_question_does_not_ask_for_two_documents(two_docs):
    """개념 질문에 '문서를 두 건 알려달라'고 되묻지 않는다."""
    table, identity = two_docs
    r = run_answer("공동수급과 하도급의 차이가 뭐야?", table, identity=identity)
    assert "문서가 두 건" not in (r.text or "")
    assert r.task_type == "qa"


def test_real_document_comparison_still_builds_a_table(two_docs):
    table, identity = two_docs
    r = run_answer("RFP-000901과 RFP-000902의 예산을 비교해줘.", table, identity=identity)
    assert r.abstained is False
    assert sorted(r.selected_document_ids) == ["RFP-000901", "RFP-000902"]
    assert "1억원" in r.text and "2억원" in r.text
