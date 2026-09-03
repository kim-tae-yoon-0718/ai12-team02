#@title 선별 질문의 의미 보존을 검증하는 독립 시험
#@markdown 합성 문서로 숫자 경계·부정 범위·미지원 조건 보존·문서 범위를 검사하며 API는 호출하지 않습니다.
"""Reviewer's independent tests: expected IDs come from arithmetic/explicit facts.

This file does not use evaluation-set question IDs, answers, or official documents.
It deliberately calls the real answer() path through the existing test helpers.
"""
from __future__ import annotations

import operator

import pytest

import test_selection_fix as h


AMOUNTS = {
    "RFP-000961": 0,
    "RFP-000962": 50_000_000,
    "RFP-000963": 100_000_000,
    "RFP-000964": 200_000_000,
    "RFP-000965": 500_000_000,
    "RFP-000966": 600_000_000,
}
UNKNOWN_IDS = {"RFP-000967", "RFP-000968", "RFP-000969", "RFP-000970"}


@pytest.fixture
def invariant_world(tmp_path):
    specs = {
        doc_id: {
            "예산": h.present(f"{amount:,}원"),
            "지역제한": h.present("부산 소재 업체만 입찰 참여 가능"),
            "사업분야": h.present("건물 청소"),
            "제출 방식": h.present("전자접수"),
        }
        for doc_id, amount in AMOUNTS.items()
    }
    specs.update({
        "RFP-000967": {"예산": h.absent()},
        "RFP-000968": {"예산": h.external()},
        "RFP-000969": {"예산": h.conflict("1억 원", "6억 원")},
        "RFP-000970": {"예산": h.not_disclosed()},
    })
    table = h.write_table(tmp_path, specs)
    scope = h.write_registry(tmp_path, [
        {"document_id": doc_id, "active": True, "retrieval_eligible": True,
         "duplicate_of_document_id": None}
        for doc_id in specs
    ])
    identity = h.write_identity(tmp_path, [
        (doc_id, f"시험기관{doc_id[-3:]}", f"시험시설{doc_id[-3:]} 정비 용역",
         "2025-01-01 17:00:00")
        for doc_id in specs
    ])
    return table, scope, identity


def execute(question, world):
    table, scope, identity = world
    result = h.run_answer(question, table, identity=identity, registry_scope=scope)
    assert not result.failure, (question, result.failure)
    return result


# Single numeric predicates include exact threshold equality, a zero amount,
# comma-bearing amounts, and unobservable amounts that must remain unknown.
SINGLE_PREDICATES = [
    ("예산 1억 이상인", operator.ge, 100_000_000),
    ("예산 1억 초과인", operator.gt, 100_000_000),
    ("예산 5억 이하인", operator.le, 500_000_000),
    ("예산 5억 미만인", operator.lt, 500_000_000),
    ("예산 1억 이상이 아닌", operator.lt, 100_000_000),
    ("예산 1억 초과가 아닌", operator.le, 100_000_000),
    ("예산 5억 이하가 아닌", operator.gt, 500_000_000),
    ("예산 5억 미만이 아닌", operator.ge, 500_000_000),
    ("예산 100,000,000원 이상인", operator.ge, 100_000_000),
    ("예산 500,000,000원 이하가 아닌", operator.gt, 500_000_000),
]


@pytest.mark.parametrize("phrase,compare,threshold", SINGLE_PREDICATES)
def test_numeric_boundary_and_unknowns(phrase, compare, threshold, invariant_world):
    result = execute(f"{phrase} 공고를 찾아줘", invariant_world)
    expected = {doc_id for doc_id, amount in AMOUNTS.items() if compare(amount, threshold)}
    assert set(result.selected_document_ids) == expected
    assert not UNKNOWN_IDS.intersection(result.selected_document_ids)
    assert result.abstained is False


# Repeated comparisons deliberately exercise the same regexp more than once.
# Every pair has an independent scalar predicate, so parser output is not the oracle.
RANGE_PREDICATES = [
    ("예산은 5억 이상은 아니지만", "예산은 1억 이상인",
     lambda amount: 100_000_000 <= amount < 500_000_000),
    ("예산은 1억 이하는 아니지만", "예산은 5억 이하인",
     lambda amount: 100_000_000 < amount <= 500_000_000),
    ("예산은 5억 초과는 아니지만", "예산은 1억 초과인",
     lambda amount: 100_000_000 < amount <= 500_000_000),
    ("예산은 1억 미만은 아니지만", "예산은 5억 미만인",
     lambda amount: 100_000_000 <= amount < 500_000_000),
]


@pytest.mark.parametrize("first,second,predicate", RANGE_PREDICATES)
@pytest.mark.parametrize("comma", [" ", ", "])
def test_repeated_operator_does_not_inherit_previous_negation(
        first, second, predicate, comma, invariant_world):
    question = f"{first}{comma}{second} 공고를 찾아줘"
    result = execute(question, invariant_world)
    assert set(result.selected_document_ids) == {
        doc_id for doc_id, amount in AMOUNTS.items() if predicate(amount)
    }, question
    assert not UNKNOWN_IDS.intersection(result.selected_document_ids)


@pytest.mark.parametrize("connector", ["이고 ", "이고, ", "이며, "])
@pytest.mark.parametrize("reverse", [False, True])
def test_and_order_is_commutative(connector, reverse, invariant_world):
    parts = ["예산은 1억 이상", "예산은 5억 미만"]
    if reverse:
        parts.reverse()
    question = connector.join(parts) + "인 공고를 찾아줘"
    result = execute(question, invariant_world)
    assert set(result.selected_document_ids) == {"RFP-000963", "RFP-000964"}


UNSUPPORTED_REQUIREMENTS = [
    "사업분야가 인공지능이어야 하고",
    "사업분야가 인공지능이어야 하며",
    "사업분야는 인공지능이어야 하고",
    "제출 방식이 방문접수여야 하고",
    "제출 방식은 방문접수여야 하며",
    "지역제한이 서울이어야 하고",
    "서울 소재 업체만 참여 가능하고",
    "서울 소재 업체만 참여 가능하며",
]


@pytest.mark.parametrize("unsupported", UNSUPPORTED_REQUIREMENTS)
@pytest.mark.parametrize("comma", [" ", ", "])
def test_unsupported_requirement_must_not_become_background(
        unsupported, comma, invariant_world):
    question = f"{unsupported}{comma}예산 5억 이상인 공고를 찾아줘"
    result = execute(question, invariant_world)
    assert result.abstained is True, question
    assert result.selected_document_ids == [], question


@pytest.mark.parametrize("background", [
    "사업분야가 궁금해서 그런데,",
    "우리 회사 본사 소재지가 부산이라,",
    "우리 회사가 인공지능 사업을 하고 있어서,",
])
def test_actual_background_does_not_destroy_supported_query(background, invariant_world):
    result = execute(f"{background} 예산 5억 이상인 공고를 찾아줘", invariant_world)
    assert set(result.selected_document_ids) == {"RFP-000965", "RFP-000966"}
    assert result.abstained is False


@pytest.mark.parametrize("question", [
    "사업분야가 인공지능이고 예산 5억 이상인 공고를 찾아줘",
    "제출 방식은 방문접수이고 예산 5억 이상인 공고를 찾아줘",
    "예산 5억 이상이고 사업분야가 인공지능인 공고를 찾아줘",
    "예산 5억 이상이고 제출 방식은 방문접수인 공고를 찾아줘",
    "예산 5억 이상 또는 1억 이하인 공고를 찾아줘",
])
def test_other_unsupported_conditions_are_never_silently_dropped(question, invariant_world):
    result = execute(question, invariant_world)
    assert result.abstained is True
    assert result.selected_document_ids == []


@pytest.mark.parametrize("negative", [
    "공동수급은 허용되지 않습니다.",
    "공동수급은 가능하지 않습니다.",
    "공동수급은 허용하지 아니합니다.",
    "공동수급을 허용할 수 없습니다.",
])
def test_source_negation_cannot_appear_in_allowed_results(negative, tmp_path):
    table = h.write_table(tmp_path, {
        "RFP-000981": {"컨소시엄 요건": h.present(negative)},
        "RFP-000982": {"컨소시엄 요건": h.present("공동수급은 허용되며 하도급은 불가합니다.")},
    })
    allowed = h.run_answer("공동수급이 허용되는 공고를 찾아줘", table)
    forbidden = h.run_answer("공동수급이 금지된 공고를 찾아줘", table)
    assert not allowed.failure and not forbidden.failure
    assert allowed.selected_document_ids == ["RFP-000982"]
    assert forbidden.selected_document_ids == ["RFP-000981"]


@pytest.mark.parametrize("question", [
    "미등록기관 자동화시스템 고도화 용역, 사업 예산을 알려줄 수 있어?",
    "미등록기관 자동화시스템 고도화 용역의 사업 예산을 알려줄 수 있어?",
    "미등록기관 자동화시스템 고도화 용역 사업 예산을 알려줄 수 있어?",
    "미등록기관 자동화시스템 고도화 사업의 필수 제출 서류 목록을 알려줘",
])
def test_unknown_single_project_never_expands_to_all_documents(question, invariant_world):
    result = execute(question, invariant_world)
    assert result.abstained is True
    assert result.selected_document_ids == []


@pytest.mark.parametrize("question", [
    "전체 공고에서 예산이 기재된 공고를 찾아줘",
    "이 사업과 별개로 전체 공고에서 예산이 기재된 사업을 찾아줘",
])
def test_explicit_global_scope_keeps_selecting(question, invariant_world):
    result = execute(question, invariant_world)
    assert result.task_type == "select"
    assert result.abstained is False
    assert set(result.selected_document_ids) == set(AMOUNTS)


STATUS_BUILDERS = [
    lambda: h.present("기재된 시험 값"),
    h.absent,
    h.external,
    h.not_disclosed,
    lambda: h.conflict("값 A", "값 B"),
    h.extraction_failed,
    lambda: {"status": "review_required", "answer_raw": "", "answer_normalized": ""},
]
STATUS_REQUESTS = [
    ("항목이 기재된", {0}),
    ("항목이 미기재된", {1}),
    ("항목은 외부 문서를 참조하는", {2}),
    ("항목이 비공개인", {3}),
    ("내용이 충돌하는", {4}),
    ("항목이 판단 보류인", {5, 6}),
]


@pytest.fixture
def rotated_status_world(tmp_path):
    # Different fields rotate the status-to-document mapping. A fixed "first doc"
    # answer cannot accidentally satisfy the entire 12-field status matrix.
    specs = {
        f"RFP-{index + 911:06d}": {
            field: STATUS_BUILDERS[(index + offset) % len(STATUS_BUILDERS)]()
            for offset, field in enumerate(h.OFFICIAL_FIELDS)
        }
        for index in range(len(STATUS_BUILDERS))
    }
    table = h.write_table(tmp_path, specs)
    scope = h.write_registry(tmp_path, [
        {"document_id": doc_id, "active": True, "retrieval_eligible": True}
        for doc_id in specs
    ])
    identity = h.write_identity(tmp_path, [
        (doc_id, f"교차기관{index}", f"교차문서{index}", "2025-01-01 17:00:00")
        for index, doc_id in enumerate(specs)
    ])
    return table, scope, identity


@pytest.mark.parametrize("offset,field", list(enumerate(h.OFFICIAL_FIELDS)))
@pytest.mark.parametrize("status_phrase,status_indices", STATUS_REQUESTS)
def test_twelve_fields_keep_six_distinct_status_groups(
        offset, field, status_phrase, status_indices, rotated_status_world):
    question = f"{field} {status_phrase} 공고를 찾아줘"
    result = execute(question, rotated_status_world)
    expected = {
        f"RFP-{index + 911:06d}"
        for index in range(len(STATUS_BUILDERS))
        if (index + offset) % len(STATUS_BUILDERS) in status_indices
    }
    assert set(result.selected_document_ids) == expected, question
    assert result.abstained is False
    assert h._diag(result)["undetermined_count"] == 0


@pytest.mark.parametrize("query_amount,threshold", [
    ("8천만원", 80_000_000),
    ("1억 5천만원", 150_000_000),
    ("49,500만원", 495_000_000),
])
@pytest.mark.parametrize("phrase,compare", [
    ("이상인", operator.ge),
    ("미만인", operator.lt),
    ("이하인", operator.le),
    ("초과인", operator.gt),
    ("이상이 아닌", operator.lt),
    ("이하가 아닌", operator.gt),
])
def test_composite_and_comma_amounts_match_arithmetic_oracle(
        query_amount, threshold, phrase, compare, tmp_path):
    amounts = {"RFP-000921": threshold - 1, "RFP-000922": threshold,
               "RFP-000923": threshold + 1}
    table = h.write_table(tmp_path, {
        doc_id: {"예산": h.present(f"{amount:,}원")}
        for doc_id, amount in amounts.items()
    })
    result = h.run_answer(f"예산 {query_amount} {phrase} 공고를 찾아줘", table)
    assert not result.failure
    assert set(result.selected_document_ids) == {
        doc_id for doc_id, amount in amounts.items() if compare(amount, threshold)
    }


def test_registry_and_deadline_filters_preserve_full_eligible_set(tmp_path):
    # A set larger than 20 proves that display/list defaults do not cut answers.
    ids = [f"RFP-{index + 800:06d}" for index in range(32)]
    table = h.write_table(tmp_path, {
        doc_id: {"예산": h.present("6억원")}
        for doc_id in ids
    })
    scope = h.write_registry(tmp_path, [
        {"document_id": doc_id, "active": index != 0,
         "retrieval_eligible": index != 1,
         "duplicate_of_document_id": ids[2] if index == 1 else None}
        for index, doc_id in enumerate(ids)
    ])
    deadlines = {ids[2]: "2024-05-31 23:59:59", ids[3]: "2024-06-01 00:00:00",
                 ids[4]: "", ids[5]: "2024-06-01 00:00:01"}
    identity = h.write_identity(tmp_path, [
        (doc_id, f"집합기관{index}", f"집합사업{index}",
         deadlines.get(doc_id, "2025-01-01 17:00:00"))
        for index, doc_id in enumerate(ids)
    ])
    result = execute("예산 5억 이상인 공고를 찾아줘", (table, scope, identity))
    expected = set(ids[3:])
    assert set(result.selected_document_ids) == expected
    assert len(result.selected_document_ids) == 29
    assert len(result.structured_answer) == 29
    assert ids[4] in result.text and "미상" in result.text
    assert ids[0] not in result.selected_document_ids
    assert ids[1] not in result.selected_document_ids
    assert ids[2] not in result.selected_document_ids
    assert h._diag(result)["dropped_by_deadline_filter"] == 1


@pytest.mark.parametrize("mode,expected_class,abstained,confirmed,unknown", [
    ("known_nonmatch", "empty", False, 0, 0),
    ("unknown_only", "undetermined_only", True, 0, 1),
    ("partial", "partial_undetermined", False, 1, 1),
])
def test_empty_unknown_and_partial_remain_different_results(
        mode, expected_class, abstained, confirmed, unknown, tmp_path):
    specs = {"RFP-000931": {"예산": h.present("1억원")}}
    if mode in {"unknown_only", "partial"}:
        specs["RFP-000932"] = {"예산": h.absent()}
    if mode == "partial":
        specs["RFP-000933"] = {"예산": h.present("6억원")}
    result = h.run_answer("예산 5억 이상인 공고를 찾아줘", h.write_table(tmp_path, specs))
    assert not result.failure
    assert result.abstained is abstained
    assert h._diag(result)["outcome_class"] == expected_class
    assert h._diag(result)["confirmed_count"] == confirmed
    assert h._diag(result)["undetermined_count"] == unknown
    if unknown:
        assert "조건에 맞는 공고가 없습니다" not in result.text.split("\n")[0]


@pytest.mark.parametrize("question", [
    "사업분야가 인공지능으로 명시된 공고 중 예산 5억 이상인 공고를 찾아줘",
    "제출 방식이 방문접수로 기재된 공고 중 예산 5억 이상인 공고를 찾아줘",
    "사업분야가 인공지능이어야 하고 명시되어 있어야 하며, 예산 5억 이상인 공고를 찾아줘",
])
def test_recognized_status_cannot_hide_unsupported_field_value(question, invariant_world):
    # "명시/기재" describes how a particular value appears; it does not erase
    # the AI/visit-submission requirement. Current scope may only ask back.
    result = execute(question, invariant_world)
    assert result.abstained is True, question
    assert result.selected_document_ids == [], question


@pytest.mark.parametrize("question", [
    "사업분야가 구체적으로 명시된 공고 중 예산 5억 이상인 공고를 찾아줘",
    "제출 방식이 명확하게 기재된 공고 중 예산 5억 이상인 공고를 찾아줘",
])
def test_status_description_adverb_is_not_a_specific_field_value(question, invariant_world):
    result = execute(question, invariant_world)
    assert result.abstained is False
    assert set(result.selected_document_ids) == {"RFP-000965", "RFP-000966"}


def test_background_without_comma_is_not_an_unknown_project_title(invariant_world):
    question = "입찰 참가를 준비 중인데 사업 예산이 기재된 공고 찾아줘"
    result = execute(question, invariant_world)
    assert result.task_type == "select"
    assert result.abstained is False
    assert set(result.selected_document_ids) == set(AMOUNTS)
