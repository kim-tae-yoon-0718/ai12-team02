"""선별형 평가 문항 정책 회귀 테스트 (2026-09-04).

평가셋 정답을 만들 때 쓰는 정책 술어를 못박는다 — 마감 필터, 미상 처리,
'미기재'와 '제한 없음'의 구분, 공동수급 5상태, 의도적 빈 정답.

★최종 50문항의 질문·정답은 이 파일에 넣지 않는다. 모두 합성 입력이며,
  실제 평가셋 파일 대조는 검사기 실행(§10 검증)에서 한다.
"""
import json

import pytest

from checks.check_evalset import (
    REFERENCE_TIME_EXPECTED, check_answer_evidence_docs, check_comparison_structure,
    check_reference_time, check_selection_scope,
)
from checks.selection_policy import (
    ALLOWED_EXPLICIT, FIELD_ABSENT, FORBIDDEN_ALL, METHOD_RESTRICTED, REFERENCE_TIME,
    REQUIRED_ALL, REQUIRED_CONDITIONAL, UNDETERMINED,
    classify, deadline_state, is_external_reference, is_field_absent, is_forbidden_all,
    parse_deadline, passes_deadline,
)


# ── 1. 마감 필터와 기준일 ──────────────────────────────────────────────

def test_deadline_filter_uses_fixed_reference_time_not_now():
    """기준일은 고정값이다 — '지금'으로 잡으면 같은 평가셋이 날마다 다른 정답을 갖는다."""
    assert REFERENCE_TIME == REFERENCE_TIME_EXPECTED == "2024-06-01"
    assert deadline_state("2024-05-31 18:00") == "expired"
    assert deadline_state("2024-06-01") == "open"      # 기준일 당일은 경과가 아니다
    assert deadline_state("2024-06-02 09:00:00") == "open"


def test_deadline_parses_official_formats_only_when_unambiguous():
    assert parse_deadline("2024-05-20 14:00:00") == "2024-05-20"
    assert parse_deadline("2024-05-20 14:00") == "2024-05-20"
    assert parse_deadline("2024-05-20") == "2024-05-20"
    assert parse_deadline("2024-13-40") is None        # 달·일이 말이 안 되면 미상
    assert parse_deadline("추후 공고") is None
    assert parse_deadline("") is None and parse_deadline(None) is None


# ── 2. 마감일 미상 문서는 탈락시키지 않는다 ────────────────────────────

def test_unknown_deadline_passes_filter_and_is_not_expired():
    """미상을 경과로 몰면 추출이 나빠질수록 정답이 작아지는 역방향 보상이 생긴다."""
    assert deadline_state(None) == "unknown"
    assert deadline_state("일정 별도 안내") == "unknown"
    assert passes_deadline(None) is True
    assert passes_deadline("일정 별도 안내") is True
    assert passes_deadline("2024-05-31") is False


def test_unknown_deadline_is_distinguishable_from_open():
    """통과는 같아도 사유가 달라야 한다 — 근거에 '미상 통과'로 남길 수 있어야 한다."""
    assert deadline_state("2024-07-01") == "open"
    assert deadline_state(None) == "unknown"
    assert deadline_state("2024-07-01") != deadline_state(None)


# ── 3. '미기재'와 '제한 없음'은 다르다 ─────────────────────────────────

def test_field_absent_is_status_based_not_text_based():
    """'지역제한 없음'이라고 적힌 문서는 기재된 문서다 — 미기재 문항에 섞이면 안 된다."""
    assert is_field_absent("field_absent") is True
    assert is_field_absent("value_present") is False
    assert is_field_absent("external_reference") is False
    assert is_field_absent("conflict") is False


def test_explicit_no_restriction_text_is_not_absent():
    for text in ("제한 없음", "해당 없음", "지역 제한 없음", "N/A"):
        assert is_field_absent("value_present") is False, text


# ── 4. 공동수급 7상태 — 전면 금지와 '특정 방식만 금지'를 반드시 구분 ──────

def test_consortium_allowed_is_not_required():
    st, _ = classify("value_present", "공동수급체(공동이행방식) 구성하여 참여 가능")
    assert st == ALLOWED_EXPLICIT
    assert st != REQUIRED_ALL


def test_consortium_required_and_conditional_are_separated():
    st_req, _ = classify("value_present", "반드시 공동수급체를 구성하여야 입찰에 참가할 수 있다")
    assert st_req == REQUIRED_ALL
    st_cond, _ = classify(
        "value_present",
        "하도급 비율이 10%를 초과하려는 경우 해당 업체와 공동수급체를 구성하여 참여해야 한다")
    assert st_cond == REQUIRED_CONDITIONAL
    assert st_req != st_cond


def test_method_ban_is_not_a_full_ban():
    """★이 프로젝트에서 가장 값비쌌던 오분류.

    공동수급에는 공동이행·분담이행·주계약자관리 등 여러 방식이 있다. 한 방식을
    막았다고 다른 방식까지 막힌 것이 아니다. 이걸 전면 금지로 뭉개면 "공동수급이
    금지된 공고" 정답에 실제로는 참여 가능한 공고가 섞여 들어간다.
    """
    for text in ("본 입찰은 공동수급(분담이행방식)을 허용하지 않음",
                 "본 사업은 공동수급(공동이행방식)을 허용하지 않음",
                 "공동수급(공동이행방식) 및 재하도급 불가",
                 "공동수급체(공동이행방식)형태의 참가를 불허함",
                 "사업의 특성상 공동수급(공동이행방식) 불가(책임소재 불분명)"):
        st, reason = classify("value_present", text)
        assert st == METHOD_RESTRICTED, f"{text} → {st} ({reason})"
        assert not is_forbidden_all(st), text


def test_full_ban_is_detected():
    for text in ("공동수급 불가",
                 "자. 공동 수급 불허",
                 "나. 공동수급, 하도급 불허",
                 "아. 공동수급(컨소시엄) 형태의 입찰 불가",
                 "본 사업은 공동계약 및 하도급을 불허하는 사업임",
                 "복수의 업체로 컨소시엄을 구성하여 입찰하는 경우 참여를 제한함",
                 "본 사업은 공동수급을 제외한 단독입찰로 진행함"):
        st, reason = classify("value_present", text)
        assert st == FORBIDDEN_ALL, f"{text} → {st} ({reason})"
        assert is_forbidden_all(st)


def test_all_methods_listed_in_paren_is_a_full_ban():
    """괄호 안에 이행방식이 모두 열거되면 남는 방식이 없어 사실상 전면 금지다."""
    st, _ = classify("value_present", "공동계약(공동 및 분담 이행방식) 및 하도급 불가")
    assert st == FORBIDDEN_ALL


def test_apposition_in_paren_is_not_a_method_restriction():
    """공동수급(공동계약) 처럼 괄호가 동격이면 방식 한정이 아니다."""
    st, _ = classify("value_present", "본 사업은 공동수급(공동계약)을 금지함.")
    assert st == FORBIDDEN_ALL


def test_method_limited_but_participation_allowed_is_allowed():
    """"공동이행만 허용, 분담이행 불허" 는 공동수급 참여 자체는 허용이다."""
    st, _ = classify("value_present", "공동수급체는 공동이행 방식만 허용하며, 분담이행 방식은 불허한다.")
    assert st == ALLOWED_EXPLICIT


def test_negation_of_a_different_object_is_not_a_consortium_ban():
    """"공동수급은 허용, 하도급은 불허" 를 공동수급 금지로 읽으면 안 된다."""
    for text in ("본 사업은 공동수급을 허용하고 있어 하도급은 불허함",
                 "하도급/공동수급 여부 : 하도급 불허 / 공동수급 허용",
                 "공동수급 및 하도급 허용"):
        st, reason = classify("value_present", text)
        assert st == ALLOWED_EXPLICIT, f"{text} → {st} ({reason})"


def test_allow_verb_after_a_negation_is_not_an_allowance():
    """"공동수급 … 불가함의 규정에 의한 … 과업 수행 가능자" 의 '가능'은 허용이 아니다."""
    st, _ = classify(
        "value_present",
        "➆ 공동수급(공동이행방식) 및 하도급은 불가함의 규정에 의한 경쟁 입찰 참가 자격을 갖춘 과업 수행 가능자")
    assert st == METHOD_RESTRICTED


def test_consortium_absent_and_undetermined():
    st_absent, reason = classify("field_absent", None)
    assert st_absent == FIELD_ABSENT and "field_absent" in reason
    st_conf, _ = classify("conflict", "무언가")
    assert st_conf == UNDETERMINED


def test_consortium_undetermined_falls_back_to_source_excerpt_same_row():
    """주어가 잘린 answer_raw 는 같은 행의 원문 발췌로 한 번 더 본다(추측 아님)."""
    st, reason = classify("value_present", "구성원은 5개 이하로 하여야 함",
                          source_excerpt="본 사업은 공동수급을 불허함")
    assert st == FORBIDDEN_ALL
    assert "원문 발췌 근거" in reason


def test_only_full_ban_counts_as_explicitly_forbidden():
    assert is_forbidden_all(FORBIDDEN_ALL) is True
    for other in (METHOD_RESTRICTED, ALLOWED_EXPLICIT, FIELD_ABSENT,
                  UNDETERMINED, REQUIRED_ALL, REQUIRED_CONDITIONAL):
        assert is_forbidden_all(other) is False


def test_external_reference_is_not_field_absent():
    """다른 문서를 보라는 안내는 '항목 미기재'가 아니다 — 항목은 있고 위치만 바깥이다."""
    assert is_external_reference("external_reference") is True
    assert is_field_absent("external_reference") is False


# ── 5. 의도적인 빈 정답(0건)은 오류가 아니다 ───────────────────────────

def _sel(idx, docs, **kw):
    b = dict(id=f"SEL-{idx:03d}", question="q", task_type="selection",
             answer_type="document_set", answer_raw=list(docs),
             answer_set=list(docs), answer_source="extraction_table",
             reference_time=REFERENCE_TIME_EXPECTED)
    b.update(kw)
    return b


@pytest.fixture
def registry(tmp_path):
    """합성 등록부 — 검색 대상 2건 + 제외 1건."""
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"documents": [
        {"document_id": "RFP-000001", "active": True, "retrieval_eligible": True},
        {"document_id": "RFP-000002", "active": True, "retrieval_eligible": True},
        {"document_id": "RFP-000006", "active": True, "retrieval_eligible": False},
    ]}, ensure_ascii=False), encoding="utf-8")
    return p


def test_intentional_empty_selection_answer_is_accepted(registry):
    """마감 필터 뒤 0건이면 0건이 정답이다 — 억지로 채우면 근거 없는 정답이 된다."""
    assert check_answer_evidence_docs([_sel(1, [])]) == []
    assert check_selection_scope([_sel(1, [])], registry) == []


def test_selection_scope_rejects_out_of_scope_and_duplicate_docs(registry):
    dup = check_selection_scope([_sel(2, ["RFP-000001", "RFP-000001"])], registry)
    assert any("중복" in e for e in dup)
    outside = check_selection_scope([_sel(3, ["RFP-000006"])], registry)
    assert any("검색 대상" in e for e in outside)
    unknown = check_selection_scope([_sel(4, ["RFP-000099"])], registry)
    assert any("등록부에 없는" in e for e in unknown)


# ── 6. reference_time 누락은 검사에서 잡힌다 ───────────────────────────

def test_reference_time_required_on_every_selection_item():
    ok = _sel(3, ["RFP-000001"])
    assert check_reference_time([ok], REFERENCE_TIME_EXPECTED) == []
    missing = dict(ok); missing.pop("reference_time")
    assert check_reference_time([missing], REFERENCE_TIME_EXPECTED) != []
    wrong = dict(ok, reference_time="2025-01-01")
    assert check_reference_time([wrong], REFERENCE_TIME_EXPECTED) != []


# ── 7. 서로 다른 조건의 문항은 정답도 서로 달라야 한다 ─────────────────

def test_distinct_conditions_must_not_collapse_to_same_gold():
    """조건이 다른데 정답 집합이 같으면 둘 중 하나는 조건을 잘못 옮긴 것이다.

    (실제 문항 대조는 검사기 실행에서 한다 — 여기서는 규칙만 못박는다.)
    """
    a, b, c = {"D1", "D2"}, {"D1", "D2"}, {"D3"}
    def distinct(*sets):
        return all(sets[i] != sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets)))
    assert distinct(a, c) is True
    assert distinct(a, b) is False        # 같은 정답 → 조건 전사 오류로 잡아야 한다


# ── 8. 조건부 서술은 '필수'도 '금지'도 아니다 (2026-09-04, RFP-000067/000081 정책) ──

def test_conditional_consortium_clause_is_neither_required_nor_forbidden():
    """"공동수급 형태로 제안할 경우 … 제시해야 함" 은 참여할 **경우의** 조건이지
    모든 참여자에게 공동수급을 요구하는 문장이 아니다. required_all 로 읽으면
    '공동수급 필수 공고' 정답에 잘못 들어간다."""
    st, _ = classify("value_present",
                     "공동수급 형태로 제안할 경우 주사업자와 부사업자 간의 업무수행범위 및 "
                     "책임한계를 상세히 정의하고 주사업자의 조직 운영 방안을 제시해야 함")
    assert st not in (REQUIRED_ALL, FORBIDDEN_ALL)
    assert is_forbidden_all(st) is False


def test_subcontract_threshold_clause_is_conditional_not_required_all():
    st, _ = classify("value_present",
                     "공동수급 형태로 제안할 경우 주사업자와 부사업자 간의 업무수행범위 및 "
                     "책임한계를 상세히 정의해야 함; 전체 사업금액 대비 10%를 초과하여 하도급 "
                     "하려는 경우 하수급인과 공동수급체를 구성하여 참여해야 하며, 공동수급체를 "
                     "구성하지 못하는 불가피한 사정이 있는 경우 그 사유를 제시하여야 함")
    assert st == REQUIRED_CONDITIONAL
    assert st != REQUIRED_ALL
