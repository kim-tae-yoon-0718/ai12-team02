"""company_match 실험(느슨한 자격증 대조 + 컨소시엄 요건) 단위 테스트.

본문 company_match.py는 안 건드리는지도 같이 확인한다(제일 중요한 계약).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "company_match"))

from loose_match_experiment import (  # noqa: E402
    evaluate_certification_loose, evaluate_consortium, loose_keyword_match,
    load_consortium_needed, save_consortium_needed,
)
from company_match import CompanyProfile, save_profile  # noqa: E402


class TestLooseKeywordMatch:
    def test_finds_token_even_when_full_phrase_differs(self):
        text = "소프트웨어사업자(컴퓨터관련서비스사업, 업종코드 : 1468)로 신고한 업체"
        hit = loose_keyword_match(text, ["소프트웨어사업자 신고필증"])
        assert hit is not None
        candidate, token = hit
        assert candidate == "소프트웨어사업자 신고필증"
        assert token == "소프트웨어사업자"

    def test_generic_token_alone_does_not_match(self):
        # '신고필증'을 뺀 나머지 낱말('신고')은 GENERIC_TOKENS라 매칭 근거가 안 된다
        text = "사업자등록증을 제출한다"
        assert loose_keyword_match(text, ["신고"]) is None

    def test_no_overlap_returns_none(self):
        assert loose_keyword_match("전혀 관련 없는 문장입니다", ["소프트웨어사업자 신고필증"]) is None


class TestCertificationLooseEvaluator:
    def test_field_absent_is_neutral(self):
        status, _ = evaluate_certification_loose(
            CompanyProfile("A", certifications=["소프트웨어사업자 신고필증"]),
            {"status": "field_absent", "answer_normalized": ""})
        assert status == "정보없음"

    def test_loose_match_passes_where_exact_match_would_fail(self):
        row = {"status": "value_present",
              "answer_normalized": "소프트웨어사업자(컴퓨터관련서비스사업)로 신고한 업체"}
        status, reason = evaluate_certification_loose(
            CompanyProfile("A", certifications=["소프트웨어사업자 신고필증"]), row)
        assert status == "충족"
        assert "소프트웨어사업자" in reason


class TestConsortiumEvaluator:
    def test_field_absent_is_neutral(self):
        status, _ = evaluate_consortium(True, {"status": "field_absent", "answer_normalized": ""})
        assert status == "정보없음"

    def test_deny_and_company_needs_consortium_is_rejected(self):
        row = {"status": "value_present", "answer_normalized": "공동수급을 허용하지 않음"}
        status, reason = evaluate_consortium(True, row)
        assert status == "부적합"

    def test_deny_and_company_does_not_need_consortium_passes(self):
        row = {"status": "value_present", "answer_normalized": "공동수급을 허용하지 않음"}
        status, _ = evaluate_consortium(False, row)
        assert status == "충족"

    def test_allow_passes_regardless(self):
        row = {"status": "value_present", "answer_normalized": "공동수급(공동이행방식)을 허용함"}
        status, _ = evaluate_consortium(True, row)
        assert status == "충족"

    def test_conditional_allow_needs_verification_not_pass(self):
        """조건부 허용은 조건 충족 여부를 대조한 적이 없으므로 '충족'으로
        단정하면 조건을 못 채우는 회사도 적합으로 보인다(회귀: 코드리뷰 지적)."""
        row = {"status": "value_present",
              "answer_normalized": "공동이행 방식은 허용하나 분담이행 방식은 불가하다"}
        status, _ = evaluate_consortium(True, row)
        assert status == "확인 필요"

    def test_deny_with_unknown_solo_capability_needs_verification(self):
        """consortium_needed가 등록 안 된(None) 상태는 '회사가 단독 참여
        가능한지 모른다'는 뜻이지 '가능하다'가 아니다(회귀: 코드리뷰 지적)."""
        row = {"status": "value_present", "answer_normalized": "공동수급을 허용하지 않음"}
        status, _ = evaluate_consortium(None, row)
        assert status == "확인 필요"


class TestConsortiumProfileStorage:
    def test_save_then_load_roundtrip(self, tmp_path):
        save_profile(tmp_path, CompanyProfile(company_name="테스트기업", region="서울"))
        save_consortium_needed(tmp_path, "테스트기업", True)
        assert load_consortium_needed(tmp_path, "테스트기업") is True

    def test_base_profile_still_loads_fine_after_experimental_key_added(self, tmp_path):
        """본문 CompanyProfile.from_json은 모르는 키(consortium_needed)를 무시해야 한다."""
        from company_match import load_profile
        save_profile(tmp_path, CompanyProfile(company_name="테스트기업", region="서울"))
        save_consortium_needed(tmp_path, "테스트기업", True)
        profile = load_profile(tmp_path, "테스트기업")
        assert profile.region == "서울"
