"""기업 프로필 저장 + 매칭 도구 단위 테스트 — 순수 로직만(공식 자료 안 건드림)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "company_match"))

from company_match import (  # noqa: E402
    CompanyProfile, evaluate_business_field, evaluate_certification, evaluate_region,
    find_region_in_text, load_profile, overall_verdict, region_matches, save_profile,
)


class TestRegionMatching:
    def test_finds_known_region_in_free_text(self):
        text = "주된 영업소의 소재지가 부산광역시인 기업"
        assert find_region_in_text(text) == "부산광역시"

    def test_alias_matches_canonical(self):
        assert region_matches("부산", "부산광역시") is True

    def test_different_region_does_not_match(self):
        assert region_matches("서울", "부산광역시") is False

    def test_no_region_name_returns_none(self):
        assert find_region_in_text("입찰공고일 전일부터 계약체결일까지") is None


class TestFieldEvaluators:
    def test_region_field_absent_is_neutral_not_confirmed(self):
        """명시가 없는 건 실제로 대조해서 확인된 게 아니므로 '정보없음'이어야
        한다 — '충족'으로 두면 조건 자체가 없는 공고까지 '실제로 맞다고
        확인됨'처럼 보인다(다른 필드 평가 함수들과 같은 규칙)."""
        status, reason = evaluate_region(CompanyProfile("A", region="서울"),
                                          {"status": "field_absent", "answer_normalized": ""})
        assert status == "정보없음"

    def test_region_mismatch_is_rejected(self):
        row = {"status": "value_present",
              "answer_normalized": "주된 영업소의 소재지가 부산광역시인 기업"}
        status, reason = evaluate_region(CompanyProfile("A", region="서울"), row)
        assert status == "부적합"
        assert "부산광역시" in reason

    def test_region_match_passes(self):
        row = {"status": "value_present",
              "answer_normalized": "주된 영업소의 소재지가 부산광역시인 기업"}
        status, _ = evaluate_region(CompanyProfile("A", region="부산"), row)
        assert status == "충족"

    def test_business_field_absent_is_neutral(self):
        status, _ = evaluate_business_field(CompanyProfile("A"),
                                            {"status": "field_absent", "answer_normalized": ""})
        assert status == "정보없음"

    def test_business_field_mismatch_rejected(self):
        row = {"status": "value_present", "answer_normalized": "농림수산"}
        status, _ = evaluate_business_field(
            CompanyProfile("A", business_fields=["소프트웨어개발"]), row)
        assert status == "부적합"

    def test_business_field_match_passes(self):
        row = {"status": "value_present", "answer_normalized": "농림수산"}
        status, _ = evaluate_business_field(
            CompanyProfile("A", business_fields=["농림수산", "소프트웨어개발"]), row)
        assert status == "충족"

    def test_certification_no_match_needs_review_not_reject(self):
        """법령 인용문이 섞인 자유서술이라 못 찾았다고 부적합으로 단정하지 않는다."""
        row = {"status": "value_present",
              "answer_normalized": "「국가를 당사자로 하는 계약에 관한 법률」 시행령 제12조에 의한 유자격자"}
        status, _ = evaluate_certification(
            CompanyProfile("A", certifications=["소프트웨어사업자 신고필증"]), row)
        assert status == "확인 필요"

    def test_certification_match_passes(self):
        row = {"status": "value_present",
              "answer_normalized": "소프트웨어사업자 신고필증을 보유한 업체"}
        status, reason = evaluate_certification(
            CompanyProfile("A", certifications=["소프트웨어사업자 신고필증"]), row)
        assert status == "충족"


class TestOverallVerdict:
    def test_any_reject_wins(self):
        v = overall_verdict({"a": ("충족", ""), "b": ("부적합", ""), "c": ("확인 필요", "")})
        assert v == "부적합"

    def test_review_when_no_reject(self):
        v = overall_verdict({"a": ("충족", ""), "b": ("확인 필요", "")})
        assert v == "확인 필요"

    def test_pass_when_all_clear(self):
        v = overall_verdict({"a": ("충족", ""), "b": ("정보없음", "")})
        assert v == "적합"

    def test_weak_pass_when_nothing_to_check_at_all(self):
        """전부 '정보없음'(=조건 자체가 없어서 아무것도 못 걸러낸 상태)이면
        '적합'이 아니라 '적합(근거 약함)'으로 구분해야 한다."""
        v = overall_verdict({"a": ("정보없음", ""), "b": ("정보없음", "")})
        assert v == "적합(근거 약함)"


class TestProfileStorage:
    def test_save_then_load_roundtrip(self, tmp_path):
        profile = CompanyProfile(company_name="테스트기업", region="서울",
                                 business_fields=["소프트웨어개발"],
                                 certifications=["소프트웨어사업자 신고필증"])
        save_profile(tmp_path, profile)
        loaded = load_profile(tmp_path, "테스트기업")
        assert loaded.region == "서울"
        assert loaded.business_fields == ["소프트웨어개발"]

    def test_load_missing_company_raises(self, tmp_path):
        import pytest
        with pytest.raises(SystemExit):
            load_profile(tmp_path, "없는회사")
