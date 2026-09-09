"""회사 추천기가 자유 서술을 자동 합격시키지 않는지 확인한다."""
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "company_match"
sys.path.insert(0, str(TOOLS_DIR))
import company_match as C  # noqa: E402


def row(field_name, status="value_present", answer="", document_id="RFP-1"):
    return {
        "active": "true",
        "retrieval_eligible": "true",
        "document_id": document_id,
        "field_name": field_name,
        "status": status,
        "answer_normalized": answer,
        "representative_location": {"heading": field_name, "line_start": 10},
    }


def complete_rows(overrides=None):
    overrides = overrides or {}
    return [
        overrides.get(
            field_name,
            row(field_name, status="field_absent"),
        )
        for field_name in C.MATCHED_FIELDS
    ]


class TestProfileStorage:
    def test_round_trip_preserves_consortium_state(self, tmp_path):
        profile = C.CompanyProfile(
            "테스트회사", "서울", ["AI"], ["정보통신공사업"], True,
        )
        C.save_profile(tmp_path, profile)
        assert C.load_profile(tmp_path, "테스트회사") == profile

    def test_unknown_company_is_not_silently_created(self, tmp_path):
        with pytest.raises(LookupError):
            C.load_profile(tmp_path, "없는회사")


class TestSafeFieldDecisions:
    def test_exact_closed_region_match_is_confirmed(self):
        decision = C.evaluate_region(
            C.CompanyProfile("A", region="부산"),
            row(C.REGION_FIELD, answer="주된 영업소가 부산광역시에 소재한 기업"),
        )
        assert decision["status"] == C.FIELD_CONFIRMED

    def test_exact_closed_region_mismatch_is_only_exclusion_rule(self):
        decision = C.evaluate_region(
            C.CompanyProfile("A", region="서울"),
            row(C.REGION_FIELD, answer="지역제한(광주광역시)"),
        )
        assert decision["status"] == C.FIELD_CONTRADICTION

    def test_multiple_regions_are_not_guessed(self):
        decision = C.evaluate_region(
            C.CompanyProfile("A", region="서울"),
            row(C.REGION_FIELD, answer="서울특별시 또는 경기도 소재 업체"),
        )
        assert decision["status"] == C.FIELD_REVIEW

    def test_business_substring_overlap_never_passes(self):
        decision = C.evaluate_business_field(
            C.CompanyProfile("A", business_fields=["소프트웨어 개발"]),
            row(C.BUSINESS_FIELD, answer="소프트웨어 개발 및 정보통신공사업"),
        )
        assert decision["status"] == C.FIELD_REVIEW

    def test_certification_substring_overlap_never_passes(self):
        decision = C.evaluate_certification(
            C.CompanyProfile("A", certifications=["소프트웨어사업자 신고필증"]),
            row(C.CERT_FIELD, answer="소프트웨어사업자로 등록하고 최근 실적을 보유한 업체"),
        )
        assert decision["status"] == C.FIELD_REVIEW

    @pytest.mark.parametrize(
        "text",
        [
            "공동수급을 허용하지 않음",
            "공동수급은 허용하지만 구성원별 지분은 10% 이상",
            "공동수급은 필수가 아니며 단독 참여 가능",
        ],
    )
    def test_consortium_language_is_always_reviewed(self, text):
        decision = C.evaluate_consortium(
            C.CompanyProfile("A", consortium_needed=True),
            row(C.CONSORTIUM_FIELD, answer=text),
        )
        assert decision["status"] == C.FIELD_REVIEW
        assert decision["evidence"] == text


class TestVerdictAndScope:
    def test_exact_match_prioritizes_but_keeps_review_flag(self):
        rows = complete_rows({
            C.REGION_FIELD: row(C.REGION_FIELD, answer="서울특별시 소재 업체"),
            C.CERT_FIELD: row(C.CERT_FIELD, answer="관련 법령에 따른 참가 자격"),
        })
        result = C.match_company(C.CompanyProfile("A", region="서울"), rows)[0]
        assert result["verdict"] == C.VERDICT_PRIORITY
        assert result["review_count"] == 1
        assert result["reasons"][C.CERT_FIELD]["status"] == C.FIELD_REVIEW

    def test_explicit_region_contradiction_excludes(self):
        rows = complete_rows({
            C.REGION_FIELD: row(C.REGION_FIELD, answer="부산광역시 소재 업체"),
        })
        result = C.match_company(C.CompanyProfile("A", region="서울"), rows)[0]
        assert result["verdict"] == C.VERDICT_EXCLUDED

    def test_all_unstated_is_review_not_participation_confirmation(self):
        result = C.match_company(C.CompanyProfile("A", region="서울"), complete_rows())[0]
        assert result["verdict"] == C.VERDICT_REVIEW

    def test_registry_scope_is_respected(self):
        rows = complete_rows() + complete_rows({
            field_name: row(field_name, status="field_absent", document_id="RFP-2")
            for field_name in C.MATCHED_FIELDS
        })
        results = C.match_company(
            C.CompanyProfile("A", region="서울"), rows, {"RFP-2"},
        )
        assert [item["document_id"] for item in results] == ["RFP-2"]

    def test_duplicate_active_row_is_rejected(self):
        duplicate = row(C.REGION_FIELD)
        with pytest.raises(ValueError, match="중복"):
            C.group_rows_by_document([duplicate, dict(duplicate)])
