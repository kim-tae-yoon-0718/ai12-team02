"""session_start_experiment의 회사 조회/등록 분기 테스트 — input()을 흉내내서 확인.

'아니오'를 고르면 등록하지 않고 이름을 다시 물어보는지, '예'를 고르면 그 자리에서
등록하는지 확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "company_match"))

import pytest  # noqa: E402

import company_match as C  # noqa: E402
import session_start_experiment as S  # noqa: E402


def _feed(monkeypatch, answers: list[str]):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(it))


class TestGetOrRegisterProfile:
    def test_existing_company_loads_without_asking_to_register(self, tmp_path, monkeypatch):
        C.save_profile(tmp_path, C.CompanyProfile(company_name="기존회사", region="서울"))
        _feed(monkeypatch, [])  # 아무 입력도 필요 없어야 한다
        name, profile, consortium = S.get_or_register_profile(tmp_path, "기존회사")
        assert name == "기존회사"
        assert profile.region == "서울"

    def test_no_reenters_name_instead_of_registering(self, tmp_path, monkeypatch):
        C.save_profile(tmp_path, C.CompanyProfile(company_name="진짜회사", region="부산"))
        # 없는회사 -> 등록? '아니오' -> 다시 물어봄 -> '진짜회사' 입력 -> 바로 불러옴
        _feed(monkeypatch, ["아니오", "진짜회사"])
        name, profile, consortium = S.get_or_register_profile(tmp_path, "없는회사")
        assert name == "진짜회사"
        assert profile.region == "부산"

    def test_yes_registers_on_the_spot(self, tmp_path, monkeypatch):
        # 없는회사 -> 등록? '예' -> 각 항목마다 입력 후 '맞나요?' 확인까지 거쳐서 저장
        _feed(monkeypatch, [
            "예",                      # 등록할까요?
            "대전", "예",               # 지역 입력 + 확인
            "소프트웨어개발", "예",       # 사업분야 입력 + 확인
            "", "예",                  # 자격증 입력(빈 값) + 확인
            "n",                       # 컨소시엄 필요 여부
        ])
        name, profile, consortium = S.get_or_register_profile(tmp_path, "새회사")
        assert name == "새회사"
        assert profile.region == "대전"
        assert profile.business_fields == ["소프트웨어개발"]
        assert consortium is False
        # 실제로 저장까지 됐는지 재조회로 확인
        reloaded = C.load_profile(tmp_path, "새회사")
        assert reloaded.region == "대전"

    def test_wrong_value_can_be_retyped_before_saving(self, tmp_path, monkeypatch):
        # 지역을 잘못 입력 -> '아니오' -> 다시 입력 -> '예'로 확정
        _feed(monkeypatch, [
            "예",                      # 등록할까요?
            "대전", "아니오",           # 지역 오타 입력 + 부정
            "서울", "예",               # 재입력 + 확인
            "", "예",                  # 사업분야 없음 + 확인
            "", "예",                  # 자격증 없음 + 확인
            "n",
        ])
        name, profile, consortium = S.get_or_register_profile(tmp_path, "오타회사")
        assert profile.region == "서울"


class TestCancel:
    def test_quit_word_at_company_name_cancels(self, tmp_path, monkeypatch):
        _feed(monkeypatch, ["취소"])
        with pytest.raises(S.CancelledByUser):
            S.get_or_register_profile(tmp_path, None)

    def test_quit_word_at_register_confirm_cancels(self, tmp_path, monkeypatch):
        _feed(monkeypatch, ["q"])
        with pytest.raises(S.CancelledByUser):
            S.get_or_register_profile(tmp_path, "없는회사")

    def test_quit_word_mid_registration_cancels_without_saving(self, tmp_path, monkeypatch):
        # 등록? 예 -> 지역 입력 중에 '그만'
        _feed(monkeypatch, ["예", "그만"])
        with pytest.raises(S.CancelledByUser):
            S.get_or_register_profile(tmp_path, "중도포기회사")
        # 취소했으니 저장도 안 돼야 한다
        with pytest.raises(SystemExit):
            C.load_profile(tmp_path, "중도포기회사")
