import json

import pytest

from grader.judge import (
    Judge,
    JudgeConfig,
    JudgeNotReady,
    StubJudge,
    make_item_matcher,
    parse_verdict,
    tier_correlation,
)
from grader.prompts import PromptRepository


def test_final_judge_requires_all_three_gates():
    cfg = JudgeConfig(model="m", family="llama", tier="final", temperature=0.7)
    judge = Judge(cfg, prompt_repo=PromptRepository({}))
    with pytest.raises(JudgeNotReady):
        judge.assert_ready(generator_family="llama")


def test_dev_judge_reports_but_does_not_raise():
    cfg = JudgeConfig(model="m", family="llama", tier="dev", temperature=0.0)
    judge = Judge(cfg, prompt_repo=PromptRepository({}))
    problems = judge.assert_ready(generator_family="qwen")
    assert any("③" in p for p in problems)  # 미검증은 반드시 보고된다


class _RealishProvider:
    """실제 모델을 호출하는 provider 를 흉내낸 최소 대역 — mock 과 달리
    usable_for_final 을 False 로 표시하지 않는다."""

    name = "openai_compatible"

    def judge(self, prompt: str) -> dict:
        return {"raw_text": '{"verdict": true, "score": 1.0, "reason": "ok"}'}


def ready_judge(tmp_path, **over):
    """최종 준비 조건을 모두 갖춘 심판 — 개별 조건을 하나씩 무너뜨려 볼 기준값.

    [2026-09-04] 프롬프트 파일·버전·해시도 최종 준비 조건이라(§2) 픽스처가
    실제로 그것들을 갖춰야 한다. 단언은 그대로 두고 픽스처만 조건에 맞췄다.
    """
    rec = {"agreement": 0.9, "n": 40, "blind": True}
    path = tmp_path / "verify.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    prompt = tmp_path / "judge_faithfulness.v1.md"
    prompt.write_text("<!-- name=judge_faithfulness | version=v1 -->\n{answer}\n",
                      encoding="utf-8")
    base = dict(model="m", family="claude", tier="final", temperature=0.0,
                verification_path=str(path),
                prompt_versions={"judge_faithfulness": "v1"})
    base.update(over)
    judge = Judge(JudgeConfig(**base),
                  prompt_repo=PromptRepository({"judge_faithfulness": prompt}),
                  provider=_RealishProvider())
    judge.prompt_files = {"judge_faithfulness": str(prompt)}
    judge.prompt_hashes = {"judge_faithfulness": "0" * 64}
    return judge


def test_verification_record_satisfies_gate(tmp_path):
    judge = ready_judge(tmp_path)
    problems = judge.assert_ready(generator_family="gpt")
    assert problems == []


def test_gate_rejects_missing_or_dummy_provider(tmp_path):
    """★실제 심판이 아닌데 final 게이트를 통과하면 안 된다(§6-3)."""
    from grader.judge import StubJudge
    from grader.providers import MockJudgeProvider

    rec = {"agreement": 0.9, "n": 40, "blind": True}
    path = tmp_path / "verify.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    base = dict(model="m", family="claude", tier="final", temperature=0.0,
                verification_path=str(path))

    no_provider = Judge(JudgeConfig(**base), prompt_repo=PromptRepository({}))
    assert any("provider 미주입" in p for p in no_provider.assert_ready(
        generator_family="gpt", strict=False))

    mocked = Judge(JudgeConfig(**base), prompt_repo=PromptRepository({}),
                   provider=MockJudgeProvider())
    assert mocked.is_mock is True
    assert any("더미(mock)" in p for p in mocked.assert_ready(
        generator_family="gpt", strict=False))

    stub = StubJudge()
    assert any("StubJudge" in p for p in stub.assert_ready(
        generator_family="gpt", strict=False))


def test_gate_requires_recorded_model_family_and_tier(tmp_path):
    rec = {"agreement": 0.9, "n": 40, "blind": True}
    path = tmp_path / "verify.json"
    path.write_text(json.dumps(rec), encoding="utf-8")

    def probe(**over):
        base = dict(model="m", family="claude", tier="final", temperature=0.0,
                    verification_path=str(path))
        base.update(over)
        j = Judge(JudgeConfig(**base), prompt_repo=PromptRepository({}),
                  provider=_RealishProvider())
        return j.assert_ready(generator_family=over.pop("_gen", "gpt"), strict=False)

    assert any("model=" in p for p in probe(model="stub"))
    assert any("family=" in p for p in probe(family="none"))
    assert any("tier=" in p for p in probe(tier="dev"))
    # 생성 모델 계열이 기록 안 되면 '심판과 같은 계열인지' 판단 자체가 불가능하다
    j = Judge(JudgeConfig(model="m", family="claude", tier="final", temperature=0.0,
                          verification_path=str(path)),
              prompt_repo=PromptRepository({}), provider=_RealishProvider())
    assert any("생성 모델 계열" in p for p in j.assert_ready(generator_family=None, strict=False))


def test_tier_split_requires_correlation():
    r = tier_correlation([0.5], [0.6])
    assert r["usable"] is False


def test_stub_judge_is_not_a_judge():
    out = StubJudge().score("judge_list_item", item="가", answer="가 나")
    assert out["score"] == 1.0
    assert "stub" in out["_judge"]


def test_parse_verdict_yes_no_json():
    assert parse_verdict('{"verdict": "yes", "reason": "ok"}')["score"] == 1.0
    assert parse_verdict('{"verdict": "no", "reason": "bad"}')["score"] == 0.0
    assert parse_verdict("garbage")["score"] == 0.0


def test_make_item_matcher_uses_judge_score():
    matcher = make_item_matcher(StubJudge(), "judge_list_item", "item")
    assert matcher("사업자등록증", "사업자등록증 사본 포함") is True
    assert matcher("법인등기부등본", "사업자등록증만 있음") is False
