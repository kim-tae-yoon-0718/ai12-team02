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


def test_verification_record_satisfies_gate(tmp_path):
    rec = {"agreement": 0.9, "n": 40, "blind": True}
    path = tmp_path / "verify.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    cfg = JudgeConfig(model="m", family="claude", tier="final", temperature=0.0,
                      verification_path=str(path))
    judge = Judge(cfg, prompt_repo=PromptRepository({}))
    problems = judge.assert_ready(generator_family="gpt")
    assert problems == []


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
