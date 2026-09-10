"""결함 A — 최종 LLM 심판 준비 판정 (2026-09-04).

★핵심: '실제 LLM을 연결했다'만으로는 최종 심판이 아니다. 준비 조건을 전부 통과해야
  하고, 그 판정은 **한 곳(readiness.evaluate)** 에서만 하며 세 곳이 같은 결과를 쓴다.
    ① 최종 실행 차단  ② judge.usable_for_final  ③ score_breakdown.final_overall

실제 API 는 호출하지 않는다 — 결정적 가짜 provider 와 임시 검증 기록만 쓴다.
"""
import json

import pytest

from grader.judge import Judge, JudgeConfig, JudgeNotReady, StubJudge
from grader.prompts import PromptRepository
from grader.providers import MockJudgeProvider
from grader.readiness import evaluate, normalize_family
from grader.runner import judge_identity, score_breakdown


class FakeProvider:
    """결정적 가짜 provider — 실제 모델을 부르지 않지만 mock 으로 표시되지도 않는다.
    '실제 LLM을 붙였다'는 상황만 흉내낸다."""

    name = "openai_compatible"

    def judge(self, prompt: str) -> dict:
        return {"raw_text": '{"verdict": true, "score": 1.0, "reason": "ok"}'}


def _ready_judge(tmp_path, *, agreement=0.9, blind=True, with_record=True,
                 with_prompt=True, **over):
    """모든 준비 조건을 갖춘 심판. over 로 조건을 하나씩 무너뜨려 본다."""
    vpath = None
    if with_record:
        vpath = tmp_path / "verify.json"
        vpath.write_text(json.dumps({"agreement": agreement, "n": 40, "blind": blind}),
                         encoding="utf-8")
    prompts, files, hashes, versions = {}, {}, {}, {}
    if with_prompt:
        f = tmp_path / "judge_faithfulness.v1.md"
        f.write_text("<!-- name=judge_faithfulness | version=v1 -->\n{answer}\n",
                     encoding="utf-8")
        prompts = {"judge_faithfulness": f}
        files = {"judge_faithfulness": str(f)}
        hashes = {"judge_faithfulness": "a" * 64}
        versions = {"judge_faithfulness": "v1"}
    base = dict(model="claude-sonnet-5", family="anthropic", tier="final",
                temperature=0.0, verification_path=(str(vpath) if vpath else None),
                prompt_versions=versions)
    base.update(over)
    j = Judge(JudgeConfig(**base), prompt_repo=PromptRepository(prompts),
              provider=FakeProvider())
    j.prompt_files = files
    j.prompt_hashes = hashes
    return j


def _ready(judge, **kw):
    kw.setdefault("mode", "final")
    kw.setdefault("generator_family", "openai")
    return evaluate(judge, **kw)


# ── 11. 모든 조건을 만족하면 준비 통과 (기준값) ────────────────────────

def test_fully_prepared_judge_is_ready(tmp_path):
    r = _ready(_ready_judge(tmp_path))
    assert r.ready is True, r.problems
    assert r.problems == []
    assert all(r.checks.values())


# ── 1·2. Stub / Mock 은 항상 final 사용 불가 ──────────────────────────

def test_stub_judge_is_never_usable_for_final():
    r = _ready(StubJudge())
    assert r.ready is False and r.judge_kind == "stub"
    assert any("StubJudge" in p for p in r.problems)


def test_mock_provider_is_never_usable_for_final(tmp_path):
    j = _ready_judge(tmp_path)
    j.provider = MockJudgeProvider()
    r = _ready(j)
    assert r.ready is False and r.judge_kind == "mock"
    assert any("더미(mock)" in p for p in r.problems)


# ── 3. 실제 provider 지만 dev/independent 등급 ────────────────────────

@pytest.mark.parametrize("tier", ["dev", "independent"])
def test_real_provider_with_non_final_tier_is_not_ready(tmp_path, tier):
    r = _ready(_ready_judge(tmp_path, tier=tier))
    assert r.ready is False
    assert any(f"tier={tier}" in p for p in r.problems)


# ── 4·5·6. 사람 대조 기록 ─────────────────────────────────────────────

def test_missing_human_verification_blocks_final(tmp_path):
    r = _ready(_ready_judge(tmp_path, with_record=False))
    assert r.ready is False
    assert any("사람 채점 대조 기록 없음" in p for p in r.problems)
    assert r.checks["agreement_meets_threshold"] is False


def test_agreement_below_threshold_blocks_final(tmp_path):
    r = _ready(_ready_judge(tmp_path, agreement=0.5))
    assert r.ready is False
    assert any("일치도" in p for p in r.problems)


def test_non_blind_verification_blocks_final(tmp_path):
    r = _ready(_ready_judge(tmp_path, blind=False))
    assert r.ready is False
    assert any("블라인드" in p for p in r.problems)


# ── 7. 채점 프롬프트 ──────────────────────────────────────────────────

def test_missing_prompt_file_blocks_final(tmp_path):
    j = _ready_judge(tmp_path)
    j.prompt_files = {"judge_faithfulness": str(tmp_path / "없는파일.md")}
    r = _ready(j)
    assert r.ready is False
    assert any("프롬프트 파일 없음" in p for p in r.problems)


def test_missing_prompt_version_or_hash_blocks_final(tmp_path):
    r = _ready(_ready_judge(tmp_path, prompt_versions={"judge_faithfulness": ""}))
    assert r.ready is False
    assert any("프롬프트 버전 미기록" in p for p in r.problems)

    j = _ready_judge(tmp_path)
    j.prompt_hashes = {}
    assert any("프롬프트 해시 미기록" in p for p in _ready(j).problems)


# ── 8. 생성 모델 계열 미기록 ──────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "none", "UNKNOWN"])
def test_missing_generator_family_blocks_final(tmp_path, value):
    r = _ready(_ready_judge(tmp_path), generator_family=value)
    assert r.ready is False
    assert any("생성 모델 계열" in p for p in r.problems)


# ── 9·10. 계열 정규화와 자기 편향 ─────────────────────────────────────

def test_family_aliases_normalize():
    assert normalize_family("gpt") == normalize_family("openai") == "openai"
    assert normalize_family("claude") == normalize_family("anthropic") == "anthropic"
    assert normalize_family("gemini") == normalize_family("google") == "google"
    assert normalize_family("GPT") == "openai"          # 대소문자 무시
    assert normalize_family(None) is None and normalize_family("none") is None


def test_gpt_and_openai_are_not_treated_as_different_families(tmp_path):
    """★'gpt' 와 'openai' 를 다른 계열로 보면 자기 편향 검사를 통과해 버린다."""
    j = _ready_judge(tmp_path, family="openai")
    r = _ready(j, generator_family="gpt")
    assert r.ready is False
    assert any("자기 편향" in p for p in r.problems)
    assert r.generator_family == r.judge_family == "openai"


def test_same_family_blocks_and_different_family_passes(tmp_path):
    assert _ready(_ready_judge(tmp_path, family="anthropic"),
                  generator_family="claude").ready is False
    assert _ready(_ready_judge(tmp_path, family="anthropic"),
                  generator_family="openai").ready is True


# ── 12·13. 세 곳이 같은 판정을 쓴다 ───────────────────────────────────

class _Runner:
    def __init__(self, judge, use_stub=False):
        self.judge = judge
        class _C:
            use_stub_judge = use_stub
        self.config = _C()


def test_final_overall_is_a_number_only_when_fully_ready(tmp_path):
    judge = _ready_judge(tmp_path)
    r = _ready(judge)
    ident = judge_identity(_Runner(judge), r)
    rep = {"overall_score": 0.87, "main": {"by_task_type": {}}, "faithfulness": {}}
    out = score_breakdown(rep, ident, r)
    assert ident["usable_for_final"] is True
    assert out["final_overall"] == 0.87


def test_development_run_with_a_real_provider_has_no_final_overall(tmp_path):
    """★실제 LLM 을 붙여도 개발 실행이면 최종 점수가 아니다."""
    judge = _ready_judge(tmp_path)
    r = evaluate(judge, mode="development", generator_family="openai")
    ident = judge_identity(_Runner(judge), r)
    rep = {"overall_score": 0.87, "main": {"by_task_type": {}}, "faithfulness": {}}
    out = score_breakdown(rep, ident, r)
    assert r.ready is False
    assert ident["usable_for_final"] is False
    assert out["final_overall"] is None
    # 버리지는 않는다 — 잠정값으로 남긴다
    assert out["provisional_real_judge"]["applicable"] is True
    assert out["provisional_real_judge"]["value"] == 0.87


def test_unverified_real_judge_is_consistent_across_all_three_places(tmp_path):
    """게이트가 막았는데 보고서만 최종 점수를 내던 모순이 재발하지 않는다."""
    judge = _ready_judge(tmp_path, with_record=False)
    r = _ready(judge)
    ident = judge_identity(_Runner(judge), r)
    out = score_breakdown({"overall_score": 0.9, "main": {"by_task_type": {}}}, ident, r)
    assert r.ready is False                      # ① 실행 차단
    assert ident["usable_for_final"] is False    # ② 심판 표시
    assert out["final_overall"] is None          # ③ 통합 점수
    assert ident["kind"] == "real_judge"         # 실제 LLM 이라는 사실 자체는 남는다
    with pytest.raises(JudgeNotReady):
        judge.assert_ready(generator_family="openai", strict=True, mode="final")
