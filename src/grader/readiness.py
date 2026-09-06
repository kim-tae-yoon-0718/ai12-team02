"""최종 심판 준비 판정 — **한 곳에서만** 계산한다.

★왜 한 곳인가:
  예전에는 세 곳이 각자 판단했다.
    ① 최종 실행 차단   (judge.assert_ready — 모든 조건 검사)
    ② judge.usable_for_final (stub/mock 여부만 봄)
    ③ score_breakdown.final_overall (stub/mock 여부만 봄)
  그래서 **검증되지 않은 실제 LLM**을 붙이면 ②③이 "최종 사용 가능 / 통합 점수 = 0.87"
  이라고 보고했다. 실행은 ①이 막았는데 보고서는 최종 점수를 냈다 — 읽는 사람이
  그 숫자를 최종 점수로 받아들인다.

  이제 evaluate() 하나가 판정하고 세 곳이 같은 결과를 쓴다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# 모델 계열 별칭 정규화. 같은 계열을 다른 이름으로 적어 두면 "생성·심판이 같은 계열"
# 검사가 통과해 버린다 — 자기 편향을 못 잡는다.
FAMILY_ALIASES = {
    "gpt": "openai", "openai": "openai", "chatgpt": "openai", "o1": "openai",
    "claude": "anthropic", "anthropic": "anthropic",
    "gemini": "google", "google": "google", "palm": "google",
    "llama": "meta", "meta": "meta",
    "qwen": "alibaba", "alibaba": "alibaba",
}

_EMPTY = {None, "", "none", "null", "unknown", "todo", "[대기]"}


def normalize_family(value) -> str | None:
    """모델 계열 문자열을 표준값으로. 알 수 없으면 소문자 원문, 비어 있으면 None."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _EMPTY:
        return None
    return FAMILY_ALIASES.get(s, s)


@dataclass
class FinalReadiness:
    """최종 심판으로 쓸 수 있는가 — 판정 결과와 근거."""

    ready: bool
    problems: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    mode: str | None = None
    judge_kind: str | None = None
    generator_family: str | None = None
    judge_family: str | None = None

    def model_dump(self) -> dict:
        return {
            "ready": self.ready, "mode": self.mode, "judge_kind": self.judge_kind,
            "generator_family": self.generator_family, "judge_family": self.judge_family,
            "checks": self.checks, "problems": self.problems,
        }


def judge_kind_of(judge, config=None) -> str:
    """stub / mock / real_judge — provider 가 스스로 밝힌 값을 신뢰한다."""
    from .judge import StubJudge
    if (config is not None and bool(getattr(config, "use_stub_judge", False))) \
            or isinstance(judge, StubJudge):
        return "stub"
    if bool(getattr(judge, "is_mock", False)):
        return "mock"
    return "real_judge"


def evaluate(judge, *, mode: str | None = None, generator_family=None,
             config=None, prompt_hashes: dict | None = None) -> FinalReadiness:
    """최종 심판 준비 상태를 판정한다.

    ★"실제 LLM이다"만으로는 부족하다. 아래를 **전부** 통과해야 최종 심판이다.
      실행 모드가 final / Stub·Mock 아님 / 모델명·계열 기록 / 등급 final /
      temperature 0 / 생성 모델과 다른 계열 / 사람 대조 기록 존재·일치도·블라인드 /
      프롬프트 파일 존재 + 버전·해시 기록 / 생성 모델 계열 기록
    """
    from .judge import load_verification

    cfg = judge.cfg
    kind = judge_kind_of(judge, config)
    gen_fam = normalize_family(generator_family)
    jud_fam = normalize_family(cfg.family)
    problems: list[str] = []
    checks: dict[str, bool] = {}

    def need(key: str, ok: bool, message: str) -> None:
        checks[key] = bool(ok)
        if not ok:
            problems.append(message)

    need("mode_is_final", mode == "final",
         f"실행 모드가 final 이 아니다(mode={mode}) — 최종 점수는 final 실행에서만 낸다")
    need("judge_is_not_stub", kind != "stub",
         "⓪ StubJudge — 경계 포함 문자열 매처이며 의미 비교가 아니다")
    need("judge_is_not_mock", kind != "mock",
         f"⓪ provider={getattr(judge, 'provider_name', '?')} 는 더미(mock) — "
         f"실제 모델을 호출하지 않는다")
    need("provider_present", judge.provider is not None,
         "⓪ provider 미주입 — 호출할 모델이 없다")
    need("judge_model_recorded",
         bool(cfg.model) and str(cfg.model).lower() not in _EMPTY | {"stub", "mock"},
         f"⓪ 심판 model={cfg.model!r} — 실제 모델명이 기록되지 않았다")
    need("judge_family_recorded", jud_fam is not None,
         f"⓪ 심판 family={cfg.family!r} — 계열이 기록되지 않았다")
    need("judge_tier_is_final", cfg.tier == "final",
         f"④ 심판 tier={cfg.tier} — final 실행에는 tier=final 이어야 한다")
    need("temperature_is_zero", cfg.temperature == 0,
         f"① temperature={cfg.temperature} ≠ 0 — 측정 도구가 흔들린다")
    need("generator_family_recorded", gen_fam is not None,
         "② 생성 모델 계열(family)이 기록되지 않았다 — 심판과 같은 계열인지 "
         "확인할 수 없다. --generator-family 로 지정하라")
    # 계열 비교는 **정규화한 값**으로. 'gpt' 와 'openai' 는 같은 계열이다.
    same_family = bool(gen_fam and jud_fam and gen_fam == jud_fam)
    need("families_differ", not same_family,
         f"② 심판 계열({cfg.family}→{jud_fam})이 생성 모델({generator_family}→{gen_fam})과 "
         f"같다 — 자기 편향")

    vpath = cfg.verification_path
    has_record = bool(vpath) and os.path.exists(vpath or "")
    need("human_verification_exists", has_record,
         "③ 사람 채점 대조 기록 없음 — 검증 전 심판은 채점이 아니다(3-9)")
    if has_record:
        rec = load_verification(vpath)
        need("agreement_meets_threshold", rec.get("agreement", 0) >= cfg.min_agreement,
             f"③ 사람-심판 일치도 {rec.get('agreement')} < {cfg.min_agreement} — "
             f"채점 프롬프트를 고치고 재검증할 것")
        need("verification_is_blind", bool(rec.get("blind", False)),
             "③ 사람 채점이 블라인드가 아니었음 — 기준 자체가 흔들린다")
    else:
        checks["agreement_meets_threshold"] = False
        checks["verification_is_blind"] = False

    versions = dict(cfg.prompt_versions or {})
    need("prompt_versions_recorded", bool(versions) and all(versions.values()),
         f"⑤ 심판 프롬프트 버전 미기록: {[n for n, v in versions.items() if not v] or '(비어 있음)'}")
    files = dict(getattr(judge, "prompt_files", None) or
                 getattr(config, "judge", None) and getattr(config.judge, "prompt_files", {}) or {})
    missing_files = [str(p) for p in files.values() if not os.path.exists(p)]
    need("prompt_files_exist", bool(files) and not missing_files,
         f"⑤ 심판 프롬프트 파일 없음: {missing_files or '(설정에 경로 없음)'}")
    # 해시는 인자로 받되, 없으면 심판 객체가 들고 있는 값을 쓴다 — 호출부마다
    # 다른 결과가 나오지 않게 한 곳에서 같은 순서로 찾는다.
    hashes = dict(prompt_hashes or getattr(judge, "prompt_hashes", None) or {})
    need("prompt_hashes_recorded", bool(hashes) and all(hashes.values()),
         f"⑤ 심판 프롬프트 해시 미기록: {[n for n, v in hashes.items() if not v] or '(비어 있음)'}")

    return FinalReadiness(
        ready=not problems, problems=problems, checks=checks, mode=mode,
        judge_kind=kind, generator_family=gen_fam, judge_family=jud_fam)
