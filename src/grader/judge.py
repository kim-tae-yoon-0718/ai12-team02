"""
grader.judge — 3-8 LLM 채점기 설계 / 3-9 검증 / 3-11 심판 등급

★ 심판의 기본 세팅 세 가지 — 이 모듈이 **코드로 강제**한다.
   ① temperature = 0        : 같은 답변에 같은 점수가 나와야 한다.
   ② 답변 생성 모델과 다른 계열 : 같은 계열이면 자기 스타일을 후하게 채점한다.
   ③ 사람 채점 샘플과 대조해 검증한 뒤 사용 : 검증 전의 심판은 "숫자가 나오는 것"이지
                              채점이 아니다.
  이 셋을 안 잡고 채점을 시작하면 이후 실험 비교가 전부 흔들린다.
  ⇒ Judge.assert_ready() 를 통과하지 못하면 tier=final 실행을 거부한다(김태윤 팀장 합의:
    사람 대조 + independent 검증 → final Judge 순서).

【22】채점 프롬프트는 생성 프롬프트보다 엄격히 관리한다.
【23】프롬프트는 파일로 분리하고 버전을 붙인다 → data/prompts/<name>.<version>.md
  ★채점 프롬프트가 바뀌면 이전 점수와의 비교가 무효가 될 수 있다 — grader.contamination 참고.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable

from .normalize import normalize_text
from .prompts import PromptRepository
from .providers import JudgeProvider


class JudgeNotReady(RuntimeError):
    """3-8/3-9 세팅 위반. 최종(tier=final) 채점을 진행시키지 않는다."""


@dataclass
class JudgeConfig:
    model: str
    family: str  # 예: "llama", "qwen", "claude", "gpt"
    tier: str = "dev"  # dev(작고 싼) | independent | final(강한) — 3-11 캐스케이드
    temperature: float = 0.0
    prompt_versions: dict[str, str] = field(default_factory=dict)  # {"judge_faithfulness": "v1"}
    verification_path: str | None = None  # 3-9 사람 대조 기록 파일
    min_agreement: float = 0.80  # 사람-심판 일치도 하한 [대기] 3-9 실측 후 조정

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            {"m": self.model, "t": self.temperature, "p": self.prompt_versions},
            sort_keys=True).encode()).hexdigest()[:12]


def load_verification(path: str) -> dict:
    """3-9 검증 기록. 최소 구조:
       {"agreement": 0.86, "n": 40, "blind": true, "rubric_version": "v1",
        "date": "2026-09-02", "reviewers": ["김하루","임현진"], "notes": "..."}"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_verdict(raw: str) -> dict:
    """심판 출력 파싱. 이진(yes/no) 스케일을 강제한다 —
    세밀한 점수일수록 채점기 신뢰도가 떨어지는 경향이 있다(3-8)."""
    raw = (raw or "").strip()
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        v = d.get("verdict")
        yes = v in (True, "yes", "Y", 1) or (isinstance(v, str) and v.strip().lower() == "yes")
        return {"verdict": yes, "score": 1.0 if yes else 0.0, "reason": d.get("reason", "")}
    except Exception:
        yes = raw.lower().startswith(("yes", "예", "true", "1"))
        return {"verdict": yes, "score": 1.0 if yes else 0.0, "reason": raw[:200]}


class Judge:
    """심판 껍데기. 실제 호출은 provider 로 주입한다 —
    로컬 모델/API/스텁을 바꿔 끼워도 채점 규칙과 가드는 그대로 유지된다."""

    def __init__(self, cfg: JudgeConfig, prompt_repo: PromptRepository,
                 provider: JudgeProvider | None = None, cache_path: str | None = None):
        self.cfg = cfg
        self.prompt_repo = prompt_repo
        self.provider = provider
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self._cache = json.load(f)

    # ---- ★ 세팅 3종 가드 ----
    def assert_ready(self, generator_family: str | None = None, strict: bool = True) -> list[str]:
        problems = []
        if self.cfg.temperature != 0:
            problems.append(f"① temperature={self.cfg.temperature} ≠ 0 — 측정 도구가 흔들린다")
        if generator_family and generator_family == self.cfg.family:
            problems.append(f"② 심판 계열({self.cfg.family})이 생성 모델과 같다 — 자기 편향")
        if not self.cfg.verification_path or not os.path.exists(self.cfg.verification_path or ""):
            problems.append("③ 사람 채점 대조 기록 없음 — 검증 전 심판은 채점이 아니다(3-9)")
        else:
            rec = load_verification(self.cfg.verification_path)
            if rec.get("agreement", 0) < self.cfg.min_agreement:
                problems.append(
                    f"③ 사람-심판 일치도 {rec.get('agreement')} < {self.cfg.min_agreement} — "
                    "채점 프롬프트를 고치고 재검증할 것")
            if not rec.get("blind", False):
                problems.append("③ 사람 채점이 블라인드가 아니었음 — 기준 자체가 흔들린다")
        if problems and strict and self.cfg.tier == "final":
            raise JudgeNotReady("; ".join(problems))
        return problems

    # ---- 채점 ----
    def score(self, prompt_name: str, **vars_) -> dict:
        prompt = self.prompt_repo.render(prompt_name, vars_)
        version = self.cfg.prompt_versions.get(prompt_name) or self.prompt_repo.version_of(prompt_name)
        key = hashlib.sha256(
            (self.cfg.fingerprint() + prompt).encode("utf-8")).hexdigest()[:20]
        if key in self._cache:
            return self._cache[key]
        if self.provider is None:
            raise JudgeNotReady("provider 미주입 — 실제 모델을 붙이거나 StubJudge 를 쓰라")
        raw = self.provider.judge(prompt).get("raw_text", "")
        out = parse_verdict(raw)
        out["_prompt_version"] = version
        out["_judge"] = f"{self.cfg.model}/{self.cfg.tier}"
        self._cache[key] = out
        self._flush()
        return out

    def _flush(self):
        if self.cache_path:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)


class StubJudge(Judge):
    """오프라인 대체 심판. 문자열 포함 여부로 판정한다.
    ★이것은 심판이 아니다 — 채점기 자체 테스트와 파이프라인 점검 전용.
    최종 점수 산출에 쓰면 안 되며, tier='final' 이면 가드가 막는다."""

    def __init__(self):
        super().__init__(
            JudgeConfig(model="stub", family="none", tier="dev", temperature=0.0),
            prompt_repo=PromptRepository({}),
        )

    def score(self, prompt_name: str, **vars_) -> dict:
        target = str(vars_.get("item") or vars_.get("checkpoint") or "")
        text = str(vars_.get("answer") or "")
        ok = bool(target) and normalize_text(target) in normalize_text(text)
        return {"verdict": ok, "score": 1.0 if ok else 0.0,
                "reason": "stub: 문자열 포함 판정", "_judge": "stub/dev"}


def tier_correlation(dev_scores: list[float], final_scores: list[float]) -> dict:
    """3-11 : 심판을 등급으로 나누기 **전에** 두 심판의 점수 상관을 먼저 측정할 것.
    결이 다르면 개발용 심판이 엉뚱한 실험을 통과시키거나 좋은 실험을 걸러낸다."""
    n = len(dev_scores)
    if n < 2 or n != len(final_scores):
        return {"n": n, "correlation": None, "usable": False,
                "note": "표본 부족 — 등급 분리 금지"}
    mx, my = sum(dev_scores) / n, sum(final_scores) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(dev_scores, final_scores))
    vx = sum((a - mx) ** 2 for a in dev_scores) ** 0.5
    vy = sum((b - my) ** 2 for b in final_scores) ** 0.5
    r = cov / (vx * vy) if vx and vy else None
    return {"n": n, "correlation": round(r, 4) if r is not None else None,
            "usable": bool(r is not None and r >= 0.8),
            "note": "r<0.8 이면 개발용 심판으로 실험을 가르지 말 것. "
                    "어느 심판으로 잰 점수인지 결과에 반드시 기록"}


ItemMatcher = Callable[[str, str], bool]


def make_item_matcher(judge: Judge, prompt_name: str = "judge_list_item",
                      var_name: str = "item") -> ItemMatcher:
    """항목 단위 '언급했다' 판정을 심판에게 위임하는 어댑터.

    문자열 매칭으로는 표현 변형을 놓친다(3-4-4). judge_list_item(목록형) /
    judge_checkpoint(요약형, var_name="checkpoint")에 공용으로 쓴다.
    """
    def _match(gold_item: str, answer_text: str) -> bool:
        kwargs = {var_name: gold_item, "answer": answer_text}
        out = judge.score(prompt_name, **kwargs)
        return bool(out.get("score", 0) >= 1.0)
    return _match
