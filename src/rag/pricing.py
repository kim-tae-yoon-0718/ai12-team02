"""
API 비용 계산 — 단가 단위는 **100만(1M) 토큰당 달러**로 통일한다.

⚠️ 2026-09-02: 예전 base.yaml은 `input_per_1k`라는 이름에 1M 기준 값을 넣을
   위험이 있었다(1,000배 오차). 키 이름에 단위를 박아(`*_per_1m`) 혼동을 없앴다.

공식 가격표 (2026-09-02 확인)
  gpt-5-mini             : 입력 $0.25 / 캐시입력 $0.025 / 출력 $2.00  per 1M
  text-embedding-3-small : 입력 $0.02                                per 1M
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

PRICING_UNIT = "per_1m_tokens"
_PER = 1_000_000


class PricingError(RuntimeError):
    pass


@dataclass
class Usage:
    """한 문항에서 실제로 쓴 토큰. 문항 시작 시 항상 새로 만든다."""
    generation_requests: int = 0
    generation_input_tokens: int = 0
    generation_cached_input_tokens: int = 0
    generation_output_tokens: int = 0
    embedding_requests: int = 0
    embedding_tokens: int = 0

    def add_generation(self, prompt_tokens: int, completion_tokens: int,
                       cached_tokens: int = 0) -> None:
        self.generation_requests += 1
        self.generation_input_tokens += int(prompt_tokens or 0)
        self.generation_cached_input_tokens += int(cached_tokens or 0)
        self.generation_output_tokens += int(completion_tokens or 0)

    def add_embedding(self, tokens: int) -> None:
        self.embedding_requests += 1
        self.embedding_tokens += int(tokens or 0)

    @property
    def called_api(self) -> bool:
        return self.generation_requests > 0 or self.embedding_requests > 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _model_prices(cfg: dict[str, Any], model: str) -> dict[str, float]:
    table = cfg.get("pricing") or {}
    return table.get(model) or {}


def compute_cost(
    cfg: dict[str, Any], usage: Usage,
) -> tuple[float | None, dict[str, Any]]:
    """(총비용 USD, 상세). 단가가 없으면 그 항목은 None으로 남기고 지어내지 않는다.

    생성 비용과 임베딩 비용을 항상 분리해서 돌려준다.
    """
    unit = cfg.get("pricing_unit", PRICING_UNIT)
    if unit != PRICING_UNIT:
        raise PricingError(
            f"pricing_unit이 {unit!r}입니다. 이 코드는 {PRICING_UNIT!r}(100만 토큰당 달러)만 "
            f"계산합니다 — 단위를 섞지 마세요."
        )

    gen_model = cfg.get("generation_model") or ""
    emb_model = cfg.get("embedding_model") or ""
    gen_p = _model_prices(cfg, gen_model)
    emb_p = _model_prices(cfg, emb_model)

    generation_cost: float | None = None
    if usage.generation_requests:
        if gen_p.get("input_per_1m") is not None and gen_p.get("output_per_1m") is not None:
            cached = usage.generation_cached_input_tokens
            fresh_input = max(usage.generation_input_tokens - cached, 0)
            cached_rate = gen_p.get("cached_input_per_1m", gen_p["input_per_1m"])
            generation_cost = (
                fresh_input / _PER * gen_p["input_per_1m"]
                + cached / _PER * cached_rate
                + usage.generation_output_tokens / _PER * gen_p["output_per_1m"]
            )
    else:
        # 추출표·선별표만 쓴 문항 — 생성 API를 안 불렀으니 생성 비용은 0
        generation_cost = 0.0

    embedding_cost: float | None = None
    if usage.embedding_requests:
        if emb_p.get("input_per_1m") is not None:
            embedding_cost = usage.embedding_tokens / _PER * emb_p["input_per_1m"]
    else:
        embedding_cost = 0.0

    total: float | None
    if generation_cost is None or embedding_cost is None:
        total = None
    else:
        total = round(generation_cost + embedding_cost, 10)

    detail = {
        "pricing_unit": unit,
        "generation_model": gen_model,
        "embedding_model": emb_model,
        "generation_cost_usd": (
            None if generation_cost is None else round(generation_cost, 10)
        ),
        "embedding_cost_usd": (
            None if embedding_cost is None else round(embedding_cost, 10)
        ),
        "usage": usage.as_dict(),
        "rates": {"generation": gen_p, "embedding": emb_p},
    }
    return total, detail
