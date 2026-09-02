"""
J — 답변 생성. 오픈AI 트랙(2026-08-31 확정), gpt-4o-mini 저비용 시작.
temperature=0 고정(재현성, 4-12 확정). 프롬프트 지시 4종을 항상 포함한다:
① 근거에만 기반해 답하라(충실성) ② 근거에 없으면 모른다고 하라
③ 출처를 표기하라 ④ 형식·톤·길이 요구

프롬프트는 코드에 안 넣고 prompts/generate_v1.txt에서 읽는다(message.txt
4번 확정) — config.py가 base.yaml을 찾는 것과 같은 방식(파일 위치 기준
상위 탐색)이라 저장소 구조가 바뀌어도(src/rag/ vs flat) 그대로 동작한다.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def _find_prompt_file(name: str = "generate_v1.txt") -> Path:
    """generation_client.py 기준 상위 경로 어딘가의 prompts/{name}을 찾는다.
    없으면 조용히 하드코딩 문자열로 대체하지 않고 명시적으로 에러."""
    override = os.environ.get("RAG_PROMPT_DIR")
    if override:
        p = Path(override) / name
        if p.exists():
            return p
        raise RuntimeError(f"RAG_PROMPT_DIR이 가리키는 파일이 없습니다: {p}")

    here = Path(__file__).resolve().parent
    for candidate_root in [here, *here.parents]:
        candidate = candidate_root / "prompts" / name
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"prompts/{name}을 찾지 못했습니다. generation_client.py 기준 상위 경로 "
        f"어디에도 prompts/{name}이 없습니다. RAG_PROMPT_DIR 환경변수로 직접 "
        f"지정하거나 prompts/{name}을 만들어 두세요. 프롬프트를 코드에 다시 "
        f"하드코딩하지 않습니다(message.txt 4번 확정)."
    )


def _load_system_prompt_template() -> str:
    path = _find_prompt_file()
    return path.read_text(encoding="utf-8")


class GenerationClient:
    def __init__(self, cfg: dict[str, Any]):
        if cfg.get("generation_provider") != "openai":
            raise NotImplementedError("이 클라이언트는 오픈AI 트랙 전용입니다.")
        if not cfg.get("data_egress_confirmed", False):
            raise RuntimeError(
                "data_egress_confirmed=false — 데이터 반출 확인이 안 끝났습니다."
            )
        if OpenAI is None:
            raise ImportError("pip install openai --break-system-packages")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
        self.client = OpenAI(api_key=api_key)
        model = cfg.get("generation_model")
        if not model:
            raise RuntimeError(
                "generation_model이 비어 있습니다. 코드가 조용히 기본값"
                "(gpt-4o-mini 등)을 선택하지 않습니다 — base.yaml에 사람이 직접 "
                "채워야 합니다(message.txt 4번 확정)."
            )
        self.model = model
        self.temperature = cfg.get("temperature", 0)
        self.max_tokens = cfg.get("max_tokens")
        # 프롬프트 파일이 없으면 실제 생성 시점이 아니라 클라이언트 생성 시점에
        # 바로 에러 — 문제를 최대한 일찍 드러낸다.
        self._system_prompt_template = _load_system_prompt_template()
        # 2026-09-02 추가 — 하루님 responses.jsonl의 cost_usd 계산 기반.
        # 여기서는 토큰 수만 남기고 $ 환산은 안 함(모델별 단가는 팀이
        # 확정해야 할 값이라 코드가 임의로 단가를 지어내지 않는다).
        self.last_usage: dict[str, int] | None = None

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        format_instruction: str = "간결하고 명확하게 답하세요.",
    ) -> str:
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(근거 없음)"
        system = self._system_prompt_template.format(format_instruction=format_instruction)
        user = f"[근거]\n{context}\n\n[질문]\n{question}"

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # ⚠️ 버그 수정(실제 실행 중 발견): GPT-5 계열(gpt-5-mini 등)은
        # temperature=0을 지원 안 함 — API가 "기본값(1)만 지원"이라고 명시적
        # 거부함(invalid_request_error, unsupported_value). 4-12 확정
        # "temperature=0 고정(재현성)"이 이 모델군에선 물리적으로 불가능한
        # 제약이라, gpt-5 계열은 temperature 자체를 안 보내 기본값(1)을
        # 쓰게 한다 — 재현성이 이전만큼 안 보장되는 트레이드오프가 생겼다는
        # 뜻이라 팀 확인 필요.
        if not self.model.startswith("gpt-5"):
            kwargs["temperature"] = self.temperature
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.last_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return resp.choices[0].message.content or ""
