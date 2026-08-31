"""
J — 답변 생성. 오픈AI 트랙(2026-08-31 확정), gpt-4o-mini 저비용 시작.
temperature=0 고정(재현성, 4-12 확정). 프롬프트 지시 4종을 항상 포함한다:
① 근거에만 기반해 답하라(충실성) ② 근거에 없으면 모른다고 하라
③ 출처를 표기하라 ④ 형식·톤·길이 요구
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


SYSTEM_PROMPT_TEMPLATE = """당신은 RFP(제안요청서) 문서를 근거로 답하는 조수입니다.
반드시 아래 원칙을 지키세요:
1. 제공된 근거(context)에만 기반해 답하세요. 근거에 없는 내용을 지어내지 마세요.
2. 근거에서 답을 찾을 수 없으면 "확인할 수 없습니다"라고 답하세요. 억지로 답을 만들지 마세요.
3. 답변에 반드시 출처(문서명, 장절 또는 표 번호)를 표기하세요.
4. {format_instruction}
"""


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

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        format_instruction: str = "간결하고 명확하게 답하세요.",
    ) -> str:
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(근거 없음)"
        system = SYSTEM_PROMPT_TEMPLATE.format(format_instruction=format_instruction)
        user = f"[근거]\n{context}\n\n[질문]\n{question}"

        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
