from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import json
import os
import httpx


class JudgeProvider(Protocol):
    name: str

    def judge(self, prompt: str) -> dict[str, Any]:
        ...


@dataclass
class MockJudgeProvider:
    name: str = "mock"

    def judge(self, prompt: str) -> dict[str, Any]:
        # 실제 팀 prompt/출력 계약 연결 전 개발용 provider.
        return {
            "raw_text": json.dumps(
                {"status": "MOCK", "message": "Real judge provider not connected yet."},
                ensure_ascii=False,
            )
        }


@dataclass
class OpenAICompatibleProvider:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 120
    temperature: float = 0.0
    name: str = "openai_compatible"

    def judge(self, prompt: str) -> dict[str, Any]:
        if not self.api_key or not self.model:
            raise RuntimeError(
                "OpenAI-compatible provider를 사용하려면 GRADER_API_KEY와 GRADER_MODEL이 필요합니다."
            )

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an evaluation judge. Follow the provided rubric exactly.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"]["content"]
        return {"raw_text": text, "provider_response": data}


def build_provider(runtime) -> JudgeProvider:
    provider = os.getenv("GRADER_PROVIDER", "mock").lower()

    if provider == "mock":
        return MockJudgeProvider()

    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            base_url=os.getenv("GRADER_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("GRADER_API_KEY", ""),
            model=os.getenv("GRADER_MODEL", ""),
            timeout_seconds=runtime.timeout_seconds,
            temperature=runtime.temperature,
        )

    raise ValueError(f"지원하지 않는 GRADER_PROVIDER: {provider}")
