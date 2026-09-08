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
    """개발용 더미 provider — 실제 심판이 아니다.

    ★usable_for_final=False 를 클래스 속성으로 노출한다. 이게 없으면 채점기가
      "use_stub_judge=false 니까 실제 심판" 이라고 오판하고, mock 이 돌려주는
      {"status":"MOCK"} 를 파싱 실패 → 0점으로 조용히 처리한다. 그러면 실제로는
      아무도 채점하지 않은 점수가 '심판 점수'로 보고된다.
    """

    name: str = "mock"
    usable_for_final: bool = False

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
    """실제 모델 호출 provider."""


    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 120
    temperature: float | None = 0.0
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
            "messages": [
                {
                    "role": "system",
                    "content": "You are an evaluation judge. Follow the provided rubric exactly.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            # GPT-5 mini/nano는 사용자 지정 temperature를 거부한다. 그 오류일 때만
            # temperature를 빼고 한 번 재시도하며, 다른 400 오류는 그대로 알린다.
            if (
                response.status_code == 400
                and "temperature" in payload
                and "temperature" in response.text.lower()
            ):
                payload.pop("temperature")
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
