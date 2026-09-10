"""One LLM call that chooses exactly one Stage 1 action."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from generation_client import model_rejects_sampling_params
from pricing import Usage
from stage1_plan import Stage1Plan, parse_plan, plan_json_schema

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


class Stage1PlannerError(RuntimeError):
    pass


def _prompt_path(name: str) -> Path:
    here = Path(__file__).resolve().parent
    for root in [here, *here.parents]:
        path = root / "prompts" / name
        if path.exists():
            return path
    raise Stage1PlannerError(f"Stage 1 프롬프트 파일을 찾지 못했습니다: {name}")


def document_catalog(identity, eligible_ids: set[str] | None) -> list[str]:
    if identity is None:
        return []
    ids = sorted(eligible_ids if eligible_ids is not None else identity.document_ids())
    lines: list[str] = []
    for document_id in ids:
        record = identity.get(document_id)
        if record is not None:
            lines.append(f"{document_id} | {record.buyer_org} | {record.project_name}")
    return lines


class Stage1Planner:
    def __init__(self, cfg: dict[str, Any], identity, eligible_ids: set[str] | None):
        if cfg.get("generation_provider") != "openai":
            raise Stage1PlannerError("Stage 1 해석기는 OpenAI 경로만 지원합니다.")
        if not cfg.get("data_egress_confirmed", False):
            raise Stage1PlannerError("data_egress_confirmed=false입니다.")
        if OpenAI is None:
            raise Stage1PlannerError("openai 패키지가 없습니다.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise Stage1PlannerError("OPENAI_API_KEY 환경변수가 없습니다.")
        model = cfg.get("stage1_planner_model")
        prompt_name = cfg.get("stage1_planner_prompt")
        if not model or not prompt_name:
            raise Stage1PlannerError(
                "stage1_planner_model과 stage1_planner_prompt를 명시해야 합니다.")
        self.cfg = cfg
        self.model = str(model)
        self.prompt_file = str(prompt_name)
        self.system_prompt = _prompt_path(self.prompt_file).read_text(encoding="utf-8")
        self.catalog_lines = document_catalog(identity, eligible_ids)
        if not self.catalog_lines:
            raise Stage1PlannerError("질문 해석에 제공할 공식 문서 목록이 비어 있습니다.")
        self.valid_document_ids = [line.split(" | ", 1)[0] for line in self.catalog_lines]
        self.client = OpenAI(max_retries=int(cfg.get("api_max_retries", 0) or 0))
        self.usage = Usage()
        self.last_raw_plan: dict[str, Any] | None = None
        self.calls = 0

    def reset_usage(self) -> None:
        self.usage = Usage()

    def build_messages(self, question: str, session_document_id: str | None) -> list[dict]:
        catalog = "\n".join(self.catalog_lines)
        session = session_document_id or "(없음)"
        user = (
            "[현재 세션 문서 ID]\n" + session
            + "\n\n[공식 문서 목록 — 이 목록에 있는 ID만 선택]\n" + catalog
            + "\n\n[사용자 질문]\n" + question
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def _kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "stage1_one_shot_plan",
                    "strict": True,
                    "schema": plan_json_schema(self.valid_document_ids),
                },
            },
        }
        limit = self.cfg.get("stage1_planner_max_completion_tokens")
        if limit is not None:
            kwargs["max_completion_tokens"] = int(limit)
        if not model_rejects_sampling_params(self.model):
            temperature = self.cfg.get("stage1_planner_temperature")
            if temperature is not None:
                kwargs["temperature"] = temperature
        return kwargs

    def plan(self, question: str, session_document_id: str | None = None) -> Stage1Plan:
        self.calls += 1
        response = self.client.chat.completions.create(
            **self._kwargs(self.build_messages(question, session_document_id)))
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            self.usage.add_generation(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=cached or 0,
            )
        if not response.choices:
            raise Stage1PlannerError("질문 해석 응답에 후보가 없습니다.")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "stop") != "stop":
            raise Stage1PlannerError(
                f"질문 해석이 정상 종료되지 않았습니다: {choice.finish_reason}")
        if getattr(choice.message, "refusal", None):
            raise Stage1PlannerError("질문 해석 요청이 거부됐습니다.")
        try:
            raw = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise Stage1PlannerError("질문 해석 결과가 JSON이 아닙니다.") from exc
        self.last_raw_plan = raw
        return parse_plan(raw)
