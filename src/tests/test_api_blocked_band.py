#@title 유료 API 차단 대역 — 가짜 성공을 만들지 않는다 (§11)
#@markdown 생성·임베딩을 차단한 실행에서 API 가 필요한 문항은 'api_blocked' 로 기록하고,
#@markdown 시스템 오류와 분리해서 센다. 답을 지어내지 않는다.
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_RUN_EVAL = Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval_under_test", _RUN_EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_api_blocked_is_a_typed_signal_not_a_generic_error():
    run_eval = _load_run_eval()
    err = run_eval.ApiBlocked("generation")
    assert err.stage == "generation"
    assert "api_blocked" in str(err)
    assert isinstance(err, RuntimeError)


def test_blocked_mode_is_an_explicit_cli_choice():
    text = _RUN_EVAL.read_text(encoding="utf-8")
    assert '"--api-mode"' in text
    assert 'choices=("live", "blocked")' in text
    assert 'default="live"' in text          # 기본은 실제 실행 — 조용히 차단하지 않는다


def test_blocked_response_records_the_reason_and_no_answer(tmp_path):
    """차단된 문항의 응답은 answer=null + failure=api_blocked:* 여야 한다."""
    from grader.models import ModelResponse
    row = {"id": "QA-001", "answer": None, "structured_answer": None, "contexts": [],
           "retrieved": [], "citations": [], "selected_document_ids": [],
           "abstained": None, "route": None, "failure": "api_blocked:embedding",
           "latency_ms": 1.0, "cost_usd": 0.0}
    parsed = ModelResponse.model_validate(row)
    assert parsed.answer is None and parsed.failure == "api_blocked:embedding"


def test_summary_separates_blocked_from_errors():
    text = _RUN_EVAL.read_text(encoding="utf-8")
    assert '"api_blocked_count"' in text and '"error_count"' in text
    assert '"api_blocked_ids"' in text
