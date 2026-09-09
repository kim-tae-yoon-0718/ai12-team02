"""추천 ON/OFF와 현재 3단계 연결 계약을 확인한다."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "company_match"
sys.path.insert(0, str(TOOLS_DIR))
import company_match as C  # noqa: E402
import session_start as S  # noqa: E402


def args(tmp_path, skip_search=False):
    return argparse.Namespace(
        company=None,
        store_dir=str(tmp_path),
        top_n=5,
        urgent_days=14,
        skip_search=skip_search,
    )


def runtime():
    return {
        "table": [],
        "identity": None,
        "registry_scope": None,
        "store": object(),
        "locator": None,
    }


def test_toggle_defaults_on():
    assert S.recommendation_enabled({}) is True


def test_toggle_rejects_string_values():
    with pytest.raises(ValueError):
        S.recommendation_enabled({"company_recommendation_enabled": "false"})


def test_off_bypasses_all_company_functions_and_enters_stage3(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(S, "get_or_register_profile", lambda *_a: pytest.fail("회사 입력 호출"))
    monkeypatch.setattr(C, "match_company", lambda *_a: pytest.fail("추천기 호출"))
    monkeypatch.setattr(S, "run_search_loop", lambda *_a: called.append("stage3"))
    result = S.run_session(
        args(tmp_path), {"company_recommendation_enabled": False}, runtime(),
    )
    assert result == 0
    assert called == ["stage3"]


def test_on_runs_recommendation_then_stage3(monkeypatch, tmp_path):
    events = []
    profile = C.CompanyProfile("회사", region="서울")
    monkeypatch.setattr(
        S, "get_or_register_profile",
        lambda *_a: (events.append("profile") or ("회사", profile)),
    )
    monkeypatch.setattr(
        C, "match_company", lambda *_a: (events.append("match") or []),
    )
    monkeypatch.setattr(
        S, "print_recommendations", lambda *_a: events.append("show"),
    )
    monkeypatch.setattr(
        S, "print_urgent_candidates", lambda *_a: events.append("urgent"),
    )
    monkeypatch.setattr(S, "run_search_loop", lambda *_a: events.append("stage3"))
    S.run_session(args(tmp_path), {"company_recommendation_enabled": True}, runtime())
    assert events == ["profile", "match", "show", "urgent", "stage3"]


def test_skip_search_still_shows_recommendation(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        S, "get_or_register_profile",
        lambda *_a: ("회사", C.CompanyProfile("회사")),
    )
    monkeypatch.setattr(C, "match_company", lambda *_a: [])
    monkeypatch.setattr(S, "print_recommendations", lambda *_a: events.append("show"))
    monkeypatch.setattr(S, "print_urgent_candidates", lambda *_a: None)
    monkeypatch.setattr(S, "run_search_loop", lambda *_a: pytest.fail("검색 루프 호출"))
    S.run_session(
        args(tmp_path, skip_search=True),
        {"company_recommendation_enabled": True},
        runtime(),
    )
    assert events == ["show"]


def test_search_loop_supplies_current_stage2_factory(monkeypatch):
    captured = {}

    class Client:
        def reset_usage(self):
            return None

    class Agent(Client):
        def __init__(self, *_args):
            pass

    monkeypatch.setattr(S.AP, "EmbeddingClient", lambda _cfg: Client())
    monkeypatch.setattr(S.AP, "GenerationClient", lambda _cfg: Client())
    monkeypatch.setattr(S.AP, "Stage1Planner", Agent)
    monkeypatch.setattr(S.AP, "Stage2Agent", Agent)

    def fake_answer(*_args, **kwargs):
        captured.update(kwargs)
        kwargs["get_stage1_planner"]()
        kwargs["get_stage2_agent"]()
        return SimpleNamespace(route_matched_rule="test", session_banner=None, text="답")

    monkeypatch.setattr(S.AP, "answer", fake_answer)
    answers = iter(["질문", ""])
    monkeypatch.setattr("builtins.input", lambda *_a: next(answers))
    rt = runtime()
    S.run_search_loop(rt, {})
    assert callable(captured["get_stage1_planner"])
    assert callable(captured["get_stage2_agent"])
