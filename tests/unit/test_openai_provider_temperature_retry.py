"""채점기 v3: temperature 비지원 응답만 안전하게 한 번 재시도한다."""

from __future__ import annotations

import pytest

import grader.providers as providers
from grader.providers import OpenAICompatibleProvider


class _Response:
    def __init__(self, status_code: int, text: str = "", content: str = "ok"):
        self.status_code = status_code
        self.text = text
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}: {self.text}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _Client:
    responses: list[_Response] = []
    calls: list[dict] = []

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, _url, *, headers, json):
        self.calls.append({"headers": dict(headers), "json": dict(json)})
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _fake_httpx(monkeypatch):
    _Client.responses = []
    _Client.calls = []
    monkeypatch.setattr(providers.httpx, "Client", _Client)


def _provider(temperature=0.0):
    return OpenAICompatibleProvider(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5-mini",
        temperature=temperature,
    )


def test_temperature_rejection_retries_once_without_temperature():
    _Client.responses = [
        _Response(400, "Unsupported parameter: temperature"),
        _Response(200, content='{"score": 1}'),
    ]

    result = _provider().judge("judge this")

    assert result["raw_text"] == '{"score": 1}'
    assert len(_Client.calls) == 2
    assert _Client.calls[0]["json"]["temperature"] == 0.0
    assert "temperature" not in _Client.calls[1]["json"]


def test_unrelated_bad_request_is_not_retried():
    _Client.responses = [_Response(400, "Invalid model")]

    with pytest.raises(RuntimeError, match="Invalid model"):
        _provider().judge("judge this")

    assert len(_Client.calls) == 1


def test_none_temperature_is_omitted_without_retry():
    _Client.responses = [_Response(200)]

    _provider(temperature=None).judge("judge this")

    assert len(_Client.calls) == 1
    assert "temperature" not in _Client.calls[0]["json"]


def test_supported_temperature_success_is_not_retried():
    _Client.responses = [_Response(200)]

    _provider(temperature=0.7).judge("judge this")

    assert len(_Client.calls) == 1
    assert _Client.calls[0]["json"]["temperature"] == 0.7
