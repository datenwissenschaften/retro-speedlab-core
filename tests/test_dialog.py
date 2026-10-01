from types import SimpleNamespace

import httpx
import pytest

from datenwissenschaften.training import dialog as dialog_module
from datenwissenschaften.training.dialog import SECONDS_BETWEEN_LINES, DialogWriter


class InlineThread:
    def __init__(self, target, args, name, daemon) -> None:
        self.target, self.args = target, args

    def start(self) -> None:
        self.target(*self.args)


def _reply(content: str) -> SimpleNamespace:
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"choices": [{"message": {"content": content}}]})


@pytest.fixture
def writer(monkeypatch) -> DialogWriter:
    monkeypatch.setattr(dialog_module.threading, "Thread", InlineThread)
    return DialogWriter("Retra", "SnakeRattleNRoll-Nes-v0", ("test/model:free", "test/fallback:free"), "key")


def test_a_line_is_written_in_the_persona_voice_and_rate_limited(writer, monkeypatch):
    requests, lines = [], []

    def post(url, **kwargs):
        requests.append(kwargs)
        return _reply(" Got it! ")

    monkeypatch.setattr(dialog_module.httpx, "post", post)
    clock = iter([100.0, 101.0, 100.0 + SECONDS_BETWEEN_LINES])
    monkeypatch.setattr(dialog_module.time, "monotonic", lambda: next(clock))

    for _ in range(3):
        writer.comment("Find Scale done.", lines.append)

    assert lines == ["Got it!", "Got it!"]
    assert requests[0]["json"]["models"] == ["test/model:free", "test/fallback:free"]
    assert requests[0]["json"]["reasoning"] == {"enabled": False}
    assert "Retra" in requests[0]["json"]["messages"][0]["content"]
    assert requests[0]["headers"] == {"Authorization": "Bearer key"}


def test_a_failed_request_says_nothing_and_frees_the_writer(writer, monkeypatch):
    def fail(url, **kwargs):
        raise httpx.ConnectError("offline")

    lines = []
    monkeypatch.setattr(dialog_module.httpx, "post", fail)

    writer.comment("Lives down.", lines.append)

    assert lines == []
    assert writer.busy is False
