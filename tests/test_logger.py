import sys
from types import SimpleNamespace

from loguru import logger

from datenwissenschaften import logger as logger_module
from datenwissenschaften.logger import setup_logging


def _reset_logger() -> None:
    logger.remove()
    logger.configure(patcher=None)
    logger.add(sys.stderr)


def test_setup_logging_uses_explicit_level_without_loading_config(monkeypatch):
    def _fail_load_config(path):
        raise AssertionError("load_config should not be called when level is explicit")

    monkeypatch.setattr(logger_module, "load_config", _fail_load_config)

    try:
        setup_logging("info")
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
        try:
            logger.info("hello-from-test")
        finally:
            logger.remove(sink)
    finally:
        _reset_logger()

    assert any("hello-from-test" in message for message in messages)


def test_setup_logging_loads_level_from_config_when_omitted(monkeypatch):
    monkeypatch.setattr(logger_module, "load_config", lambda path: SimpleNamespace(log_level="warning"))

    try:
        setup_logging(config_path="unused.yaml")
    finally:
        _reset_logger()


def test_runtime_context_reports_game_and_savestate(monkeypatch):
    runtime = SimpleNamespace(game="Game", savestate="Level1")
    monkeypatch.setattr("datenwissenschaften.runtime.get_runtime", lambda: runtime)

    assert logger_module._runtime_context() == "[game=Game state=Level1]"


def test_runtime_context_falls_back_when_runtime_not_configured(monkeypatch):
    def _raise():
        raise RuntimeError("not configured")

    monkeypatch.setattr("datenwissenschaften.runtime.get_runtime", _raise)

    assert logger_module._runtime_context() == "[game=- state=-]"


def test_add_runtime_context_mutates_record_extra(monkeypatch):
    runtime = SimpleNamespace(game="Game", savestate="Level1")
    monkeypatch.setattr("datenwissenschaften.runtime.get_runtime", lambda: runtime)
    record = {"extra": {}}

    logger_module._add_runtime_context(record)

    assert record["extra"]["runtime_context"] == "[game=Game state=Level1]"
