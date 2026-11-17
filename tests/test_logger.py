import json

import sentry_sdk
from loguru import logger
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from datenwissenschaften.logger import FILTERED, secret_values, setup_logging

SECRET = "upload-key-4f2b9c"


def test_setup_logging_filters_below_the_configured_level(capsys):
    setup_logging("WARNING")

    logger.info("hidden")
    logger.warning("visible")

    output = capsys.readouterr().err
    assert "visible" in output
    assert "hidden" not in output


class Recorded(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.events.append(json.dumps(envelope.get_event()))


def leak(settings: str) -> None:
    raise RuntimeError(f"upload failed for {settings}")


def test_bugsink_events_carry_no_secret_and_no_frame_locals(monkeypatch):
    monkeypatch.setenv("BUGSINK_DSN", "https://public@bugs.invalid/1")
    monkeypatch.setenv("UPLOAD_API_KEY", SECRET)
    setup_logging("WARNING")
    recorded = Recorded()
    sentry_sdk.get_client().transport = recorded
    try:
        leak(f"Upload(api_key='{SECRET}')")
    except RuntimeError:
        sentry_sdk.capture_exception()
    sentry_sdk.get_client().close()

    assert len(recorded.events) == 1
    assert SECRET not in recorded.events[0]
    assert FILTERED in recorded.events[0]
    assert '"vars"' not in recorded.events[0]


def test_secret_values_are_the_long_values_of_secret_names():
    environment = {"UPLOAD_API_KEY": SECRET, "GITHUB_TOKEN": "short", "FRONTEND_URL": "https://retrospeedlab.com"}

    assert secret_values(environment) == [SECRET]
