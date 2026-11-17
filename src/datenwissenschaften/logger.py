import json
import os
import re
import sys
from collections.abc import Mapping
from importlib.metadata import version

import sentry_sdk
from loguru import logger
from sentry_sdk.integrations.loguru import LoguruIntegration
from sentry_sdk.scrubber import EventScrubber
from sentry_sdk.types import Event

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
DSN_VARIABLE = "BUGSINK_DSN"
PACKAGE = "datenwissenschaften"
RELEASE_NAME = "retro-speedlab-core"
EVENT_FORMAT = "{name}:{function} - {message}"
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|DSN|JWT|DATABASE_URL", re.IGNORECASE)
SHORTEST_SECRET = 8
FILTERED = "[Filtered]"


def secret_values(environment: Mapping[str, str]) -> list[str]:
    return [value for name, value in environment.items() if SECRET_NAME.search(name) and len(value) >= SHORTEST_SECRET]


def scrubbed(event: Event, secrets: list[str]) -> Event:
    text = json.dumps(event)
    for secret in secrets:
        text = text.replace(json.dumps(secret)[1:-1], FILTERED)
    return json.loads(text)


def setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level=level)
    if DSN_VARIABLE in os.environ:
        secrets = secret_values(os.environ)
        sentry_sdk.init(
            dsn=os.environ[DSN_VARIABLE],
            release=f"{RELEASE_NAME}@{version(PACKAGE)}",
            integrations=[LoguruIntegration(event_format=EVENT_FORMAT, breadcrumb_format=EVENT_FORMAT)],
            traces_sample_rate=0,
            include_local_variables=False,
            event_scrubber=EventScrubber(recursive=True),
            before_send=lambda event, hint: scrubbed(event, secrets),
        )
