import os
import sys
from importlib.metadata import version

import sentry_sdk
from loguru import logger
from sentry_sdk.integrations.loguru import LoguruIntegration

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
DSN_VARIABLE = "BUGSINK_DSN"
PACKAGE = "datenwissenschaften"
RELEASE_NAME = "retro-speedlab-core"


def setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level=level)
    if DSN_VARIABLE in os.environ:
        sentry_sdk.init(
            dsn=os.environ[DSN_VARIABLE],
            release=f"{RELEASE_NAME}@{version(PACKAGE)}",
            integrations=[LoguruIntegration()],
            traces_sample_rate=0,
        )
