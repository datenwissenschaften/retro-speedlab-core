from loguru import logger

from datenwissenschaften.logger import setup_logging


def test_setup_logging_filters_below_the_configured_level(capsys):
    setup_logging("WARNING")

    logger.info("hidden")
    logger.warning("visible")

    output = capsys.readouterr().err
    assert "visible" in output
    assert "hidden" not in output
