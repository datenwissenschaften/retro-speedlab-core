import threading
from pathlib import Path

from datenwissenschaften.lab.run import publish_lab_run
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tas.live import ATTEMPT_FILE, LIVE_DIR
from datenwissenschaften.tas.paths import tas_dir
from datenwissenschaften.tas.run import speedrun_forever
from datenwissenschaften.ui.relay import LiveRelay, relay_key
from datenwissenschaften.workflow.environment import speedlab

LAB_POLL_SECONDS = 10.0


def start_relay(settings: RetroSpeedlabConfig) -> None:
    if not settings.ui.enabled:
        return
    attempt = tas_dir(settings) / LIVE_DIR / ATTEMPT_FILE
    LiveRelay(settings.upload.url, relay_key(settings.upload), settings.ui, attempt).start()


def watch_lab(marker: Path, stopped: threading.Event) -> None:
    while not stopped.wait(LAB_POLL_SECONDS):
        publish_lab_run(marker)


@speedlab.task
def speedrun(wrapper: str, config: str) -> None:
    settings = load_config(Path(config))
    start_relay(settings)
    marker = settings.paths.lab_run_marker
    threading.Thread(target=watch_lab, args=(marker, threading.Event()), name="lab-status", daemon=True).start()
    speedrun_forever(wrapper, Path(config))
