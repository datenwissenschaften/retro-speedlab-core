import threading
from pathlib import Path

from datenwissenschaften.environment.factory import POWER_ON
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tas.run import speedrun_forever
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.lab_run import POLL_SECONDS, LabRun
from datenwissenschaften.ui.relay import LiveRelay
from datenwissenschaften.workflow.environment import speedlab


def start_relay(settings: RetroSpeedlabConfig) -> None:
    if not settings.ui.enabled:
        return
    if settings.upload.api_key is None:
        raise RuntimeError("ui.enable relays the stream to the backend and needs upload.api_key.")
    LiveRelay(
        settings.upload.url, settings.upload.api_key, settings.ui, RunContext(settings, POWER_ON).record_root
    ).start()


def watch_lab(lab: LabRun, stopped: threading.Event) -> None:
    while not stopped.wait(POLL_SECONDS):
        lab.publish()


@speedlab.task
def speedrun(wrapper: str, config: str) -> None:
    settings = load_config(Path(config))
    start_relay(settings)
    lab = LabRun(settings.paths.lab_run_marker)
    threading.Thread(target=watch_lab, args=(lab, threading.Event()), name="lab-status", daemon=True).start()
    speedrun_forever(wrapper, Path(config))
