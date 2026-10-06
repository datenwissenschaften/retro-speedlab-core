import time
from pathlib import Path

from datenwissenschaften.training import lab_run as lab_run_module
from datenwissenschaften.training.lab_run import LabRun


def test_only_a_marker_with_a_future_deadline_is_an_active_lab_run(tmp_path: Path):
    marker = tmp_path / ".lab-run"
    run = LabRun(marker)

    assert run.deadline() is None
    marker.write_text(f"{time.time() - 60:.0f}\n")
    assert run.deadline() is None
    marker.write_text(f"{time.time() + 600:.0f}\n")
    assert run.deadline() is not None


def test_training_waits_until_the_lab_run_removes_its_marker(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    marker.write_text(f"{time.time() + 600:.0f}\n")
    published, sleeps = [], []
    monkeypatch.setattr(lab_run_module, "publish_metadata", lambda section, values: published.append(values["active"]))
    monkeypatch.setattr(lab_run_module, "consume_model_reset", lambda: None)

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            marker.unlink()

    monkeypatch.setattr(lab_run_module.time, "sleep", sleep)

    assert LabRun(marker).wait() is None
    assert (published, len(sleeps)) == ([True, True, False], 2)


def test_a_model_reset_during_the_pause_is_handed_back(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    marker.write_text(f"{time.time() + 600:.0f}\n")
    monkeypatch.setattr(lab_run_module, "publish_metadata", lambda section, values: None)
    monkeypatch.setattr(lab_run_module, "consume_model_reset", lambda: "reset")

    assert LabRun(marker).wait() == "reset"


def test_without_a_lab_run_the_stream_is_told_training_runs(tmp_path: Path, monkeypatch):
    published = []
    monkeypatch.setattr(lab_run_module, "publish_metadata", lambda section, values: published.append(values))

    assert LabRun(tmp_path / ".lab-run").wait() is None
    assert published == [{"active": False, "until": None}]
