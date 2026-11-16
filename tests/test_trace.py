import json
from pathlib import Path

import msgspec
import numpy as np
from speedlab_engine.cartridge import BUTTONS
from speedlab_engine.collection import DEMO_COLLECTION, cartridge_name
from speedlab_engine.console import Console
from typer.testing import CliRunner

from datenwissenschaften.speedrun import trace
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.record import StoredRecord

RIGHT = [int(name == "RIGHT") for name in BUTTONS]
STEPS = 10
START_X = 16
DOT_Y = 120


def walked_record(path: Path) -> Path:
    console = Console(DEMO_COLLECTION, cartridge_name("walk"), False)
    for _ in range(STEPS):
        console.step(np.asarray(RIGHT, dtype=np.int8))
    segment = Segment("Walk", [RIGHT] * STEPS, console.em.get_state(), False)
    path.write_bytes(msgspec.msgpack.encode(StoredRecord("1.0.0", [segment])))
    return path


def test_a_record_splits_into_pieces_that_replay_into_identical_frame_traces(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(trace, "WRAPPER", "test_demo_search:WalkWrapper")
    cli = CliRunner()

    split = cli.invoke(trace.app, ["pieces", str(DEMO_COLLECTION), "walk", str(walked_record(tmp_path / "record"))])
    piece = json.loads(split.stdout)["pieces"][0]
    replayed = cli.invoke(trace.app, ["frames", str(DEMO_COLLECTION), "walk"], input=json.dumps(piece) + "\n")
    traced = json.loads(replayed.stdout)

    assert (piece["entered"], piece["reached"], piece["frames"]) == ("Walk", "Walk", STEPS)
    assert traced["identical"] and traced["labels"] == ["X"]
    assert traced["frames"][-1][:4] == [STEPS, 0, START_X + STEPS, DOT_Y]
