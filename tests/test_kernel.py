from pathlib import Path

import numpy as np
import pytest
import walk_kernel
from fakes import write_config
from speedlab_engine.cartridge import BUTTONS, RAM_SIZE
from test_demo_search import WRAPPER

from datenwissenschaften.kernel import batch as kernel_batch
from datenwissenschaften.kernel.batch import Batch, BeamKernel
from datenwissenschaften.kernel.compiler import Compiler
from datenwissenschaften.kernel.expand import STAY, expander
from datenwissenschaften.speedrun import workers
from datenwissenschaften.speedrun.beam import beam_search, macros, packed

SECONDS = 60.0
HOLD = 8


def test_a_batch_steps_every_node_with_every_macro_like_the_cartridge(tmp_path: Path):
    runner = workers.worker_runner(WRAPPER, write_config(tmp_path))
    start = runner.power_on()
    moves = macros(runner)

    found = Batch(walk_kernel, BeamKernel.CPU).expand(
        np.frombuffer(start.emulator_state, np.uint8)[np.newaxis],
        np.zeros((1, walk_kernel.MEMORY), np.int64),
        *packed(moves),
        0,
    )

    for index, macro in enumerate(moves):
        runner.restore(start)
        rewards = [runner.press(buttons)[2] for buttons in macro]
        assert found.rams[index].tobytes() == runner.snapshot()
        assert (found.rewards[index], found.outcomes[index], found.frames[index]) == (sum(rewards), STAY, len(macro))


def test_the_beam_scout_finds_the_same_crossing_on_the_kernel_as_on_the_cartridge(tmp_path: Path):
    runner = workers.worker_runner(WRAPPER, write_config(tmp_path))

    stepped = beam_search(runner, runner.power_on(), SECONDS, None, None)
    batched = beam_search(runner, runner.power_on(), SECONDS, None, Batch(walk_kernel, BeamKernel.CPU))

    assert stepped.crossings and len(stepped.crossings) == len(batched.crossings)
    for first, second in zip(stepped.crossings, batched.crossings, strict=True):
        assert np.array_equal(first.pressed, second.pressed) and first.end == second.end


def test_the_beam_worker_runs_the_kernel_its_config_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = write_config(tmp_path)
    config.write_text(config.read_text().replace("beam_kernel: cartridge", "beam_kernel: cpu"))
    monkeypatch.setattr(kernel_batch, "KERNEL", "walk_kernel")
    start = workers.worker_runner(WRAPPER, config).power_on()

    searched = workers.beam(WRAPPER, config, start, 1, SECONDS, tmp_path / "live").searched

    assert workers.worker_batch(config) is not None
    assert searched.crossings and searched.crossings[0].won


def test_a_kernel_that_writes_past_a_byte_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(walk_kernel, "GOAL_X", walk_kernel.GOAL_X * 2)
    ram = np.zeros((1, RAM_SIZE), np.uint8)
    ram[0, walk_kernel.X] = 250
    right = np.zeros((1, len(BUTTONS)), np.int8)
    right[0, walk_kernel.RIGHT] = 1

    with pytest.raises(RuntimeError, match="outside a byte"):
        carried = np.zeros((1, walk_kernel.MEMORY), np.int64)
        Batch(walk_kernel, BeamKernel.CPU).expand(ram, carried, *packed([np.repeat(right, 8, axis=0)]), 0)


def test_the_expand_body_runs_as_plain_python():
    expand = expander(walk_kernel.step, walk_kernel.prime, walk_kernel.judge)
    rams, carried = np.zeros((1, RAM_SIZE), np.uint8), np.zeros((1, walk_kernel.MEMORY), np.int64)
    macros_array = np.zeros((1, HOLD, len(BUTTONS)), np.int32)
    macros_array[0, :, walk_kernel.RIGHT] = 1
    wide, found = np.zeros((1, RAM_SIZE), np.int32), np.zeros((1, RAM_SIZE), np.uint8)
    working = (np.zeros((1, 1), np.int64), np.zeros((1, 1), np.int64))
    rewards, outcomes, frames = (np.zeros(1, np.int64) for _ in range(3))

    expand(
        0,
        rams,
        carried,
        macros_array,
        np.array([HOLD]),
        0,
        walk_kernel.DATA,
        wide,
        found,
        *working,
        rewards,
        outcomes,
        frames,
    )

    assert (found[0, walk_kernel.X], rewards[0], outcomes[0], frames[0]) == (HOLD, HOLD, STAY, HOLD)


def test_a_batch_runs_on_a_kernel_never_on_the_cartridge():
    with pytest.raises(ValueError, match="cpu or cuda"):
        Batch(walk_kernel, BeamKernel.CARTRIDGE)


def test_a_kernel_function_must_not_be_a_closure():
    reach = 1

    def closure() -> int:
        return reach

    with pytest.raises(TypeError, match="closure"):
        Compiler(__name__, lambda function: function).function(closure)
