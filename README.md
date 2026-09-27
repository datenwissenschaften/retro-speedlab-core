# Retro Speedlab Core

[![CI](https://github.com/datenwissenschaften/retro-speedlab-core/actions/workflows/ci.yml/badge.svg)](https://github.com/datenwissenschaften/retro-speedlab-core/actions/workflows/ci.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg) ![Decisions](https://img.shields.io/badge/decisions-Laya-6f42c1.svg) [![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE) ![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64.svg) [![codecov](https://codecov.io/gh/datenwissenschaften/retro-speedlab-core/graph/badge.svg)](https://codecov.io/gh/datenwissenschaften/retro-speedlab-core)

Retro game agents built entirely on [Laya](https://huggingface.co/convaiinnovations/laya),
a non-autoregressive decision model. Laya reads the game state as text, answers
the active state's question with calibrated probabilities over named actions,
and learns from rewards by fine-tuning all of its weights.

Retro Speedlab Core is the engine behind
[Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab), which
provides game packages and runnable entry points.

## How it works

1. **The game speaks text.** A game package maps RAM addresses to a `RamInfo`
   and returns a readable `describe()` dict, for example
   `{"score": 120, "lives": 3, "ship_position_percent_from_left": 40}`.
2. **Every state asks a question.** Each `State` in the state machine has a
   `description`, which is the question Laya answers in that state. A state
   transition changes the question.
3. **Every action has a meaning.** `action_table` has the shape
   `(actions, frames, buttons)`: each action is the button sequence played for
   one decision, so a game can tap a button instead of holding it.
   `action_descriptions` name each action, and Laya scores each description.
4. **Laya decides.** One forward pass returns a probability per action; the
   agent samples from it.
5. **Laya learns.** Every 64 decisions the discounted rewards-to-go are
   normalized across the group, and group-relative policy gradients (Laya's own
   RLCD recipe) update the encoder and the decision head. A KL penalty keeps
   every update close to the policy that played, so Laya cannot collapse onto
   one action. There is no critic.
   bf16 autocast, gradient checkpointing, and 8-bit AdamW keep the full model
   trainable on a 6 GB GPU.

## Vision

States receive every frame. `datenwissenschaften.vision` provides a
`TemplateDetector` (template images from the game's `assets/`), a
`ColorBlobDetector` for sprites with a distinctive color, and
`describe_target`, which turns detections into text such as
`{"visible": true, "direction": "up-left", "distance": 128}`. A state returns
that from `describe()` so it becomes part of the game state Laya reads, and
from `detections()` so the stream draws the boxes on the replayed video.

The reverse curriculum, automatic savestates, BK2 recordings, best-run videos,
signed run uploads, file-backed telemetry, and the dashboard work around that
loop.

## Game packages

A game package subclasses `StateMachineGymWrapper` and starts training with
`LayaTrainer`:

```python
from pathlib import Path

import numpy as np

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.training.trainer import LayaTrainer


class AirstrikerWrapper(StateMachineGymWrapper[AirstrikerRam]):
    start_state_cls = SurviveAndScore
    state_classes = (SurviveAndScore,)
    ram_info_cls = AirstrikerRam
    action_table = np.array([[[1] + [0] * 11, [0] * 12, [0] * 12, [0] * 12]], dtype=np.int8)
    action_descriptions = {"fire": "hold position and shoot straight ahead"}


LayaTrainer(AirstrikerWrapper, Path("config.yaml")).train()
```

## Dashboard and stream view

With `ui.enable: true` the dashboard runs at `http://127.0.0.1:18080`.
`/stream` is a 1920×1080 page for OBS. Training records every frame with the
decision behind it, and the page replays that recording at a steady frame rate
a few seconds behind live, so learning pauses never freeze the stream.

## Installation

Retro Speedlab Core requires **Python 3.12**. Training state lives in a JSON file (`paths.database`).

```bash
git clone https://github.com/datenwissenschaften/retro-speedlab-core.git
cd retro-speedlab-core
poetry install
cp config.example.yaml config.yaml
```

The published package and import path are `datenwissenschaften`.

## Configuration

Every key in `config.example.yaml` is required; missing or malformed values
raise immediately. `laya.checkpoint` selects the Laya model on the Hugging Face
Hub or a local directory.

## Development

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest --cov
```

After modifying the Vue dashboard, rebuild its assets:

```bash
cd src/datenwissenschaften/ui/frontend
npm ci
npm run build
```

## Limitations

- Training uses one emulator; each update learns from 64 decisions.
- Laya only knows what the game package describes. Unmapped RAM, such as enemy
  positions, is invisible to it.
- A full checkpoint stores all Laya weights (about 1.7 GB).
- Binding the dashboard to `0.0.0.0` exposes it without authentication.

## License

Copyright © datenwissenschaften contributors.

Distributed under the [GNU General Public License v3.0](LICENSE).
