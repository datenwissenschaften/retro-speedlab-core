# Retro Speedlab Library

**Reinforcement-learning engine for training recurrent agents on classic video games.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

The reusable reinforcement-learning engine behind
[Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab).

It combines **visual observations, emulator RAM, recurrent PPO, Random Network
Distillation (RND), vectorized environments, resumable training, replay capture,
and live telemetry** in a reproducible Python training system.

### Highlights

- Recurrent **CNN-LSTM PPO** for partially observable environments
- Multi-input policies combining **RGB frames and emulator RAM**
- Adaptive **Random Network Distillation (RND)** for sparse-reward exploration
- Parallel vectorized emulator environments
- Automatic worker selection, CUDA tuning, and CPU fallback
- Atomic, resumable checkpoints including exploration state
- `.bk2` episode recording
- Live Vue-based training telemetry
- Reusable abstractions for RAM, game states, rewards, and actions

> **Looking for game runners, end-to-end examples, and user-facing
> documentation?**
> Start with [Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab).

## Architecture

A training run combines game-specific definitions with reusable environment,
model, training, persistence, and monitoring components.

```text
                         Retro Speedlab
                              │
                    ┌─────────▼─────────┐
                    │  Game Definition  │
                    │                   │
                    │ RAM · Rewards ·   │
                    │ States · Actions  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Vectorized Retro  │
                    │   Environments    │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
             RGB Frames                 RAM State
                 │                         │
            CNN Encoder                 Encoder
                 │                         │
                 └────────────┬────────────┘
                              │
                             LSTM
                              │
                    ┌─────────▼─────────┐
                    │  Recurrent PPO    │
                    │      + RND        │
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
          Checkpoints      Replays       Telemetry
                                             │
                                     ┌───────▼───────┐
                                     │ Live Dashboard │
                                     └───────────────┘
```

At runtime:

1. A game package defines RAM structures, training states, rewards, and action
   translation.
2. The environment factory creates vectorized emulator workers and processed
   visual observations.
3. A model builder creates or restores the selected policy.
4. The trainer coordinates learning, checkpoints, replay capture, telemetry,
   and optional uploads.
5. The dashboard exposes the active run without coupling the learner to a
   separate monitoring service.

## Why this engine

### Exploration for sparse rewards

`AdaptiveRecurrentRNDModel` combines visual frames, normalized RAM, temporal
memory, and recurrent PPO with normalized, clipped, and annealed Random Network
Distillation.

RND provides an intrinsic reward for novel observations, encouraging exploration
when useful external rewards are rare. Its influence decays during training so
that learned external rewards increasingly determine policy behavior.

### Temporal and state-aware policies

Visual observations are processed alongside structured emulator RAM. An LSTM
maintains temporal context for environments where the current observation alone
does not fully describe the game state.

Environment wrappers use RGB observations and one emulator step per selected
action as fixed engine defaults rather than game-level options.

### Efficient execution

Training supports vectorized environments, automatic worker selection, CUDA
tuning, and CPU fallback.

The engine is designed to separate game-specific definitions from reusable
training infrastructure, allowing the same training pipeline to operate across
different environments.

### Reliable, resumable training

Training state is persisted through atomic checkpoints.

For the adaptive recurrent RND model, checkpoints preserve:

- policy state
- RND predictor
- fixed RND target
- optimizer state
- reward statistics
- adaptation state
- annealing progress

This allows interrupted experiments to resume without silently resetting the
exploration process.

### Operational visibility

Training telemetry is exposed through a local browser dashboard without
requiring a separate monitoring service.

Episode recordings can additionally be persisted as `.bk2` replays for later
inspection.

## Models

### `AdaptiveRecurrentRNDModel`

The recommended model for sparse-reward games and partially observable
environments.

It combines:

- visual CNN encoding
- normalized emulator RAM
- LSTM temporal memory
- recurrent PPO
- adaptive intrinsic RND exploration

The model automatically configures parameters including score-staleness windows,
missing-win windows, exploration multipliers, entropy, learning rate, clip
range, and RND update pressure using characteristics of the action space,
rollout size, training horizon, fitness volatility, score staleness, and win
staleness.

The default profile uses:

| Parameter | Default |
| --- | ---: |
| Rollout length | `512` steps |
| LSTM size | `256` units |
| `gamma` | `0.999` |
| `gae_lambda` | `0.98` |
| RND decay horizon | `5,000,000` steps |

These defaults preserve more temporal context and delayed reward information
than a shorter arcade-oriented baseline while retaining conservative PPO
updates.

### Custom Stable-Baselines3 model

Experiments requiring a standard Stable-Baselines3 algorithm can integrate
through the same model builder, trainer, callbacks, and dashboard
infrastructure.

| Model | Best suited to |
| --- | --- |
| `AdaptiveRecurrentRNDModel` | Sparse-reward games and partially observable state |
| Custom SB3 model | Experiments using standard Stable-Baselines3 algorithms |

## Installation

Retro Speedlab Library requires **Python 3.12**.

Install the published package:

```bash
pip install datenwissenschaften
```

For local development:

```bash
git clone https://github.com/datenwissenschaften/retro-speedlab-library.git
cd retro-speedlab-library

poetry install
cp config.example.yaml config.yaml
```

> The Python package is currently published as `datenwissenschaften`.

## Training dashboard

Enable the dashboard in the configuration:

```yaml
ui:
  enable: true
```

Then open:

```text
http://127.0.0.1:18080
```

The dashboard exposes live training telemetry without interrupting the learner,
including:

- episode outcomes
- reward distributions
- environment details
- PPO parameters
- RND progress

Dashboard history and non-file training state are persisted to Redis. This
includes best-episode references and metrics, callback state, and target memory.

Best episodes are scoped independently by game identity and game savestate, for
example:

```text
level1-1
```

Model checkpoints and `.bk2` episode recordings remain on disk.

The default Redis connection is:

```text
redis://127.0.0.1:6379/0
```

History keys use:

```text
datenwissenschaften:history
```

The `ui` configuration accepts:

```yaml
ui:
  enable: true
  host: 127.0.0.1
  port: 18080
  max_episodes: 1000
  redis_url: redis://127.0.0.1:6379/0
  history_key_prefix: datenwissenschaften:history
```

Snapshots retain the latest 1,000 episodes by default and include summarized
totals for discarded episodes.

Set `max_episodes` to another positive integer or `null` for unlimited retained
rows.

> Binding the dashboard to `0.0.0.0` makes it reachable from other machines on
> the local network. Use this only on a trusted network.

## Project responsibilities

The library focuses on the reusable reinforcement-learning engine rather than
individual game implementations.

Its responsibilities include:

```text
Environment
├── emulator integration
├── vectorized workers
├── visual observations
└── action translation

Game state
├── RAM models
├── state machines
├── rewards
└── training objectives

Learning
├── CNN encoding
├── recurrent PPO
├── LSTM memory
└── RND exploration

Training
├── model construction
├── checkpointing
├── callbacks
├── replay capture
└── telemetry

Monitoring
├── episode history
├── training metrics
├── Redis persistence
└── Vue dashboard
```

Game runners, end-to-end examples, and higher-level project documentation live
in [Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab).

## Development

Install the development environment:

```bash
poetry install
```

Run the Python quality checks:

```bash
ruff check src
black --check src
python -m compileall -q src
```

After modifying the Vue dashboard frontend, rebuild its assets:

```bash
cd src/datenwissenschaften/ui/frontend
npm ci
npm run build
```

## License

Copyright © datenwissenschaften contributors.

Distributed under the [GNU General Public License v3.0](LICENSE).
