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
                                     ┌───────▼────────┐
                                     │ Live Dashboard │
                                     └────────────────┘
