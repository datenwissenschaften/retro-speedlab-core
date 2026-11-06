# Retro Speedlab Core

[![CI](https://github.com/datenwissenschaften/retro-speedlab-core/actions/workflows/ci.yml/badge.svg)](https://github.com/datenwissenschaften/retro-speedlab-core/actions/workflows/ci.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg) [![Last commit](https://img.shields.io/github/last-commit/datenwissenschaften/retro-speedlab-core)](https://github.com/datenwissenschaften/retro-speedlab-core/commits/main) [![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE) ![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64.svg) [![codecov](https://codecov.io/gh/datenwissenschaften/retro-speedlab-core/graph/badge.svg)](https://codecov.io/gh/datenwissenschaften/retro-speedlab-core)

Tool-assisted speedrun search for classic video games. The engine searches the
fastest movie of buttons to press on every frame from power-on, verifies that it
replays, and streams the run to [Retro Speedlab](https://www.retrospeedlab.com).
Game packages supply the game knowledge (RAM map, states, rewards, actions);
this repository turns it into a speedrun.

## Highlights

- **Power-on start like a real speedrun**: every movie boots the game at its
  title screen; levels are states of one full-game run
- **Every level practised on its own**: the engine keeps the frame before the
  first level change a route makes and loads any level from it; the levels take
  turns of three searches each
- **Three search modes per segment** on many emulator processes: Go-Explore
  walks, a breadth-first frontier search over short input combinations and a
  Monte Carlo tree search
- **Learned guidance**: RAM values never seen before, a move table learned from
  every stored piece, and a PPO policy that trains on the GPU
- **Every segment is verified**: the whole movie replays from power-on before a
  new segment counts
- **Route memory** from the practised levels guides the search towards each
  level's exit
- `.bk2` movie and MP4 video of the verified run
- **Live relay** of the newest verified segment, the run's time, level splits
  and the lab's status to the Retro Speedlab backend
- **Lab runs**: when the search is stuck or stalls, Claude Code improves the
  game package on its own branch

## Architecture

```mermaid
flowchart TD
    Game["Game package<br/>RAM · States · Rewards · Actions"]
    Env["Stable Retro emulator<br/>StateMachineGymWrapper"]
    Search["Segment search<br/>one worker per process"]
    Movie["Verified movie<br/>progress.msgpack · tas.bk2 · screen.mp4"]
    Relay["Live relay<br/>snapshot · whole movie"]
    Lab["Lab run<br/>agent/daily.sh"]

    Game --> Env
    Env --> Search
    Search --> Movie
    Movie --> Relay
    Movie -- "stuck or stalled" --> Lab
    Lab --> Game
```

## Speedrun search

The app's default workflow is `speedrun`: it writes the fastest movie from
power-on for any game package and improves it forever:

```bash
python -m datenwissenschaften.tas.run search <wrapper> <config> <until-state>
python -m datenwissenschaften.tas.run improve <wrapper> <config>
python -m datenwissenschaften.tas.run verify <wrapper> <config> <progress>
```

The run is split at the state machine's transitions. Each segment is searched on
`speedrun.workers` emulator processes over an archive that keeps the fastest
savestate per cell (state, 8 px position, height, airborne and
`RamInfo.dynamics()`, the game's coarse view of whatever moves nearby). Three
modes share each search by time:

- **Go-Explore walks** (60 %) branch from rarely visited cells with short input
  bursts; half the time they start from the tenth of cells that earned the most
  of the segment's reward.
- **Frontier search** (20 %) tries every action with holds of 1, 3 and 8
  decisions, three deep and breadth first, from a chosen cell: the frame-precise
  jumps random walks rarely hit.
- **Monte Carlo tree search** (20 %) expands the moves with the best upper
  confidence bound and scores short rollouts by reward and new cells.

Cells gain weight for RAM byte values never seen before in the segment and for
being close to the exit on the route memory. Walks and rollouts follow a move
table learned from every stored piece (`cache/tas/prior.msgpack`) 30 % of the
time and the PPO policy 20 % of the time. A full archive drops its least
promising cell. Each search process keeps 1000 cells of the last 12 starts it
explored, half with the most reward and half the rarest, and saves them to
`cache/tas/archives/<start>/<process>.msgpack`: a later search from the same
start goes on where the last one stopped, and after a restart the processes'
files are merged. Before it searches a state, the TAS replays the hints for it
from each of its starts in that state: the game package's `hints.json` (written
by the lab, `[{"state": ..., "presses": [[frames, [buttons]]]}]`) and the
shortest way out the PPO policy found (`cache/tas/policy_exits.msgpack`); a hint
that crosses and holds becomes the next piece. A crossing into a later state ends a branch and counts only when
the new state holds; deaths and backward transitions drop it. Of the three
fastest distinct crossings, the one whose next segment is also fastest is kept,
and frames it does not need are shaved off. A segment gets
`speedrun.segment_seconds` per round and twice as long with each retry. Random
numbers live in the savestates, so the search plans against the real RNG and can
wait frames to change it. A segment without a way out is marked stuck in
`cache/tas/stuck.json` (in practice since the state's first failed search
without a find, across the other levels' turns, `cache/tas/failures.json`) and
the way into it is searched again; a search that only ends in a checkpoint counts
as one without a way out too. Checkpoints in a state come only within its first
turn of searches since a piece last left it (practice: one level's turn,
otherwise 3 searches): a chain of checkpoints that has not led out by then is
dropped down to the last piece. Going back drops every checkpoint in a row down
to the last piece as well, and either way that level gets no new checkpoint in
the state it went back from until a piece leaves that state (finding the way
back into it is not enough), also after a restart (`cache/tas/barred.json`).
A checkpoint is the reached cell with the fewest frames left on the route map:
from a start on the map that cell must be at least 60 frames closer to the
exit, from a start off the map any reached map cell will do; otherwise it is
the cell with the most reward. A checkpoint on the map always moves closer to
the exit, so it is also kept after the first turn; of several workers'
checkpoints the one closest to the exit on the map wins, then the most reward.

After every segment the whole movie is replayed from power-on; segments that do
not reproduce are searched again. `cache/tas/progress.msgpack` lets a run
resume, `cache/tas/tas.bk2` is the verified movie, `screen.mp4` its video and
`search.log` the search's log. `metadata.speedrun` tells the stream the run's
time from power-on and from Start, its segments, the level splits (from
`GameWrapper.levels`, with the part before the first level split under the
first state's name, `opening`), each state's `goal`, the game and its record
sources, the engine's events, milestones and time history (`journal.json`) and
the segment being searched. While a segment is searched every worker writes its
cells and frames to `live/<seed>.msgpack` every 2 s and its latest frame at half
size to `live/<seed>.jpg` once a second; the main process folds the reports into
`search` every 5 s and the relay puts a mosaic of the first nine workers' frames
(seed order, row by row, seeds in `X-Seeds`) to `/live/attempts.jpg` once a
second. Whenever the verified movie changes a background thread renders it from
power-on to an H.264 MP4 at twice the NES resolution, puts it to
`/live/movie.mp4` in parts, puts the game memory sampled every 6 frames of the
power-on replay to `/live/movie-ram.json` and publishes `speedrun.movie`: its
version (the first 12 hex digits of its SHA-256), frames, seconds, frame rate
and each segment's state, the state it reaches and its start frame, so the
stream plays the whole run in sync with its splits. When `improve` finds a
faster segment, the old and the new piece are rendered side by side from the
same savestate, put to `/live/compare.mp4` and published as `speedrun.compare`
with the state, both frame counts and the time. `improve` searches every segment
again, bounded by its current frames; a faster crossing replaces it and the rest
is searched anew. `verify` fails unless a progress file still replays with a
game package.

## Route memory

`cache/reference/route.json` lists, per 16 px cell of every beaten level, how
many frames the level's route still needed from there to its exit, and is
rewritten after every level's turn. `cache/reference/tas_route.json` holds the
same for every level of the TASVideos run. The long dark score screens between
levels split the TAS video into its levels; a small CNN, trained on screens of
the stored routes, of each level started by the level loader and of short
random rollouts from up to 2 saved states per 16 px cell of the search's
archives (`cache/tas/archives`, so also where the search is stuck), places the
snake within the known level, and only frames that look like a screen it
trained on count. The frames left run to the level's score screen. The map is
built once, when the file is missing, and only when every level has cells; the
log names each level's video frames and cells. A game package may also ship
`tas_ram_route.json` next to its `config.yaml`: the same rows taken from the RAM
of a TAS replayed in the emulator it was recorded on, exact for every frame. A row
may end in what the TAS had there, in the game's `dynamics()` order (Snake Rattle
'n Roll: weight, pieces on the scale, speed); such a row counts only for a snake
that has at least as much, so the map leads over nibbleys and power-ups too. The
search weights cells on any of these routes by the fewest frames left to the exit.

## PPO policy

With `speedrun.policy`, a separate process trains a policy with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io) PPO on the GPU, always from the start
the search is working on (`cache/tas/policy_start.msgpack`). `policy.env.PolicyEnv` is a
[Gymnasium](https://gymnasium.farama.org) environment around one emulator: RAM in, a button
combination out. Its reward is the state's reward, RAM values never seen on that level, a bonus
for the exit and a penalty for a death; an episode ends on the exit, a death, another state or
after 128 decisions. `speedrun.policy_emulators` environments play in parallel in a
`SubprocVecEnv`. After every round of 1024 decisions per emulator the policy network (two tanh
layers of 256) is exported to `cache/tas/policy.npz`, which the search workers read with numpy,
the shortest episode that left the state is kept for the search to replay, and the round's
reward, exits, entropy and value fit go to Weights & Biases.

## Workflows and the lab

[Flyte](https://flyte.org) runs the work as tasks (`datenwissenschaften.workflow`):
`speedrun` relays the stream, publishes the lab's status and searches and
improves the movie forever. `lab` first starts the critic (`agent/critic.sh`)
once five new reports landed in `agent/reports/` since its last review; the
stream shows its lessons as `metadata.critic`. Then it starts a lab run
(`agent/daily.sh` of the game package) once no level stored a piece, the movie
did not improve and no lab run ended for `agent.stall_minutes` minutes, as the
journal records them; a checkpoint is no piece, and a level
resting after its turn does not call the lab, since the next level may move. Claude Code works
for `agent.minutes` minutes on its own git worktree of the game package, and its
branch ships only when its checks pass and the current movie still replays with
it (`tas.run verify`). The lead session delegates to the subagents defined in the
game package's `.claude/agents/`; the stream shows each member's role, state and
latest step and every hand-off with its short task, the lead's latest actions
(only the kind of action and the source files it touches), its commit subjects
with URLs and secret-like words hidden, and the lines it changed. Every task
runs locally with `flyte run --local`.

### Experiments

The search and the policy each log one [Weights & Biases](https://wandb.ai) run per release
(project `retro-speedlab`, grouped by `ui.release`, jobs `search` and `policy`):

- **search:** the run from power-on as `run/seconds`, `run/pieces`, `run/levels_beaten` and the state it
  ends in, `run/reached`; every level's split as `levels/<level>`; running counts of `search/found`,
  `search/improved`, `search/checkpoint`, `search/stuck` and `search/lab`; the frames the trimming saved,
  `search/frames_saved`; and with every value `search/hours_since_progress`, the hours since a piece was
  found or made faster;
- **policy:** per PPO round `policy/reward`, `policy/exits`, `policy/episodes`, `policy/exit_rate`,
  `policy/entropy` and `policy/explained_variance`.

Without `WANDB_API_KEY` the runs are disabled and logged as not tracked.

### Lab runs

After the verdict, `python -m datenwissenschaften.lab.record CONFIG LOG STAMP BASE OUTCOME COMMITS`
(outcome `rejected`, `notes` or `deployed`) reads the run's Claude Code log once and

- publishes its numbers and a swimlane timeline (the lead and every subagent as lanes, tool calls as
  blocks, hand-offs as arrows) to the website at `PUT /lab-runs/{game}/{stamp}`;
- sends it to [Langfuse](https://langfuse.com) as the session `lab-run STAMP`, one trace per
  `claude -p` round, as OpenTelemetry spans with their recorded times. Every model turn is a
  generation with its token usage, every tool call a tool span and every subagent an agent span
  holding its own turns and tools. The session is scored with `accepted`, `deployed`, `cost_usd`,
  `commits`, `minutes`, `turns`, `subagents` and `tool_errors`.

Langfuse needs `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`; without them the
run is logged as not traced.

## Live relay

With `ui.enable: true` and an upload key, the engine relays to the Retro
Speedlab backend (`upload.url`) every few seconds the snapshot (the published
metadata and the release) and the search's newest frame; the whole verified movie
goes up whenever it changes. The website's `/stream` page, opened with the stream
key (`?api_key=`), plays them for OBS.
The engine itself serves nothing.

## Installation

Retro Speedlab Core requires **Python 3.12**.

```bash
git clone https://github.com/datenwissenschaften/retro-speedlab-core.git
cd retro-speedlab-core
uv sync
cp config.example.yaml config.yaml
```

The published package and import path are `datenwissenschaften`.

## Minimal usage

This repository does not ship a playable game; a game package supplies the RAM
layout, states, rewards and actions:

```python
import numpy as np

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper


class AirstrikerWrapper(StateMachineGymWrapper[AirstrikerRam]):
    start_state_cls = SurviveAndScore
    state_classes = (SurviveAndScore,)
    ram_info_cls = AirstrikerRam
    action_table = np.array([[[1] + [0] * 11, [0] * 12, [0] * 12, [0] * 12]], dtype=np.int8)
    levels = ()
```

```bash
python -m datenwissenschaften.tas.run search airstriker:AirstrikerWrapper config.yaml SurviveAndScore
```

Every state sets a one-line `goal`. The ROM is imported through Stable Retro
from `paths.roms`; the engine's own test suite needs no ROM.

## Configuration

Every key in `config.example.yaml` is required; missing or malformed values
raise an error when the configuration is loaded.

| Key | Meaning |
|-----|---------|
| `paths.roms`, `paths.integrations` | ROMs and custom Stable Retro integrations |
| `paths.models` | Route memory per state |
| `paths.cache` | The search's movie, progress, stuck marker, journal, live reports and log under `tas/` |
| `paths.lab_run` | Marker file of a running lab run |
| `training.game` | Stable Retro game id |
| `upload.url`, `upload.api_key` | Retro Speedlab API; `null` disables the relay |
| `ui.enable`, `ui.release` | Live relay on or off and the release label shown on stream |
| `twitch.enabled` | Whether the stream page goes live on Twitch |
| `agent.stall_minutes` | Minutes without a new piece, a faster movie or a lab run before the `lab` task starts one |
| `speedrun.workers`, `speedrun.segment_seconds` | Search processes and seconds per segment round |
| `speedrun.id`, `speedrun.title`, `speedrun.platform` | The game as the stream names it |
| `speedrun.category`, `speedrun.speedrun_com`, `speedrun.tasvideos` | The category run and where its records are read: speedrun.com game, category and variables, TASVideos publication |
| `log_level` | Loguru log level |

## Testing

```bash
uv run pytest --cov=datenwissenschaften --cov-fail-under=95
```

Tests run on a scripted emulator, so they need no ROM.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
```

## Relationship to Retro Speedlab

- **Retro Speedlab Core** (this repository): the engine: segment search,
  verified movies, the live relay and the lab workflow.
- **Game packages**: one repository per game with its RAM map, states,
  rewards, actions and Dokku deployment, improved by lab runs.
- **[retrospeedlab.com](https://www.retrospeedlab.com)**: the lab's website,
  showing the stream and the speedrun.

## License

Copyright © datenwissenschaften contributors.

Distributed under the [GNU General Public License v3.0](LICENSE).
