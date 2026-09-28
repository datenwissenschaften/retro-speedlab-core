---
name: beat-nes-game
description: Build, train, diagnose and regularly optimize a retro-speedlab game package that beats an NES level with Laya (one Laya model per state machine state). Use when creating a new game package, when a game's training stalls or collapses, when asked to review rewards, detections or curriculum progress, or on a scheduled optimization run for a game project.
---

# Beat an NES game with Laya

A game package (for example `snakerattlenroll-nes-v0`) sits next to `retro-speedlab-core` and contains only
game knowledge: RAM map, detectors, actions, states, rewards. The engine owns learning, curriculum, UI and
deployment. Work in small verified steps, follow the core `AGENTS.md` (no defaults, fail fast, ruff, loguru,
python-box, no comments), and record every verified game fact in the package's `NOTES.md`.

## 1. Package layout

```
<game>/
    app.py                  LayaTrainer(Wrapper, CONFIG_PATH).train()
    <package>/ram.py        GameRam(RamInfo) with ram(0x...) fields and describe()
    <package>/vision.py     detectors built from assets/ templates and RAM positions
    <package>/actions.py    ACTION_TABLE (actions, frames, buttons) and ACTION_DESCRIPTIONS
    <package>/heading.py    maps an offset to the action that moves toward it (when movement is not screen-aligned)
    <package>/states/       one State per phase, a shared base state, exploring vs targeted states
    <package>/wrapper.py    StateMachineGymWrapper subclass wiring the above
    assets/                 template PNGs cut from real frames
    tests/                  one test per mechanic: detection, reward, transition
    dokku/, Dockerfile      deployment (see section 8)
    NOTES.md                verified RAM addresses, mechanics, pitfalls, open questions
```

Start new games from the retro-arena cookiecutter. The core version is the user's decision: never bump it.

## 2. Reconnaissance before writing states

Never guess mechanics. Verify each one in the emulator and write it into `NOTES.md` with the evidence.

- **Run headless**: `stable_retro.make(game, state=..., render_mode=None)`. Restore any savestate with
  `env.em.set_state(bytes); env.data.reset(); env.data.update_ram()`. Poke RAM for experiments with
  `env.unwrapped.data.memory.assign(address, "|u1", value)`.
- **Find sprite positions in RAM** with `scripts/ram_locate.py`. It correlates every byte with the on-screen
  center of a uniquely colored sprite and tests byte differences for isometric height:
  ```bash
  PYTHONPATH=. python ../retro-speedlab-core/.claude/skills/beat-nes-game/scripts/ram_locate.py \
      --game <Game-Nes-v0> --state Level1 --starts <extra.state> --actions <package>.actions:ACTION_TABLE \
      --color 248,116,96 --min-area 70 --max-fill 0.9 --playfield-bottom 201 --decisions 1500 --seed 3 --out locate.json
  ```
  Accept a formula only with a median error of about 1 px and a 90th percentile under the sprite size.
  Prefer RAM positions over color blobs: signs, pickups and HUD icons often share the player's colors.
- **Map the level** with `scripts/explore.py` (Go-Explore). It returns milestones (first time each tracked
  RAM value appears, with a savestate and action path each), death hotspots and maxima:
  ```bash
  PYTHONPATH=. python ../retro-speedlab-core/.claude/skills/beat-nes-game/scripts/explore.py \
      --game <Game-Nes-v0> --state Level1 --start none --actions <package>.actions:ACTION_TABLE \
      --cell 0x04C3,0x0093,0x04D7/24,0x04FF/24,0x0062,0x00B6 --priority 0,0,0,0,4,20 \
      --lives 0x03DF --won 0x0432 --minutes 20 --seed 0 --out explore/
  ```
  Put progress bytes (weight, keys, door, score) in `--cell` with a high `--priority`. Continue from a
  milestone with `--start explore/<milestone>.state`. If a gate value is never reached, the state machine
  cannot pass that gate: fix the gate before training.
- **Prove every detector on real frames** from the level start and from later milestones: draw
  `state.detections()` with `datenwissenschaften.vision.overlay.draw_detections` and look at the image. Check
  both misses and false positives (same-colored signs, the player's own head, HUD, thin edges).
- **Prove every transition condition** by reaching it (explorer milestone or steering with the `move` hint)
  and reading the RAM byte change. A condition that never fires blocks the whole curriculum.

## 3. State machine

- One state per phase with a single goal: search for X, reach X, operate X, leave through X.
- `description` is Laya's question: one short line, `"You are the <hero>. <goal>. Which move?"`. Long
  questions cost tokens and overflow the stream panel.
- `describe()` gives only what the choice needs: the target with `visible`, `direction`, `distance`, and a
  `move` field naming the action that heads toward it (`last_seen` when it left the screen), the nearest
  enemy, a few RAM facts. Pretrained Laya follows a `move` hint about 89% of the time before any training.
- `_next()` returns the next state class only on a verified condition. `_won()` only in the final state.
- `_terminated()` on life loss and on damage that the game counts (for example weight dropping).
- `_truncated()` after a fixed frame budget per state (3600 frames = 60 s at 60 Hz).

## 4. Rewards

Each state trains its own Laya, and a transition ends that model's trajectory. So inside every state:

| Signal | Rule |
|---|---|
| Transition to the next state | the largest reward of the state (10); must beat anything farmable before it |
| Winning the level | 20 in the final state |
| Approach | potential-based: `0.05 × pixels closer`, ignore jumps above the target-switch distance (24 px) |
| Progress | only for verified RAM progress (eaten items +1, weight +2) |
| Exploration | only in searching states: new screen +1, new 16 px spot +0.1 |
| Failure | −10 on life or damage loss, ends the episode |
| Step cost | 0.002 per frame, so a full timeout costs about −7 |

Check each state: finishing fast beats wandering until the timeout, dying is worse than timing out, and no
loop can farm reward (approach rewards are refunded when moving away). Episode scores accumulate across the
curriculum: a checkpoint stores the score that reached it and episodes starting there continue from it.

## 5. Actions

- `ACTION_TABLE` has shape `(actions, frames, buttons)`; 4 frames per decision works well.
- Hold movement buttons for all frames; tap edge-triggered buttons (fire, tongue, jump) on frame 0 only.
- Name actions by what they do on screen (`up-right`, not `UP`) and map offsets to them in `heading.py`
  when the game projects movement (isometric: UP moves up-right, DOWN down-left, LEFT up-left, RIGHT down-right).

## 6. How Laya learns here (engine facts, verify before changing)

- One checkpoint per state: `working/models/<game>/<savestate>/<State>/laya.pt` (weights, 8-bit AdamW,
  trust region, trained decisions). The active state's model is swapped into the single GPU slot.
- Options are shuffled by a stable hash of state and question. Without it Laya gives about 97% to the first
  listed option and every fresh model starts stuck on one action.
- Group-relative policy gradient with importance weights for the exploration mixture (20% random actions,
  annealed to 5% over 50k decisions). Keep the importance weights: without them the policy saturates on
  wrong choices.
- The trust region targets KL 0.01 per update: it grows the step by the square root of the shortfall (at most
  ×10) and shrinks it on overshoot; an overshooting update is blended back toward a CPU weight snapshot.
- Float16 GPUs (Turing, RTX 20xx) need learning-rate scales around 10³–10⁴; bfloat16 is noisier (7-bit
  mantissa) and relies on backtracking.
- The entropy bonus is a small constant (0.001). An adaptive entropy target pushed every model toward
  uniform play and unlearned correct choices.
- Changing what a model sees (option order, observation layout) requires a new `MODEL_LAYOUT` value in
  `training/identity.py`; the server then starts fresh once. Never use the core version for this.

## 7. Diagnose a run

Read `http://<host>/api/snapshot` (`metadata.model.laya`, `metadata.savestate_curriculum`, `summary`) and
`/api/live/episode`, `/api/live/frames?episode=<id>&start=<n>` (per-frame status with probabilities, action,
state description, and JPEG frames to look at).

| Symptom | Cause | Action |
|---|---|---|
| One action at 100%, entropy near 0 | collapse | check option shuffle, importance weights, trust region; start the state fresh |
| `learning_rate_scale` at its maximum, `kl` far below 0.01 | steps too small | widen the scale range |
| Probabilities near uniform for a long time | entropy pressure or no reward signal | check the entropy constant and the state's rewards |
| Episodes end in seconds at about −10 | early damage or death | watch replays for the hazard; add it to `describe()` |
| Episodes time out at about −7 in a searching state | target never seen | verify the detector on that screen; check the exploration reward |
| A state never gains wins | transition condition never fires | reproduce with the explorer or steering, then fix the condition |
| Curriculum checkpoint deleted repeatedly | score stagnation from a bad start | inspect the checkpoint frame |

## 8. Deploy (Dokku)

- `dokku/setup.sh` once per server; `dokku/deploy.sh` builds a repo from the package plus a `git archive` of
  the committed core and pushes it. Commit the core first; never commit `config.yaml` or ROMs.
- Server data lives under `/mnt/fastdata` (app data, Docker `data-root`, containerd `root`). Run
  `sudo docker builder prune -af` when the build cache grows.
- A push that is interrupted locally keeps building on the server: check with `ps` for `git-receive-pack`
  before pushing again, and `dokku apps:unlock <app>` only after the old build is gone.
- Deploying restarts training from the saved state checkpoints; wait-to-retire is 0 so two containers never
  share the GPU.

## 9. Regular optimization run

Run this loop on a schedule for each game project, one change per run:

1. Read the snapshot and the last episodes. Record per state: wins/target, trained decisions, entropy, KL,
   learning-rate scale, typical end reason (win, damage, death, timeout).
2. Pick the earliest state that is not mastered. Look at two recent replays of it (frames plus status).
3. Find the single biggest blocker with the table in section 7. If it is a game mechanic, reproduce it
   offline (explorer, steering, RAM poke) before touching code.
4. Make the smallest fix in the game package (detector, `describe()`, reward, transition, question) or in the
   core only when it applies to every game. Add or update a test that pins the mechanic.
5. `ruff check`, `ruff format`, `pytest` in both repositories, then commit with a message that states the
   evidence, and deploy.
6. After the next update cycle, confirm the metric moved; revert the change if it made the state worse.
7. Update `NOTES.md` with what was verified, what was ruled out, and the next open question.

Guardrails: never bump the core version, never delete training data or press "delete model" without the
user's request, never commit secrets, and keep every commit buildable.
