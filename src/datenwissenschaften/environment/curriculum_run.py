from pathlib import Path

from loguru import logger

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment.curriculum_outcomes import CurriculumOutcomes
from datenwissenschaften.environment.episode_tally import EpisodeTally
from datenwissenschaften.environment.level_clock import LevelClock
from datenwissenschaften.environment.level_drill import LevelDrill
from datenwissenschaften.environment.levels import LevelTargets
from datenwissenschaften.ui.telemetry import publish_metadata

FULL_RUN = "Full run"


class CurriculumRun:
    def __init__(
        self, root: Path, targets: LevelTargets, level: str, seeds_dir: Path, clock: LevelClock, counts_outcomes: bool
    ) -> None:
        self.counts_outcomes = counts_outcomes
        self.root = root
        self.level = level
        self.state_names = targets.state_names
        self.targets = targets
        self.clock = clock
        self.seeds_dir = seeds_dir
        self.drill = LevelDrill(root)
        self.outcomes = CurriculumOutcomes(targets, self.drill, clock)
        self.start_state = self.state_names[0]
        self.outcome_recorded = False
        self.tally = EpisodeTally(0, 0.0, 0.0)
        self.reset_memory()

    def reset_memory(self) -> None:
        self.curriculum = seeded(ReverseCurriculum(self.root, self.state_names), self.state_names, self.seeds_dir)
        self.publish()

    def begin_episode(self) -> str | None:
        target = self.drill.pick(self._frontier(), self.targets)
        self.start_state = target or FULL_RUN
        self.outcome_recorded = target is None or not self.counts_outcomes
        checkpoint_state = None if target is None else self.targets.start_checkpoint(target, self.curriculum)
        entry_score = 0.0 if checkpoint_state is None else self.curriculum.entry_score(checkpoint_state)
        self.tally = EpisodeTally(0, 0.0, entry_score)
        return checkpoint_state

    def _frontier(self) -> str | None:
        active_state = self.curriculum.active_state()
        if active_state is not None or self.targets.beats_game:
            return active_state
        return self.state_names[-1]

    def checkpoint(self, state_name: str) -> bytes:
        return self.curriculum.checkpoint(state_name)

    def transition(
        self, previous_state: str, new_state: str, emulator_state: bytes, step_reward: float
    ) -> tuple[bool, bool]:
        if not self.moves_forward(previous_state, new_state):
            return False, False
        if self.curriculum.save_checkpoint(new_state, emulator_state, self.tally.score + step_reward):
            logger.info(f"Saved automatic curriculum checkpoint for {new_state}")
        if self.targets.starts(self.start_state, new_state):
            self.tally.steps = 0
        if self.outcome_recorded or not self.targets.completes(self.start_state, previous_state, new_state):
            self.publish()
            return False, False
        return True, self._record_success()

    def moves_forward(self, previous_state: str, new_state: str) -> bool:
        return self.state_names.index(new_state) >= self.state_names.index(previous_state)

    def finish_step(
        self,
        won: bool,
        ended: bool,
        reward: float,
        transition: tuple[str, str] | None,
        outcome: tuple[bool, bool],
        state: str,
    ) -> dict[str, object]:
        succeeded, mastered = outcome
        if won:
            succeeded, mastered = succeeded or not self.outcome_recorded, self._record_success() or mastered
        elif ended:
            self.fail(reward, state)
        self.tally.add_reward(reward, transition is not None)
        return {
            "curriculum_state": self.start_state,
            "curriculum_succeeded": succeeded,
            "curriculum_mastered": mastered,
        }

    def fail(self, reward: float, state: str) -> None:
        if self.outcome_recorded:
            return
        self.outcome_recorded = True
        score = self.tally.segment_return + reward
        self.outcomes.fail(self.curriculum, self.start_state, self.tally.steps, score, state)
        self.publish()

    def publish(self) -> None:
        progress = self.curriculum.progress()
        publish_metadata("savestate_curriculum", progress, replace=True)
        publish_metadata("curricula", {self.level: progress})
        self.drill.publish(self.targets.levels)
        self.clock.publish()

    def _record_success(self) -> bool:
        if self.outcome_recorded:
            return False
        self.outcome_recorded = True
        mastered = self.outcomes.succeed(self.curriculum, self.start_state, self.tally.steps)
        self.publish()
        return mastered


def seed_path(seeds_dir: Path, state_name: str) -> Path:
    return seeds_dir / f"{state_name}.state"


def seeded(curriculum: ReverseCurriculum, state_names: tuple[str, ...], seeds_dir: Path) -> ReverseCurriculum:
    for state_name in state_names:
        seed = seed_path(seeds_dir, state_name)
        if seed.is_file() and not curriculum.has_checkpoint(state_name):
            if curriculum.save_checkpoint(state_name, seed.read_bytes(), 0.0):
                logger.info(f"Seeded curriculum checkpoint for {state_name} from {seed}")
    return curriculum
