from pathlib import Path

from loguru import logger

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment.level_clock import LevelClock
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
        self.curriculum = seeded(ReverseCurriculum(root, self.state_names), self.state_names, seeds_dir)
        self.start_state = self.state_names[0]
        self.outcome_recorded = False
        self.episode_steps = 0
        self.segment_return = 0.0
        self.episode_score = 0.0
        self.publish()

    def reset_memory(self) -> None:
        self.curriculum = seeded(ReverseCurriculum(self.root, self.state_names), self.state_names, self.seeds_dir)
        self.publish()

    def begin_episode(self) -> str | None:
        active_state = self.curriculum.active_state()
        self.start_state = active_state or FULL_RUN
        self.outcome_recorded = active_state is None or not self.counts_outcomes
        self.episode_steps = 0
        self.segment_return = 0.0
        checkpoint_state = self.targets.start_checkpoint(active_state, self.curriculum)
        self.episode_score = 0.0 if checkpoint_state is None else self.curriculum.entry_score(checkpoint_state)
        return checkpoint_state

    def checkpoint(self, state_name: str) -> bytes:
        return self.curriculum.checkpoint(state_name)

    def count_step(self) -> None:
        self.episode_steps += 1

    def transition(
        self, previous_state: str, new_state: str, emulator_state: bytes, step_reward: float
    ) -> tuple[bool, bool]:
        if not self.moves_forward(previous_state, new_state):
            return False, False
        if self.curriculum.save_checkpoint(new_state, emulator_state, self.episode_score + step_reward):
            logger.info(f"Saved automatic curriculum checkpoint for {new_state}")
        if self.targets.starts(self.start_state, new_state):
            self.episode_steps = 0
        if self.outcome_recorded or not self.targets.completes(self.start_state, previous_state, new_state):
            self.publish()
            return False, False
        return True, self._record_success()

    def moves_forward(self, previous_state: str, new_state: str) -> bool:
        return self.state_names.index(new_state) >= self.state_names.index(previous_state)

    def finish_step(
        self, won: bool, ended: bool, reward: float, transition: tuple[str, str] | None, outcome: tuple[bool, bool]
    ) -> dict[str, object]:
        succeeded, mastered = outcome
        if won:
            win_succeeded = not self.outcome_recorded
            succeeded, mastered = succeeded or win_succeeded, self._record_success() or mastered
        elif ended:
            self.fail(reward)
        self.add_reward(reward, transition is not None)
        return {
            "curriculum_state": self.start_state,
            "curriculum_succeeded": succeeded,
            "curriculum_mastered": mastered,
        }

    def fail(self, reward: float) -> None:
        if self.outcome_recorded:
            return
        if self.curriculum.record_failure(self.start_state, self.episode_steps, self.segment_return + reward):
            logger.warning(f"Deleted score-stagnant automatic checkpoint for {self.start_state}")
        self.publish()

    def add_reward(self, reward: float, transitioned: bool) -> None:
        self.episode_score += reward
        self.segment_return = 0.0 if transitioned else self.segment_return + reward

    def publish(self) -> None:
        progress = self.curriculum.progress()
        publish_metadata("savestate_curriculum", progress, replace=True)
        publish_metadata("curricula", {self.level: progress})
        self.clock.publish()

    def _record_success(self) -> bool:
        if self.outcome_recorded:
            return False
        self.outcome_recorded = True
        if self.targets.is_level(self.start_state):
            self.clock.record(self.start_state, self.episode_steps)
        limit = self.curriculum.step_limit(self.start_state)
        fast_enough = self.curriculum.is_fast_enough(self.start_state, self.episode_steps)
        mastered = self.curriculum.record_success(self.start_state, self.episode_steps)
        if not fast_enough:
            logger.info(f"Too slow for {self.start_state}: {self.episode_steps} steps, limit {limit}")
            self.publish()
            return False
        wins, target = self.curriculum.wins(self.start_state), self.curriculum.win_target(self.start_state)
        logger.info(f"Curriculum win for {self.start_state}: {wins}/{target}{', mastered' if mastered else ''}")
        self.publish()
        return mastered

    def seed(self, state_name: str) -> Path:
        return self.seeds_dir / f"{state_name}.state"


def seeded(curriculum: ReverseCurriculum, state_names: tuple[str, ...], seeds_dir: Path) -> ReverseCurriculum:
    for state_name in state_names:
        seed = seeds_dir / f"{state_name}.state"
        if seed.is_file() and not curriculum.has_checkpoint(state_name):
            if curriculum.save_checkpoint(state_name, seed.read_bytes(), 0.0):
                logger.info(f"Seeded curriculum checkpoint for {state_name} from {seed}")
    return curriculum
