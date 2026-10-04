from pathlib import Path

from loguru import logger

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.ui.telemetry import publish_metadata


class CurriculumRun:
    def __init__(self, root: Path, state_names: tuple[str, ...], level: str, seeds_dir: Path) -> None:
        self.root = root
        self.level = level
        self.state_names = state_names
        self.seeds_dir = seeds_dir
        self.curriculum = self._seeded(ReverseCurriculum(root, state_names))
        self.start_state = state_names[0]
        self.outcome_recorded = False
        self.episode_steps = 0
        self.segment_return = 0.0
        self.episode_score = 0.0
        self.publish()

    def reset_memory(self) -> None:
        self.curriculum = self._seeded(ReverseCurriculum(self.root, self.state_names))
        self.publish()

    def begin_episode(self) -> str | None:
        active_state = self.curriculum.active_state()
        self.start_state = active_state or self.state_names[0]
        self.outcome_recorded = active_state is None
        self.episode_steps = 0
        self.segment_return = 0.0
        checkpoint_state = self.curriculum.episode_start_state()
        self.episode_score = 0.0 if checkpoint_state is None else self.curriculum.entry_score(checkpoint_state)
        return checkpoint_state

    def checkpoint(self, state_name: str) -> bytes:
        return self.curriculum.checkpoint(state_name)

    def count_step(self) -> None:
        self.episode_steps += 1

    def transition(
        self, previous_state: str, new_state: str, emulator_state: bytes, step_reward: float
    ) -> tuple[bool, bool]:
        if self.state_names.index(new_state) < self.state_names.index(previous_state):
            return False, False
        if self.curriculum.save_checkpoint(new_state, emulator_state, self.episode_score + step_reward):
            logger.info(f"Saved automatic curriculum checkpoint for {new_state}")
        if new_state == self.start_state:
            self.episode_steps = 0
        if self.outcome_recorded or previous_state != self.start_state:
            self.publish()
            return False, False
        return True, self._record_success()

    def finish_step(
        self, won: bool, ended: bool, reward: float, transition: tuple[str, str] | None, outcome: tuple[bool, bool]
    ) -> dict[str, object]:
        succeeded, mastered = outcome
        if won:
            win_succeeded, win_mastered = self.win()
            succeeded, mastered = succeeded or win_succeeded, mastered or win_mastered
        elif ended:
            self.fail(reward)
        self.add_reward(reward, transition is not None)
        return {
            "curriculum_state": self.start_state,
            "curriculum_succeeded": succeeded,
            "curriculum_mastered": mastered,
            "curriculum_complete": self.curriculum.is_complete(),
        }

    def win(self) -> tuple[bool, bool]:
        succeeded = not self.outcome_recorded
        return succeeded, self._record_success()

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

    def _record_success(self) -> bool:
        if self.outcome_recorded:
            return False
        self.outcome_recorded = True
        mastered = self.curriculum.record_success(self.start_state, self.episode_steps)
        if mastered:
            logger.info(f"Mastered curriculum state {self.start_state}; advancing to the next state")
        else:
            wins = self.curriculum.wins(self.start_state)
            logger.info(f"Curriculum win for {self.start_state}: {wins}/{self.curriculum.win_target(self.start_state)}")
        self.publish()
        return mastered

    def seed(self, state_name: str) -> Path:
        return self.seeds_dir / f"{state_name}.state"

    def _seeded(self, curriculum: ReverseCurriculum) -> ReverseCurriculum:
        for state_name in self.state_names:
            seed = self.seed(state_name)
            if seed.is_file() and not curriculum.has_checkpoint(state_name):
                if curriculum.save_checkpoint(state_name, seed.read_bytes(), 0.0):
                    logger.info(f"Seeded curriculum checkpoint for {state_name} from {seed}")
        return curriculum
