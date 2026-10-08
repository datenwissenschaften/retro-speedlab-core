import os
import time
from pathlib import Path

import torch

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.advisor.backplay import Backplay
from datenwissenschaften.advisor.team import AdvisorTeam
from datenwissenschaften.environment.curriculum_run import FULL_RUN
from datenwissenschaften.environment.demonstration import load_demonstrations
from datenwissenschaften.environment.factory import POWER_ON, make_environment
from datenwissenschaften.environment.levels import level_map
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.settings import load_config
from datenwissenschaften.training.beaten_level_hook import BeatenLevelHook
from datenwissenschaften.training.checkpoint_hook import CheckpointHook, model_metadata
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.curriculum_upload_hook import CurriculumUploadHook
from datenwissenschaften.training.hooks import TrainingHook
from datenwissenschaften.training.identity import TrainingIdentity
from datenwissenschaften.training.knowledge import knowledge
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.lessons import Lessons
from datenwissenschaften.training.live_stream_hook import LiveStreamHook
from datenwissenschaften.training.practice import PracticeEnvironments
from datenwissenschaften.training.report_upload_hook import ReportUploadHook
from datenwissenschaften.training.session import MAIN_ENVIRONMENT, TrainingSession
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.training.story_book import StoryBook
from datenwissenschaften.training.story_teller import StoryTeller
from datenwissenschaften.training.telemetry_hook import TelemetryHook
from datenwissenschaften.training.upload_hook import UploadHook
from datenwissenschaften.training.video_hook import BestVideoHook
from datenwissenschaften.ui.control import ModelResetRequest, configure_training_control, perform_model_reset
from datenwissenschaften.ui.live import live_feed
from datenwissenschaften.ui.relay import LiveRelay
from datenwissenschaften.ui.summaries import report_digest
from datenwissenschaften.ui.telemetry import configure_history, get_store, level_full_run_wins, publish_metadata

SESSION_SECONDS = 2 * 60 * 60
BEATEN_FULL_RUN_WINS = 8
RESERVED_CORES = 2
ADVISOR_THREADS = 2


def phases(wrapper_cls: type[StateMachineGymWrapper]) -> tuple[str, ...]:
    classes = (wrapper_cls.start_state_cls, *wrapper_cls.state_classes)
    return tuple(dict.fromkeys(state_cls.__name__ for state_cls in classes))


class LayaTrainer:
    def __init__(self, wrapper_cls: type[StateMachineGymWrapper], config_path: Path) -> None:
        self.config = load_config(config_path)
        setup_logging(self.config.log_level)
        self.context = RunContext(self.config, POWER_ON)
        self.wrapper_cls = wrapper_cls
        self.ui_started = False
        self.speedrun = False

    def train(self) -> None:
        torch.set_num_threads(ADVISOR_THREADS)
        database = JsonDatabase(self.config.paths.database_path)
        configure_history(self.config.training.game_identity, database)
        replays = self.config.paths.cache_dir / "replays" / self.config.training.game_identity
        live_feed.keep_replays_in(replays, self._curricula())
        while True:
            env = make_environment(self.wrapper_cls, self.config, MAIN_ENVIRONMENT, True)
            env.speedrun = self.speedrun = level_full_run_wins(POWER_ON) >= BEATEN_FULL_RUN_WINS
            identity = TrainingIdentity(self.context, database)
            identity.require_compatible(env)
            self._start_ui(identity, env, database)
            env.curriculum.publish()
            request = self._train_level(env, database, SESSION_SECONDS)
            env.close()
            torch.cuda.empty_cache()
            if request is not None:
                perform_model_reset(request)
                live_feed.clear()

    def _train_level(
        self, env: StateMachineGymWrapper, database: JsonDatabase, seconds: float
    ) -> ModelResetRequest | None:
        models = self._models()
        workers = range(MAIN_ENVIRONMENT + 1, len(os.sched_getaffinity(0)) - RESERVED_CORES)
        publish_run(self.wrapper_cls, self.context, self.speedrun, len(workers) + 1)
        publish_metadata("model", model_metadata(models), replace=True)
        story = StoryBook(
            database, self.config.training.game_identity, self.context.savestate, phases(self.wrapper_cls)
        )
        demonstrations, starts = load_demonstrations(env, self.config.paths.demonstrations_dir)
        publish_metadata("knowledge", knowledge(env, self.config.laya.checkpoint, demonstrations), replace=True)
        deadline = time.monotonic() + seconds
        lessons = Lessons(demonstrations, env.curriculum.curriculum)
        lab_run = LabRun(self.config.paths.lab_run_marker)
        actions = tuple(self.wrapper_cls.action_descriptions)
        advisors = Advisors(actions, configure_accelerator(), self.context.advisor_path)
        env.observer.advisor = advisors.advise
        practice = PracticeEnvironments(self.wrapper_cls, self.config, self.speedrun, workers)
        backplay = Backplay(starts, phases(self.wrapper_cls), self.context.backplay_path)
        team = AdvisorTeam(advisors, practice, lab_run, lessons, backplay)
        try:
            hooks = self._hooks(env, models, StoryTeller(story))
            return TrainingSession(env, models, hooks, deadline, lessons, lab_run, team.coach).run()
        finally:
            team.close()
            models.close()

    def _curricula(self) -> frozenset[str]:
        levels = level_map(self.wrapper_cls.levels, self.wrapper_cls.state_classes)
        return frozenset((*phases(self.wrapper_cls), *levels, FULL_RUN))

    def _models(self) -> StateModels:
        network = LayaNetwork(
            self.config.laya.checkpoint, self.wrapper_cls.action_descriptions, configure_accelerator()
        )
        state_classes = (self.wrapper_cls.start_state_cls, *self.wrapper_cls.state_classes)
        agent = LayaAgent(network, tuple(state_cls.description for state_cls in state_classes))
        return StateModels(agent, self.context, phases(self.wrapper_cls))

    def _hooks(self, env: StateMachineGymWrapper, models: StateModels, teller: StoryTeller) -> list[TrainingHook]:
        frame_rate = env.unwrapped.em.get_screen_rate()
        stream = [LiveStreamHook(frame_rate, teller, self.context.savestate)]
        digest = report_digest(self.config.ui, self.context.game, self.config.paths.reports_dir)
        return [
            *(stream if self.config.ui.twitch else []),
            TelemetryHook(self.context),
            CheckpointHook(models),
            BestVideoHook(self.context, self._curricula()),
            BeatenLevelHook(env.curriculum),
            UploadHook(self.context, models.agent, frame_rate, frozenset(env.curriculum.targets.levels)),
            ReportUploadHook(self.context, digest),
            CurriculumUploadHook(self.context, env.curriculum),
        ]

    def _start_ui(self, identity: TrainingIdentity, env: StateMachineGymWrapper, database: JsonDatabase) -> None:
        ui = self.config.ui
        if not ui.enabled:
            return
        reset = identity.reset_request(env)
        configure_training_control(
            game=reset.game,
            model_dir=reset.model_dir,
            restart_supported=True,
            artifact_dirs=reset.artifact_dirs,
            on_reset=reset.on_reset,
        )
        if self.ui_started:
            return
        upload = self.config.upload
        if upload.api_key is None:
            raise RuntimeError("ui.enable relays the stream to the backend and needs upload.api_key.")
        get_store().resize(ui.max_episodes)
        LiveRelay(upload.url, upload.api_key, ui, self.context.record_root).start()
        self.ui_started = True


def publish_run(wrapper_cls: type[StateMachineGymWrapper], context: RunContext, speedrun: bool, emulators: int) -> None:
    publish_metadata(
        "run",
        {
            "game": context.game,
            "savestate": context.savestate,
            "speedrun": speedrun,
            "emulators": emulators,
        },
    )
    publish_metadata(
        "environment",
        {
            "wrapper": wrapper_cls.__name__,
            "states": [state_cls.__name__ for state_cls in wrapper_cls.state_classes],
            "levels": level_map(wrapper_cls.levels, wrapper_cls.state_classes),
            "actions": wrapper_cls.action_descriptions,
            "frames_per_decision": wrapper_cls.action_table.shape[1],
        },
    )
