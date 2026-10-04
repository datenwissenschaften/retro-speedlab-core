import time
from pathlib import Path

import torch

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.environment.factory import POWER_ON, make_environment
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.settings import load_config
from datenwissenschaften.training.checkpoint_hook import CheckpointHook, model_metadata
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.curriculum_upload_hook import CurriculumUploadHook
from datenwissenschaften.training.hooks import TrainingHook
from datenwissenschaften.training.identity import TrainingIdentity
from datenwissenschaften.training.live_stream_hook import LiveStreamHook
from datenwissenschaften.training.report_upload_hook import ReportUploadHook
from datenwissenschaften.training.session import TrainingSession
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.training.story_book import StoryBook
from datenwissenschaften.training.story_teller import StoryTeller
from datenwissenschaften.training.telemetry_hook import TelemetryHook
from datenwissenschaften.training.upload_hook import UploadHook
from datenwissenschaften.training.video_hook import BestVideoHook
from datenwissenschaften.ui.control import ModelResetRequest, configure_training_control, perform_model_reset
from datenwissenschaften.ui.live import live_feed
from datenwissenschaften.ui.server import start_ui
from datenwissenschaften.ui.summaries import report_digest
from datenwissenschaften.ui.telemetry import configure_history, level_full_run_wins, publish_metadata

SESSION_SECONDS = 2 * 60 * 60
BEATEN_FULL_RUN_WINS = 8


class LayaTrainer:
    def __init__(self, wrapper_cls: type[StateMachineGymWrapper], config_path: Path) -> None:
        self.config = load_config(config_path)
        setup_logging(self.config.log_level)
        self.context = RunContext(self.config, POWER_ON)
        self.wrapper_cls = wrapper_cls
        self.ui_started = False
        self.speedrun = False

    def train(self) -> None:
        database = JsonDatabase(self.config.paths.database_path)
        configure_history(self.config.training.game_identity, database)
        while True:
            env = make_environment(self.wrapper_cls, self.config)
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
        self._publish_run()
        publish_metadata("model", model_metadata(models), replace=True)
        story = StoryBook(database, self.config.training.game_identity, self.context.savestate, self._phases())
        deadline = time.monotonic() + seconds
        try:
            return TrainingSession(env, models, self._hooks(env, models, StoryTeller(story)), deadline).run()
        finally:
            models.close()

    def _phases(self) -> tuple[str, ...]:
        classes = (self.wrapper_cls.start_state_cls, *self.wrapper_cls.state_classes)
        return tuple(dict.fromkeys(state_cls.__name__ for state_cls in classes))

    def _models(self) -> StateModels:
        network = LayaNetwork(
            self.config.laya.checkpoint, self.wrapper_cls.action_descriptions, configure_accelerator()
        )
        state_classes = (self.wrapper_cls.start_state_cls, *self.wrapper_cls.state_classes)
        agent = LayaAgent(network, tuple(state_cls.description for state_cls in state_classes))
        return StateModels(agent, self.context, self._phases())

    def _hooks(self, env: StateMachineGymWrapper, models: StateModels, teller: StoryTeller) -> list[TrainingHook]:
        frame_rate = env.unwrapped.em.get_screen_rate()
        stream = [LiveStreamHook(frame_rate, teller, self.context.savestate)]
        return [
            *(stream if self.config.ui.twitch else []),
            TelemetryHook(self.context),
            CheckpointHook(models),
            BestVideoHook(self.context),
            UploadHook(self.context, models.agent, frame_rate),
            ReportUploadHook(self.context),
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
        reports_dir = self.config.paths.reports_dir
        start_ui(ui, self.context.record_root, reports_dir, report_digest(ui, self.context.game, reports_dir))
        self.ui_started = True

    def _publish_run(self) -> None:
        publish_metadata(
            "run",
            {
                "game": self.context.game,
                "savestate": self.context.savestate,
                "speedrun": self.speedrun,
            },
        )
        publish_metadata(
            "environment",
            {
                "wrapper": self.wrapper_cls.__name__,
                "states": [state_cls.__name__ for state_cls in self.wrapper_cls.state_classes],
                "actions": self.wrapper_cls.action_descriptions,
                "frames_per_decision": self.wrapper_cls.action_table.shape[1],
            },
        )
