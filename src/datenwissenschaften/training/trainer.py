import time
from pathlib import Path

import torch

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.factory import curriculum_root, make_environment, state_names
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.settings import load_config
from datenwissenschaften.training.checkpoint_hook import CheckpointHook, model_metadata
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.hooks import TrainingHook
from datenwissenschaften.training.identity import TrainingIdentity
from datenwissenschaften.training.live_stream_hook import LiveStreamHook
from datenwissenschaften.training.rotation import Rotation
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
from datenwissenschaften.ui.telemetry import configure_history, publish_metadata


class LayaTrainer:
    def __init__(self, wrapper_cls: type[StateMachineGymWrapper], config_path: Path) -> None:
        self.config = load_config(config_path)
        setup_logging(self.config.log_level)
        self.context = RunContext(self.config, self.config.training.savestates[0])
        self.wrapper_cls = wrapper_cls
        self.ui_started = False

    def train(self) -> None:
        database = JsonDatabase(self.config.paths.database_path)
        training = self.config.training
        rotation = Rotation(database, training.game_identity, training.savestates, training.rotation_minutes)
        while True:
            savestate, seconds = rotation.next()
            self.context = RunContext(self.config, savestate)
            env = make_environment(self.wrapper_cls, self.config, savestate)
            identity = TrainingIdentity(self.context, database)
            identity.require_compatible(env)
            self._start_ui(identity, env, database)
            self._publish_levels(database)
            env.curriculum.publish()
            request = self._train_level(env, database, seconds)
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

    def _publish_levels(self, database: JsonDatabase) -> None:
        training = self.config.training
        for savestate in training.savestates:
            StoryBook(database, training.game_identity, savestate, self._phases())
            CurriculumRun(curriculum_root(self.config, savestate), state_names(self.wrapper_cls), savestate)

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
        return [
            LiveStreamHook(env.unwrapped.em.get_screen_rate(), teller, self.context.savestate),
            TelemetryHook(self.context),
            CheckpointHook(models),
            BestVideoHook(self.context),
            UploadHook(self.context, models.agent),
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
        configure_history(self.config.training.game_identity, database)
        start_ui(ui, self.context.record_root)
        self.ui_started = True

    def _publish_run(self) -> None:
        publish_metadata(
            "run",
            {
                "game": self.context.game,
                "savestate": self.context.savestate,
                "savestates": list(self.config.training.savestates),
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
