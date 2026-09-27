from pathlib import Path

import torch

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.environment.factory import make_environment
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
from datenwissenschaften.training.session import TrainingSession
from datenwissenschaften.training.telemetry_hook import TelemetryHook
from datenwissenschaften.training.upload_hook import UploadHook
from datenwissenschaften.training.video_hook import BestVideoHook
from datenwissenschaften.ui.control import ModelResetRequest, configure_training_control, perform_model_reset
from datenwissenschaften.ui.server import start_ui
from datenwissenschaften.ui.telemetry import configure_history, publish_metadata


class LayaTrainer:
    def __init__(self, wrapper_cls: type[StateMachineGymWrapper], config_path: Path) -> None:
        self.config = load_config(config_path)
        setup_logging(self.config.log_level)
        self.context = RunContext(self.config)
        self.wrapper_cls = wrapper_cls

    def train(self) -> None:
        env = make_environment(self.wrapper_cls, self.config)
        database = JsonDatabase(self.config.paths.database_path)
        identity = TrainingIdentity(self.context, database)
        identity.require_compatible(env)
        self._start_ui(identity, env, database)
        while True:
            request = self._train_until_reset(env)
            torch.cuda.empty_cache()
            perform_model_reset(request)

    def _train_until_reset(self, env: StateMachineGymWrapper) -> ModelResetRequest:
        agent = self._agent()
        publish_metadata("model", model_metadata(agent), replace=True)
        return TrainingSession(env, agent, self._hooks(env, agent)).run()

    def _agent(self) -> LayaAgent:
        network = LayaNetwork(
            self.config.laya.checkpoint, self.wrapper_cls.action_descriptions, configure_accelerator()
        )
        state_classes = (self.wrapper_cls.start_state_cls, *self.wrapper_cls.state_classes)
        agent = LayaAgent(network, tuple(state_cls.description for state_cls in state_classes))
        if self.context.model_path.is_file():
            agent.load(self.context.model_path)
        return agent

    def _hooks(self, env: StateMachineGymWrapper, agent: LayaAgent) -> list[TrainingHook]:
        return [
            LiveStreamHook(env.unwrapped.em.get_screen_rate()),
            TelemetryHook(self.context),
            CheckpointHook(self.context, agent),
            BestVideoHook(self.context),
            UploadHook(self.context, agent),
        ]

    def _start_ui(self, identity: TrainingIdentity, env: StateMachineGymWrapper, database: JsonDatabase) -> None:
        ui = self.config.ui
        if not ui.enabled:
            return
        configure_history(self.config.training.game_identity, database)
        reset = identity.reset_request(env)
        configure_training_control(
            game=reset.game,
            model_dir=reset.model_dir,
            restart_supported=True,
            artifact_dirs=reset.artifact_dirs,
            on_reset=reset.on_reset,
        )
        start_ui(ui, self.context.record_root)
        publish_metadata(
            "run",
            {"game": self.context.game, "savestate": self.context.savestate, "savestates": [self.context.savestate]},
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
