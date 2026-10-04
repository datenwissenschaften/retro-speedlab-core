from datenwissenschaften.environment.demonstration import Demonstrations
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper


def knowledge(env: StateMachineGymWrapper, checkpoint: str, demonstrations: Demonstrations) -> dict[str, object]:
    questions = {state_cls.__name__: state_cls.description for state_cls in (env.start_state_cls, *env.state_classes)}
    return {
        "checkpoint": checkpoint,
        "actions": len(env.action_descriptions),
        "states": [
            {
                "name": name,
                "question": questions[name],
                "seeded": env.curriculum.seed(name).is_file(),
                "demonstrations": len(demonstrations[name]) if name in demonstrations else 0,
            }
            for name in questions
        ],
    }
