from dataclasses import dataclass
from pathlib import Path

import msgspec

from datenwissenschaften.lab.branch import shown_subject

CRITIC_FILE = ".lab-critic.json"


@dataclass(slots=True, frozen=True)
class Critic:
    at: str
    runs: int
    lessons: list[str]


def read_critic(path: Path) -> Critic | None:
    if not path.is_file():
        return None
    critic = msgspec.json.decode(path.read_bytes(), type=Critic)
    return Critic(critic.at, critic.runs, [shown_subject(lesson) for lesson in critic.lessons])
