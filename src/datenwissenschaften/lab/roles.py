import re
from pathlib import Path

import yaml
from loguru import logger

from datenwissenschaften.lab.branch import shown_subject

AGENTS_DIR = Path(".claude") / "agents"
FRONT_MATTER = "---"
SENTENCE_END = ". "
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")


def agent_role(definition: Path) -> tuple[str, str]:
    _, front_matter, _ = definition.read_text(encoding="utf-8").split(FRONT_MATTER, 2)
    fields = yaml.safe_load(front_matter)
    if not SAFE_NAME.fullmatch(fields["name"]):
        raise ValueError(f"unsafe agent name {fields['name']!r}")
    sentence = fields["description"].split(SENTENCE_END)[0].rstrip(".")
    return fields["name"], shown_subject(sentence)


def agent_roles(worktree: Path) -> dict[str, str]:
    roles = {}
    for definition in sorted((worktree / AGENTS_DIR).glob("*.md")):
        try:
            name, role = agent_role(definition)
        except (ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
            logger.warning(f"The agent definition {definition.name} cannot be read: {error!r}")
            continue
        roles[name] = role
    return roles
