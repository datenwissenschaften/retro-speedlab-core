from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from datenwissenschaften.lab.activity import Action, recent_actions
from datenwissenschaften.lab.branch import BranchChanges, branch_changes
from datenwissenschaften.lab.marker import DuelResult, LabMarker

IDLE = {"active": False, "until": None}


@dataclass(slots=True, frozen=True)
class AgentStatus:
    actions: list[Action]
    changes: BranchChanges | None


@dataclass(slots=True, frozen=True)
class LabStatus:
    active: bool
    until: str
    phase: str
    agents: dict[str, AgentStatus]
    duel: DuelResult | None


def agent_status(agent: str, lab: LabMarker) -> AgentStatus:
    branch = lab.agents[agent]
    worktree = Path(branch.worktree)
    return AgentStatus(recent_actions(agent, Path(branch.log), worktree), branch_changes(worktree, lab.base))


def lab_status(lab: LabMarker) -> LabStatus:
    until = datetime.fromtimestamp(lab.until, UTC).isoformat()
    agents = {agent: agent_status(agent, lab) for agent in lab.agents}
    return LabStatus(True, until, lab.phase, agents, lab.duel)
