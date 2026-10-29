from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from datenwissenschaften.lab.activity import Action, log_events, recent_actions
from datenwissenschaften.lab.branch import BranchChanges, branch_changes
from datenwissenschaften.lab.marker import DuelResult, LabMarker
from datenwissenschaften.lab.team import Member, team_status

IDLE = {"active": False, "until": None}
TEAM_AGENT = "claude"


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
    team: list[Member]
    handoffs: list[dict[str, str]]


def agent_status(agent: str, lab: LabMarker) -> AgentStatus:
    branch = lab.agents[agent]
    worktree = Path(branch.worktree)
    actions = recent_actions(agent, log_events(Path(branch.log)), worktree)
    return AgentStatus(actions, branch_changes(worktree, lab.base))


def lab_team(lab: LabMarker) -> tuple[list[Member], list[dict[str, str]]]:
    if TEAM_AGENT not in lab.agents:
        return [], []
    branch = lab.agents[TEAM_AGENT]
    return team_status(log_events(Path(branch.log)), Path(branch.worktree))


def lab_status(lab: LabMarker) -> LabStatus:
    until = datetime.fromtimestamp(lab.until, UTC).isoformat()
    agents = {agent: agent_status(agent, lab) for agent in lab.agents}
    team, handoffs = lab_team(lab)
    return LabStatus(True, until, lab.phase, agents, lab.duel, team, handoffs)
