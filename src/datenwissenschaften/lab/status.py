from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from datenwissenschaften.lab.activity import Action, recent_actions
from datenwissenschaften.lab.branch import BranchChanges, CodeHunk, branch_changes, newest_hunk
from datenwissenschaften.lab.marker import DuelResult, LabMarker

IDLE = {"active": False, "until": None}


@dataclass(slots=True, frozen=True)
class AgentStatus:
    actions: list[Action]
    changes: BranchChanges | None
    code: CodeHunk | None


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
    actions = recent_actions(agent, Path(branch.log), worktree)
    return AgentStatus(actions, branch_changes(worktree, lab.base), newest_hunk(worktree, lab.base))


def lab_status(lab: LabMarker) -> LabStatus:
    until = datetime.fromtimestamp(lab.until, UTC).isoformat()
    agents = {agent: agent_status(agent, lab) for agent in lab.agents}
    return LabStatus(True, until, lab.phase, agents, lab.duel)
