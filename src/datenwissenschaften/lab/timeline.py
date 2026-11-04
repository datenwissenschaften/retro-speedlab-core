from collections import Counter
from datetime import datetime
from typing import Any

from datenwissenschaften.lab.transcript import AGENT, TOOL, Round, Step

LEAD = "lead"


def lane_names(agents: list[Step]) -> list[str]:
    totals, seen, names = Counter(agent.name for agent in agents), Counter[str](), [LEAD]
    for agent in agents:
        seen[agent.name] += 1
        names.append(agent.name if totals[agent.name] == 1 else f"{agent.name} {seen[agent.name]}")
    return names


def seconds(moment: datetime, start: datetime) -> float:
    return round((moment - start).total_seconds(), 1)


def timeline(rounds: list[Round]) -> dict[str, Any]:
    start = rounds[0].start
    steps = [step for lab_round in rounds for step in lab_round.steps]
    agents = [step for step in steps if step.kind == AGENT]
    lane: dict[str | None, int] = {None: 0, **{agent.key: index for index, agent in enumerate(agents, 1)}}

    def lane_of(step: Step) -> int:
        return lane[step.owner] if step.owner in lane else 0

    blocks = [
        {
            "lane": lane_of(step),
            "start": seconds(step.start, start),
            "end": seconds(step.end, start),
            "name": step.name,
            "failed": step.failed,
        }
        for step in steps
        if step.kind == TOOL
    ]
    handoffs = [
        {"at": seconds(agent.start, start), "source": lane_of(agent), "target": lane[agent.key]} for agent in agents
    ]
    return {"lanes": lane_names(agents), "blocks": blocks, "handoffs": handoffs}
