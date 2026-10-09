from collections.abc import Callable
from pathlib import Path

from datenwissenschaften.route.position import Position
from datenwissenschaften.route.state_route import StateRoute, Step
from datenwissenschaften.states.facts import Facts, Offset


class RouteMap:
    def __init__(self, path: Callable[[str], Path], actions: tuple[str, ...]) -> None:
        self.path = path
        self.actions = actions
        self.routes: dict[str, StateRoute] = {}

    def facts(self, state: str, position: Position | None) -> Facts:
        if position is None:
            return {}
        route = self._route(state)
        move = route.move(position)
        if move is not None:
            return {"route": self.actions[move.action]}
        nearest = route.nearest(position)
        if nearest is None:
            return {}
        x, y = nearest
        return {"to_route": Offset(x - position.x, y - position.y)}

    def record(self, state: str, trail: list[Step]) -> None:
        if trail:
            self._route(state).record(trail)

    def summary(self) -> dict[str, int]:
        return {state: len(route.moves) for state, route in self.routes.items() if route.moves}

    def _route(self, state: str) -> StateRoute:
        if state not in self.routes:
            self.routes[state] = StateRoute(self.path(state))
        return self.routes[state]
