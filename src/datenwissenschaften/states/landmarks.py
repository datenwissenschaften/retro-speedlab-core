from math import dist
from pathlib import Path

from datenwissenschaften.persistence import JsonDatabase

Location = tuple[int, int]

ROUTE_PREFIX = "route:"
LOOP_DISTANCE = 12
WAYPOINT_DISTANCE = 48


class Landmarks:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.database = JsonDatabase(path)

    def remember(self, label: str, location: Location) -> None:
        if not self.database.contains(label):
            self.database.set(label, location)

    def recall(self, label: str) -> Location | None:
        if not self.database.contains(label):
            return None
        x, y = self.database.get(label)
        return x, y

    def remember_route(self, label: str, path: list[Location]) -> None:
        route = without_loops(path)
        known = self._route(label)
        if len(route) > 1 and (known is None or len(route) < len(known)):
            self.database.set(ROUTE_PREFIX + label, route)

    def knows_route(self, label: str) -> bool:
        return self.database.contains(ROUTE_PREFIX + label)

    def waypoint(self, label: str, location: Location) -> Location | None:
        destination = self.recall(label)
        if destination is None:
            return None
        known = self._route(label)
        route = [destination] if known is None else [*known, destination]
        nearest = min(range(len(route)), key=lambda index: dist(route[index], location))
        return next((point for point in route[nearest:] if dist(point, location) >= WAYPOINT_DISTANCE), destination)

    def forget(self) -> None:
        self.path.unlink(missing_ok=True)
        self.database = JsonDatabase(self.path)

    def _route(self, label: str) -> list[Location] | None:
        if not self.database.contains(ROUTE_PREFIX + label):
            return None
        return [(x, y) for x, y in self.database.get(ROUTE_PREFIX + label)]


def without_loops(path: list[Location]) -> list[Location]:
    route: list[Location] = []
    for point in path:
        earlier = enumerate(route[:-1])
        revisited = next((index for index, kept in earlier if dist(kept, point) < LOOP_DISTANCE), len(route))
        route = [*route[:revisited], point]
    return route
