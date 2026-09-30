from datenwissenschaften.persistence import JsonDatabase

Location = tuple[int, int]

CELL_SIZE = 16
DANGER = "danger"
BLOCKED = "blocked"


class SafetyMap:
    def __init__(self, database: JsonDatabase) -> None:
        self.database = database

    def mark(self, kind: str, location: Location) -> None:
        key = _key(kind, location)
        self.database.set(key, self.count(kind, location) + 1)

    def count(self, kind: str, location: Location) -> int:
        key = _key(kind, location)
        return int(self.database.get(key)) if self.database.contains(key) else 0


def _key(kind: str, location: Location) -> str:
    return f"{kind}:{location[0] // CELL_SIZE}:{location[1] // CELL_SIZE}"
