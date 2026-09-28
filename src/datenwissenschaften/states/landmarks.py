from pathlib import Path

from datenwissenschaften.persistence import JsonDatabase

Location = tuple[int, int]


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

    def forget(self) -> None:
        self.path.unlink(missing_ok=True)
        self.database = JsonDatabase(self.path)
