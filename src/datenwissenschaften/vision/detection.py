from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Detection:
    label: str
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width / 2, self.top + self.height / 2
