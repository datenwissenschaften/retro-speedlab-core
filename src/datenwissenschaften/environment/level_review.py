from datenwissenschaften.curriculum import ReverseCurriculum

REVIEW_INTERVAL = 10


class LevelReview:
    def __init__(self, levels: tuple[str, ...]) -> None:
        self.levels = levels
        self.episodes = 0
        self.reviews = 0

    def target(self, frontier: str | None, curriculum: ReverseCurriculum) -> str | None:
        self.episodes += 1
        if frontier is None or self.episodes % REVIEW_INTERVAL:
            return frontier
        beaten = [level for level in self.levels if level != frontier and curriculum.is_mastered(level)]
        if not beaten:
            return frontier
        self.reviews += 1
        return beaten[(self.reviews - 1) % len(beaten)]
