from pathlib import Path

from loguru import logger

from datenwissenschaften.reference.fetch import ensure_reference
from datenwissenschaften.reference.guide import ensure_guide
from datenwissenschaften.reference.paths import route_path
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.practice import (
    LOADER_FILE,
    LevelLoader,
    ensure_loader,
    level_start,
    practised,
    write_practice_route,
)
from datenwissenschaften.tas.routes import FULL_RUN, Leg, Part, Routes, finished, load_progress, save_progress
from datenwissenschaften.tas.run import Tas, tas_run, worker_pool
from datenwissenschaften.tas.runner import Start

PRACTICE_SEARCHES = 3


class LevelTrainer:
    def __init__(self, tas: Tas, routes: Routes, power_on: Start, loader: LevelLoader | None) -> None:
        self.tas = tas
        self.routes = routes
        self.power_on = power_on
        self.loader = loader
        if not routes.path.is_file():
            routes.split(load_progress(routes.progress), power_on)
            save_progress(routes.progress, [])

    def entry(self, part: Part) -> Start | None:
        entry = self.routes.entry(part, self.power_on)
        if entry is not None or self.loader is None:
            return entry
        return level_start(self.tas.runner, self.loader, self.routes.parts.index(part) - 1, part.entry_state)

    def leg(self, part: Part) -> Leg:
        entry = self.entry(part)
        if entry is None:
            raise RuntimeError(f"{part.name} has no start yet: the part before it is not beaten")
        return Leg(part.name, entry, part.exit)

    def unbeaten(self) -> list[Part]:
        return [part for part in self.routes.parts if not finished(self.routes.segments(part.name), part.exit)]

    def share_practice(self) -> None:
        done = practised(self.tas.runner, self.routes)
        write_practice_route(self.tas.runner, self.routes, route_path(self.tas.settings), done)
        self.tas.feed.practised(done)

    def practise(self, part: Part) -> None:
        entry = self.entry(part)
        if entry is None:
            return
        self.tas.extend(Leg(part.name, entry, part.exit), self.routes.segments(part.name), PRACTICE_SEARCHES)
        self.share_practice()

    def beat_levels(self) -> None:
        self.share_practice()
        while unbeaten := self.unbeaten():
            for part in unbeaten:
                self.practise(part)
        logger.info(f"All {len(self.routes.parts)} parts are beaten on their own")

    def retrain(self) -> None:
        for part in self.routes.parts:
            self.tas.improve(self.leg(part), self.routes.segments(part.name))
        self.share_practice()

    def full_run(self) -> None:
        leg = Leg(FULL_RUN, self.power_on, None)
        segments = load_progress(self.routes.progress) or self.routes.stitched()
        self.tas.improve(leg, self.tas.extend(leg, segments, None))

    def forever(self) -> None:
        self.beat_levels()
        while True:
            self.retrain()
            self.full_run()


def speedrun_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    ensure_reference(settings)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        ensure_guide(settings, tas.runner, tas.routes.stitched())
        loader = ensure_loader(tas.output / LOADER_FILE, tas.runner, tas.routes)
        LevelTrainer(tas, tas.routes, tas.runner.power_on(), loader).forever()
