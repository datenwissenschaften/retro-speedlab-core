from pathlib import Path

from loguru import logger

from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.routes import FULL_RUN, Leg, Part, Routes, load_progress, save_progress
from datenwissenschaften.tas.run import Tas, tas_run, worker_pool
from datenwissenschaften.tas.runner import Start


class LevelTrainer:
    def __init__(self, tas: Tas, routes: Routes, power_on: Start) -> None:
        self.tas = tas
        self.routes = routes
        self.power_on = power_on
        if not routes.path.is_file():
            routes.split(load_progress(routes.progress), power_on)
            save_progress(routes.progress, [])

    def leg(self, part: Part) -> Leg:
        entry = self.routes.entry(part, self.power_on)
        if entry is None:
            raise RuntimeError(f"{part.name} has no start yet: the part before it is not beaten")
        return Leg(part.name, entry, part.exit)

    def beat_levels(self) -> None:
        for part in self.routes.parts:
            self.tas.extend(self.leg(part), self.routes.segments(part.name))
        logger.info(f"All {len(self.routes.parts)} parts are beaten on their own")

    def retrain(self) -> None:
        for part in self.routes.parts:
            self.tas.improve(self.leg(part), self.routes.segments(part.name))

    def full_run(self) -> None:
        leg = Leg(FULL_RUN, self.power_on, None)
        segments = load_progress(self.routes.progress) or self.routes.stitched()
        self.tas.improve(leg, self.tas.extend(leg, segments))

    def forever(self) -> None:
        self.beat_levels()
        while True:
            self.retrain()
            self.full_run()


def speedrun_forever(wrapper: str, config: Path) -> None:
    with worker_pool(load_config(config).speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        LevelTrainer(tas, tas.routes, tas.runner.power_on()).forever()
