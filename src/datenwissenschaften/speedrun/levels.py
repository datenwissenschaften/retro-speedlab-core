import multiprocessing
from pathlib import Path

from loguru import logger

from datenwissenschaften.reference.fetch import ensure_reference
from datenwissenschaften.route.paths import route_path
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.practice import practised, write_practice_route
from datenwissenschaften.speedrun.prior import PRIOR_FILE, write_prior
from datenwissenschaften.speedrun.reel import Reel
from datenwissenschaften.speedrun.routes import FULL_RUN, Leg, Part, Routes, finished, load_progress, save_progress
from datenwissenschaften.speedrun.run import Tas, tas_run, worker_pool
from datenwissenschaften.speedrun.runner import Start


class LevelTrainer:
    def __init__(self, tas: Tas, routes: Routes, power_on: Start, reel: Reel) -> None:
        self.tas = tas
        self.reel = reel
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

    def unbeaten(self) -> list[Part]:
        return [part for part in self.routes.parts if not finished(self.routes.segments(part.name), part.exit)]

    def share_practice(self) -> None:
        done = practised(self.tas.runner, self.routes)
        write_practice_route(self.tas.runner, self.routes, route_path(self.tas.settings), done)
        write_prior(self.tas.runner, self.routes, self.tas.output / PRIOR_FILE)
        self.tas.feed.practised(done)
        self.reel.show()

    def beat_levels(self) -> None:
        self.share_practice()
        while unbeaten := self.unbeaten():
            part = unbeaten[0]
            self.tas.extend(self.leg(part), self.routes.segments(part.name), None)
            self.share_practice()
        logger.info(f"All {len(self.routes.parts)} parts are beaten one after the other")

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


def start_policy(wrapper: str, config: Path) -> None:
    from datenwissenschaften.policy.trainer import train_policy_forever

    trainer = multiprocessing.get_context("spawn").Process(
        target=train_policy_forever, args=(wrapper, config), name="policy", daemon=False
    )
    trainer.start()
    logger.info(f"The policy trains with PPO in process {trainer.pid}")


def speedrun_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    ensure_reference(settings)
    if settings.speedrun.policy:
        start_policy(wrapper, config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        reel = Reel(settings, tas.feed, tas.runner, tas.routes)
        LevelTrainer(tas, tas.routes, tas.runner.power_on(), reel).forever()
