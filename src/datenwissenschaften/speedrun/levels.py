import multiprocessing
from functools import partial
from multiprocessing.process import BaseProcess
from pathlib import Path

from loguru import logger

from datenwissenschaften.reference.fetch import ensure_reference
from datenwissenschaften.reference.matching import ReferenceMatching
from datenwissenschaften.route.paths import route_path
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.polish import Polisher
from datenwissenschaften.speedrun.practice import practised, write_practice_route
from datenwissenschaften.speedrun.prior import PRIOR_FILE, write_prior
from datenwissenschaften.speedrun.reel import Reel
from datenwissenschaften.speedrun.routes import FULL_RUN, Leg, Part, Routes, finished, load_progress, save_progress
from datenwissenschaften.speedrun.run import Tas, frontier_workers, never_rest, tas_run, worker_pool
from datenwissenschaften.speedrun.runner import Start


class LevelTrainer:
    def __init__(self, tas: Tas, routes: Routes, power_on: Start, reel: Reel, matching: ReferenceMatching) -> None:
        self.tas = tas
        self.reel = reel
        self.matching = matching
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
        self.matching.refresh()

    def rest(self, polisher: Polisher, part: Part) -> bool:
        self.tas.feed.practised(practised(self.tas.runner, self.routes))
        return polisher.rest(part)

    def beat_levels(self) -> None:
        self.share_practice()
        polisher = Polisher(self.tas, 0)
        while unbeaten := self.unbeaten():
            part = unbeaten[0]
            self.tas.extend(self.leg(part), self.routes.segments(part.name), None, partial(self.rest, polisher, part))
            self.share_practice()
        logger.info(f"All {len(self.routes.parts)} parts are beaten one after the other")

    def retrain(self) -> None:
        for part in self.routes.parts:
            self.tas.improve(self.leg(part), self.routes.segments(part.name))
        self.share_practice()

    def full_run(self) -> None:
        leg = Leg(FULL_RUN, self.power_on, None)
        segments = load_progress(self.routes.progress) or self.routes.stitched()
        self.tas.improve(leg, self.tas.extend(leg, segments, None, never_rest))

    def forever(self) -> None:
        self.beat_levels()
        while True:
            self.retrain()
            self.full_run()


def start_policy(wrapper: str, config: Path) -> BaseProcess:
    from datenwissenschaften.policy.trainer import train_policy_forever

    trainer = multiprocessing.get_context("spawn").Process(
        target=train_policy_forever, args=(wrapper, config), name="policy", daemon=False
    )
    trainer.start()
    logger.info(f"The policy trains with PPO in process {trainer.pid}")
    return trainer


def search_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    with worker_pool(frontier_workers(settings)) as pool, worker_pool(settings.speedrun.polishers) as polishing:
        tas = tas_run(wrapper, config, pool, polishing)
        reel = Reel(settings, tas.feed, tas.runner, tas.routes)
        matching = ReferenceMatching(wrapper, config, pool, tas.feed)
        LevelTrainer(tas, tas.routes, tas.runner.power_on(), reel, matching).forever()


def speedrun_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    ensure_reference(settings)
    if not settings.speedrun.policy:
        search_forever(wrapper, config)
        return
    trainer = start_policy(wrapper, config)
    try:
        search_forever(wrapper, config)
    finally:
        logger.info("The search stopped, so the policy stops too")
        trainer.terminate()
