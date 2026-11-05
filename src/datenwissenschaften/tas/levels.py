import multiprocessing
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
from datenwissenschaften.tas.prior import PRIOR_FILE, write_prior
from datenwissenschaften.tas.reel import Reel
from datenwissenschaften.tas.routes import FULL_RUN, Leg, Part, Routes, finished, load_progress, save_progress
from datenwissenschaften.tas.run import Tas, tas_run, worker_pool
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.turns import COURSE_FILE, Turn, changed_level, load_turn, save_turn

PRACTICE_SEARCHES = 3


class LevelTrainer:
    def __init__(self, tas: Tas, routes: Routes, power_on: Start, loader: LevelLoader | None, reel: Reel) -> None:
        self.tas = tas
        self.reel = reel
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

    def in_turn(self, unbeaten: list[Part]) -> list[Part]:
        turn = load_turn(self.tas.turn_file)
        names = [part.name for part in self.routes.parts]
        if turn is None or turn.part not in names:
            return unbeaten
        first = names.index(turn.part) + turn.over(PRACTICE_SEARCHES)
        later = [part for part in unbeaten if names.index(part.name) >= first]
        return [*later, *(part for part in unbeaten if part not in later)]

    def share_practice(self) -> None:
        done = practised(self.tas.runner, self.routes)
        write_practice_route(self.tas.runner, self.routes, route_path(self.tas.settings), done)
        write_prior(self.tas.runner, self.routes, self.tas.output / PRIOR_FILE)
        self.tas.feed.practised(done)
        self.reel.show()

    def practise(self, part: Part) -> None:
        entry = self.routes.reached(part) or self.entry(part)
        if entry is None:
            return
        self.tas.extend(Leg(part.name, entry, part.exit), self.routes.segments(part.name), PRACTICE_SEARCHES)
        self.share_practice()

    def beat_levels(self) -> None:
        self.share_practice()
        while unbeaten := self.unbeaten():
            for part in self.in_turn(unbeaten):
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


def turn_to_changed_level(tas: Tas) -> None:
    changed = changed_level(tas.output / COURSE_FILE, tas.feed.course.levels)
    if changed is None:
        return
    logger.info(f"The game package changed the states of {changed}; it takes the next turn")
    save_turn(tas.turn_file, Turn(changed, 0))


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
        loader = ensure_loader(tas.output / LOADER_FILE, tas.runner, tas.routes)
        ensure_guide(settings, tas.runner, tas.routes, loader)
        reel = Reel(settings, tas.feed, tas.runner, tas.routes)
        turn_to_changed_level(tas)
        LevelTrainer(tas, tas.routes, tas.runner.power_on(), loader, reel).forever()
