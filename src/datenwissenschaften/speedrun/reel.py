from functools import partial

import msgspec

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.feed import SpeedrunFeed
from datenwissenschaften.speedrun.footage import MOVIE_MEMORY, movie_field, render_movie
from datenwissenschaften.speedrun.movie import verified_segments
from datenwissenschaften.speedrun.paths import tas_dir
from datenwissenschaften.speedrun.routes import Part, Routes
from datenwissenschaften.speedrun.run import video_upload
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.screen import Cut, Screen

REEL_DIR = "levels"
ROUTE_MOVIE = "route.bk2"


def level_video(number: int) -> str:
    return f"{REEL_DIR}/{number}.mp4"


class Reel:
    def __init__(
        self, settings: RetroSpeedlabConfig, feed: SpeedrunFeed, runner: SegmentRunner, routes: Routes
    ) -> None:
        self.feed = feed
        self.runner = runner
        self.routes = routes
        self.directory = tas_dir(settings) / REEL_DIR
        self.screens: dict[str, tuple[int, Screen]] = {}
        for number, part in enumerate(routes.parts[1:], 1):
            folder = self.directory / str(number)
            folder.mkdir(parents=True, exist_ok=True)
            render = partial(render_movie, settings.paths, folder)
            upload = video_upload(settings, level_video(number))
            shown = Screen(level_video(number), render, upload, partial(feed.reeled, part.name))
            shown.start()
            self.screens[part.name] = (number, shown)

    def show(self) -> None:
        for part in self.routes.parts[1:]:
            segments = self.routes.segments(part.name)
            number, shown = self.screens[part.name]
            key = msgspec.msgpack.encode([segment.pressed for segment in segments])
            if not segments:
                self.feed.unreeled(part.name)
            elif key != shown.queued:
                shown.queue(self.cut(part, number, key))

    def cut(self, part: Part, number: int, key: bytes) -> Cut:
        segments = self.routes.segments(part.name)
        start = Start(part.entry_state, self.routes.routes[part.name].entry)
        movie = self.directory / str(number) / ROUTE_MOVIE
        replay = verified_segments(self.runner, start, segments, movie, self.feed.catalog)
        field = movie_field(start.state, self.feed.course.frame_rate, segments)
        shown = {**field, "part": part.name, "number": number, "video": level_video(number)}
        return Cut(key, (movie.read_bytes(),), shown, {MOVIE_MEMORY: replay.memory})
