import argparse
from pathlib import Path

import stable_retro as retro
from stable_retro.scripts.playback_movie import _play as play_movie

from datenwissenschaften.roms import import_roms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roms-dir", type=Path, required=True)
    parser.add_argument("--integrations-dir", type=Path, required=True)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("movies", nargs="+")
    args = parser.parse_args()
    args.lossless = None
    args.no_video = False
    args.info_dict = False
    args.npy_actions = False
    args.viewer = None
    args.ending = None

    import_roms(args.roms_dir, args.integrations_dir)
    original_make = retro.make

    def make_headless(*args, **kwargs):
        kwargs["render_mode"] = "rgb_array"
        return original_make(*args, **kwargs)

    retro.make = make_headless
    try:
        for movie in args.movies:
            play_movie(movie, args, None)
    finally:
        retro.make = original_make


if __name__ == "__main__":
    main()
