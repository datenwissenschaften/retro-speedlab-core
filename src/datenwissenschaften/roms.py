import shutil
import subprocess
import sys
from pathlib import Path

import stable_retro


def import_roms(roms_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "stable_retro.import", str(roms_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def import_savestates(game: str, savestates_dir: Path) -> None:
    integration = Path(stable_retro.data.get_file_path(game, "data.json")).parent
    for savestate in savestates_dir.glob("*.state"):
        shutil.copyfile(savestate, integration / savestate.name)
