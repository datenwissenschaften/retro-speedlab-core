import subprocess
import sys
from pathlib import Path


def import_roms(roms_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "stable_retro.import", str(roms_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
