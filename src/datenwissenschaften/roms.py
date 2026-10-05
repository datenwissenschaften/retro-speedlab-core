from pathlib import Path

import stable_retro as retro


def import_roms(roms_dir: Path, integrations_dir: Path) -> None:
    retro.data.add_custom_integration(str(integrations_dir))
    retro.data.merge(*(str(rom) for rom in sorted(roms_dir.iterdir()) if rom.is_file()), quiet=True)
