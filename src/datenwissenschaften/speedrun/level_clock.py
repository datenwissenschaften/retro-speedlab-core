import json
import sys
from pathlib import Path

from speedlab_engine.cartridge import RAM_SIZE, Cartridge, load_cartridge
from speedlab_engine.pad import unpacked


def level_entries(cartridge: Cartridge, inputs: bytes) -> dict[str, list[list[int]] | int | None]:
    ram = bytearray(RAM_SIZE)
    cartridge.reset(ram, False)
    level = cartridge.level(bytes(ram))
    entries = [[0, level]]
    for frame, pressed in enumerate(inputs, 1):
        cartridge.step(ram, unpacked(pressed))
        if cartridge.level(bytes(ram)) != level:
            level = cartridge.level(bytes(ram))
            entries.append([frame, level])
        if cartridge.won(bytes(ram)):
            return {"levels": entries, "won_at": frame}
    return {"levels": entries, "won_at": None}


def clocked(collection: Path, name: str, inputs: bytes) -> dict[str, object]:
    cartridge = load_cartridge(collection, name)
    return {"cartridge": name, "version": cartridge.VERSION, **level_entries(cartridge, inputs)}


if __name__ == "__main__":
    sys.stdout.write(json.dumps(clocked(Path.cwd(), sys.argv[1], sys.stdin.buffer.read())))
