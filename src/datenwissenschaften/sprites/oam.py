from dataclasses import dataclass

OAM_BLOCK = b"SPRA"
OAM_SIZE = 256
PPU_BLOCK = b"PPUR"
PPU_SIZE = 4
SIZE_BYTES = 4
SPRITE_BYTES = 4
PALETTE_BITS = 0x03
TALL_SPRITES = 0x20
HIDDEN_Y = 0xEF
TILE = 8


@dataclass(slots=True, frozen=True)
class Sprite:
    x: int
    y: int
    height: int
    tile: int
    palette: int


def block(state: bytes, name: bytes, size: int) -> bytes:
    header = name + size.to_bytes(SIZE_BYTES, "little")
    found = state.find(header)
    if found < 0:
        raise ValueError(f"The emulator state has no {name.decode()} block of {size} bytes; is it an FCEUmm state?")
    start = found + len(header)
    return state[start : start + size]


def sprite_height(state: bytes) -> int:
    return 2 * TILE if block(state, PPU_BLOCK, PPU_SIZE)[0] & TALL_SPRITES else TILE


def sprites_of(state: bytes) -> list[Sprite]:
    oam, height = block(state, OAM_BLOCK, OAM_SIZE), sprite_height(state)
    entries = (oam[index : index + SPRITE_BYTES] for index in range(0, OAM_SIZE, SPRITE_BYTES))
    return [
        Sprite(x, y + 1, height, tile, attributes & PALETTE_BITS) for y, tile, attributes, x in entries if y < HIDDEN_Y
    ]
