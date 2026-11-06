import hashlib
from dataclasses import dataclass

import numpy as np

from datenwissenschaften.sprites.oam import TILE, Sprite, sprites_of

NES_WIDTH = 256
NES_HEIGHT = 240
SIGNATURE_BYTES = 4


@dataclass(slots=True, frozen=True)
class SpriteObject:
    signature: str
    left: int
    top: int
    right: int
    bottom: int


def touching(first: Sprite, second: Sprite) -> bool:
    return (
        first.x <= second.x + TILE
        and second.x <= first.x + TILE
        and first.y <= second.y + second.height
        and second.y <= first.y + first.height
    )


def groups(sprites: list[Sprite]) -> list[list[Sprite]]:
    left, found = list(sprites), list[list[Sprite]]()
    while left:
        group = [left.pop(0)]
        for member in group:
            joined = [index for index, sprite in enumerate(left) if touching(member, sprite)]
            group.extend(left[index] for index in joined)
            left = [sprite for index, sprite in enumerate(left) if index not in joined]
        found.append(group)
    return found


def signature(group: list[Sprite]) -> str:
    tiles = sorted({(sprite.tile, sprite.palette) for sprite in group})
    content = bytes(value for pair in tiles for value in pair)
    return hashlib.blake2b(content, digest_size=SIGNATURE_BYTES).hexdigest()


def placed(group: list[Sprite], screen: np.ndarray) -> SpriteObject | None:
    height, width = screen.shape[:2]
    shift_x, shift_y = (NES_WIDTH - width) // 2, (NES_HEIGHT - height) // 2
    left = max(0, min(sprite.x for sprite in group) - shift_x)
    top = max(0, min(sprite.y for sprite in group) - shift_y)
    right = min(width, max(sprite.x + TILE for sprite in group) - shift_x)
    bottom = min(height, max(sprite.y + sprite.height for sprite in group) - shift_y)
    if right <= left or bottom <= top:
        return None
    return SpriteObject(signature(group), left, top, right, bottom)


def objects_in(state: bytes, screen: np.ndarray) -> list[SpriteObject]:
    found = (placed(group, screen) for group in groups(sprites_of(state)))
    return [shown for shown in found if shown is not None]


def image_of(shown: SpriteObject, screen: np.ndarray) -> np.ndarray:
    return screen[shown.top : shown.bottom, shown.left : shown.right]
