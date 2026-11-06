from dataclasses import dataclass

import cv2
import numpy as np

from datenwissenschaften.knowledge import Knowledge, player_sprite, sprite_names
from datenwissenschaften.sprites.objects import SpriteObject

PIXEL_SCALE = 2
UNKNOWN = "?"
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.8
LINE = 1
LABEL_PADDING = 2
PLAYER_COLOR = (255, 220, 0)
NAMED_COLOR = (0, 230, 255)
UNKNOWN_COLOR = (255, 90, 200)
LABEL_BACKGROUND = (0, 0, 0)


@dataclass(slots=True, frozen=True)
class Labels:
    names: dict[str, str]
    player: str | None


def labels_of(knowledge: Knowledge) -> Labels:
    return Labels(sprite_names(knowledge), player_sprite(knowledge))


def look_of(shown: SpriteObject, labels: Labels) -> tuple[str, tuple[int, int, int]]:
    if shown.signature not in labels.names:
        return UNKNOWN, UNKNOWN_COLOR
    color = PLAYER_COLOR if shown.signature == labels.player else NAMED_COLOR
    return labels.names[shown.signature], color


def label(frame: np.ndarray, text: str, color: tuple[int, int, int], left: int, top: int) -> None:
    (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, LINE)
    bottom = max(top, height + 2 * LABEL_PADDING)
    corner = (left + width + 2 * LABEL_PADDING, bottom - height - 2 * LABEL_PADDING)
    cv2.rectangle(frame, (left, bottom), corner, LABEL_BACKGROUND, cv2.FILLED)
    cv2.putText(frame, text, (left + LABEL_PADDING, bottom - LABEL_PADDING), FONT, FONT_SCALE, color, LINE)


def annotate(screen: np.ndarray, objects: list[SpriteObject], labels: Labels) -> np.ndarray:
    frame = np.ascontiguousarray(screen.repeat(PIXEL_SCALE, axis=0).repeat(PIXEL_SCALE, axis=1))
    for shown in objects:
        text, color = look_of(shown, labels)
        left, top = shown.left * PIXEL_SCALE, shown.top * PIXEL_SCALE
        corner = (shown.right * PIXEL_SCALE - 1, shown.bottom * PIXEL_SCALE - 1)
        cv2.rectangle(frame, (left, top), corner, color, LINE)
        label(frame, text, color, left, top)
    return frame
