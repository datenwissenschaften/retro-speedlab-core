from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from speedlab_engine.cartridge import SCREEN_HEIGHT, SCREEN_WIDTH
from speedlab_engine.video import PIXEL_SCALE

from datenwissenschaften.speedrun.journal import Split, clock, spoken

PANEL_WIDTH = 160
WIDTH = (SCREEN_WIDTH + PANEL_WIDTH) * PIXEL_SCALE
HEIGHT = SCREEN_HEIGHT * PIXEL_SCALE
GAME_WIDTH = SCREEN_WIDTH * PIXEL_SCALE
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
CLOCK_SCALE = 2.4
LINE = 18
MARGIN = 14
COLUMN = 200
Color = tuple[int, int, int]
BACKGROUND: Color = (16, 16, 28)
TEXT: Color = (236, 236, 236)
DIM: Color = (150, 150, 176)
HEADING: Color = (255, 204, 0)


@dataclass(slots=True, frozen=True)
class Moment:
    frame: int
    seconds: float
    level: str
    cleared: list[Split]
    memory: dict[str, int | None]


def write(canvas: NDArray[np.uint8], text: str, at: tuple[int, int], color: Color, scale: float) -> None:
    cv2.putText(canvas, text, at, FONT, scale, color, 1, cv2.LINE_8)


def panel_lines(title: str, moment: Moment) -> list[tuple[str, str, Color]]:
    lines: list[tuple[str, str, Color]] = [
        (title.upper(), "", HEADING),
        (spoken(moment.level), "", TEXT),
        ("", "", TEXT),
        ("MEMORY", "", HEADING),
    ]
    lines += [(label, "-" if value is None else str(value), TEXT) for label, value in moment.memory.items()]
    lines += [("", "", TEXT), ("SPLITS", "", HEADING)]
    return lines + [(spoken(split.level), clock(split.seconds), DIM) for split in moment.cleared]


def draw_panel(canvas: NDArray[np.uint8], title: str, moment: Moment) -> None:
    left = GAME_WIDTH + MARGIN
    write(canvas, clock(moment.seconds), (left, MARGIN + 2 * LINE), TEXT, CLOCK_SCALE)
    lines = panel_lines(title, moment)
    shown = (HEIGHT - 4 * LINE) // LINE
    kept = lines[:shown] if len(lines) <= shown else [*lines[: shown - 1], lines[-1]]
    for row, (label, value, color) in enumerate(kept):
        y = MARGIN + (row + 4) * LINE
        write(canvas, label, (left, y), color, TEXT_SCALE)
        write(canvas, value, (left + COLUMN, y), color, TEXT_SCALE)


def annotated(screen: NDArray[np.uint8], title: str, moment: Moment) -> NDArray[np.uint8]:
    canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    canvas[:, :GAME_WIDTH] = np.repeat(np.repeat(screen, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
    draw_panel(canvas, title, moment)
    return canvas
