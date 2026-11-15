from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from speedlab_engine.cartridge import SCREEN_HEIGHT, SCREEN_WIDTH
from speedlab_engine.video import PIXEL_SCALE

from datenwissenschaften.speedrun.journal import Split, clock, spoken
from datenwissenschaften.speedrun.sprite_matches import Match

PANEL_WIDTH = 160
WIDTH = (SCREEN_WIDTH + PANEL_WIDTH) * PIXEL_SCALE
HEIGHT = SCREEN_HEIGHT * PIXEL_SCALE
GAME_WIDTH = SCREEN_WIDTH * PIXEL_SCALE
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
LABEL_SCALE = 0.8
CLOCK_SCALE = 2.4
LINE = 18
MARGIN = 14
COLUMN = 200
Color = tuple[int, int, int]
BACKGROUND: Color = (16, 16, 28)
TEXT: Color = (236, 236, 236)
DIM: Color = (150, 150, 176)
HEADING: Color = (255, 204, 0)
HUES = 180
SATURATED = 200
BRIGHT = 255


@dataclass(slots=True, frozen=True)
class Moment:
    frame: int
    seconds: float
    level: str
    cleared: list[Split]
    memory: dict[str, int | None]


def write(canvas: NDArray[np.uint8], text: str, at: tuple[int, int], color: Color, scale: float) -> None:
    cv2.putText(canvas, text, at, FONT, scale, color, 1, cv2.LINE_8)


def sprite_colors(names: list[str]) -> dict[str, Color]:
    hues = np.asarray([[[HUES * index // len(names), SATURATED, BRIGHT]] for index in range(len(names))], np.uint8)
    rgb = np.asarray(cv2.cvtColor(hues, cv2.COLOR_HSV2RGB), dtype=np.uint8).reshape(-1, 3)
    return {name: (int(red), int(green), int(blue)) for name, (red, green, blue) in zip(names, rgb, strict=True)}


def draw_matches(canvas: NDArray[np.uint8], matches: list[Match], colors: dict[str, Color]) -> None:
    for match in matches:
        color = colors[match.name]
        left, top = match.x * PIXEL_SCALE, match.y * PIXEL_SCALE
        right, bottom = left + match.width * PIXEL_SCALE - 1, top + match.height * PIXEL_SCALE - 1
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 1, cv2.LINE_8)
        write(canvas, match.name, (left, max(top - 3, LINE // 2)), color, LABEL_SCALE)


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


def annotated(
    screen: NDArray[np.uint8], matches: list[Match], colors: dict[str, Color], title: str, moment: Moment
) -> NDArray[np.uint8]:
    canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    canvas[:, :GAME_WIDTH] = np.repeat(np.repeat(screen, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)
    draw_matches(canvas, matches, colors)
    draw_panel(canvas, title, moment)
    return canvas
