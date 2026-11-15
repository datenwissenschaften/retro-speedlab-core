from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

MATCH_ERROR = 32.0
MOST_CANDIDATES = 64
MOST_MATCHES = 6
OVERLAP = 0.3
ALPHA = 3
LEAST_SCORE = 1.0 / (1.0 + MATCH_ERROR)


@dataclass(slots=True, frozen=True)
class Template:
    image: NDArray[np.float32]
    mask: NDArray[np.float32]
    energy: float
    opaque: int


@dataclass(slots=True, frozen=True)
class Match:
    name: str
    x: int
    y: int
    width: int
    height: int


class Frame:
    def __init__(self, screen: NDArray[np.uint8]) -> None:
        self.image = gray(screen)
        self.squared = self.image * self.image
        self.masked: dict[bytes, NDArray[np.float32]] = {}

    def errors(self, template: Template) -> NDArray[np.float32]:
        key = template.mask.tobytes()
        if key not in self.masked:
            self.masked[key] = correlated(self.squared, template.mask)
        crossed = correlated(self.image, template.image)
        return np.asarray((self.masked[key] - 2.0 * crossed + template.energy) / template.opaque, dtype=np.float32)


def gray(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    return np.asarray(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY), dtype=np.float32)


def correlated(image: NDArray[np.float32], template: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(cv2.matchTemplate(image, template, cv2.TM_CCORR), dtype=np.float32)


def template_of(shade: NDArray[np.float32], opaque: NDArray[np.bool_]) -> Template:
    mask = np.ascontiguousarray(opaque, dtype=np.float32)
    image = np.ascontiguousarray(shade * mask, dtype=np.float32)
    return Template(image, mask, float((image * image).sum()), int(opaque.sum()))


def sprite_templates(name: str, png: bytes) -> list[Template]:
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"The sprite {name} is no image")
    rgba = np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA), dtype=np.uint8)
    opaque: NDArray[np.bool_] = rgba[:, :, ALPHA] > 0
    if not opaque.any():
        raise ValueError(f"The sprite {name} has no opaque pixel to match")
    shade = gray(rgba[:, :, :ALPHA])
    facing, flipped = template_of(shade, opaque), template_of(shade[:, ::-1], opaque[:, ::-1])
    return [facing] if np.array_equal(facing.image, flipped.image) else [facing, flipped]


def templates_of(sprites: dict[str, bytes]) -> dict[str, list[Template]]:
    return {name: sprite_templates(name, png) for name, png in sprites.items()}


def candidates(frame: Frame, template: Template) -> list[tuple[list[int], float]]:
    errors = np.maximum(frame.errors(template), 0.0)
    ys, xs = np.nonzero(errors <= MATCH_ERROR)
    best = np.argsort(errors[ys, xs])[:MOST_CANDIDATES]
    height, width = template.image.shape
    return [([int(xs[i]), int(ys[i]), width, height], 1.0 / (1.0 + float(errors[ys[i], xs[i]]))) for i in best]


def matches_of(name: str, frame: Frame, templates: list[Template]) -> list[Match]:
    found = [candidate for template in templates for candidate in candidates(frame, template)]
    if not found:
        return []
    boxes, scores = [box for box, _ in found], [score for _, score in found]
    kept = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, LEAST_SCORE, OVERLAP, top_k=MOST_MATCHES), dtype=np.int64)
    return [Match(name, *boxes[int(index)]) for index in kept.flatten()]


def sprite_matches(screen: NDArray[np.uint8], sprites: dict[str, list[Template]]) -> list[Match]:
    frame = Frame(screen)
    return [match for name, templates in sprites.items() for match in matches_of(name, frame, templates)]
