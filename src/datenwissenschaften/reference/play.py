import numpy as np

BLACK = 32
DARK_SHARE = 0.7
BREAK_SECONDS = 2
FLASH_SECONDS = 1.5

Span = tuple[int, int]


def dark_screens(screens: np.ndarray) -> np.ndarray:
    return (screens.max(3) < BLACK).mean((1, 2)) >= DARK_SHARE


def dark_runs(dark: np.ndarray) -> list[Span]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], dark.astype(np.int8), [0]])))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def joined(runs: list[Span], flash: float) -> list[Span]:
    spans: list[Span] = []
    for start, end in runs:
        if spans and start - spans[-1][1] < flash:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def breaks(dark: np.ndarray, frames_per_second: float) -> list[Span]:
    spans = joined(dark_runs(dark), frames_per_second * FLASH_SECONDS)
    return [(start, end) for start, end in spans if end - start >= frames_per_second * BREAK_SECONDS]


def level_spans(dark: np.ndarray, level_count: int, frames_per_second: float) -> list[Span]:
    found = breaks(dark, frames_per_second)
    spans = [(before[1], after[0]) for before, after in zip(found, found[1:], strict=False)]
    if len(spans) != level_count:
        raise RuntimeError(
            f"The video shows {len(spans)} levels between {len(found)} dark score screens, not {level_count}"
        )
    return spans
