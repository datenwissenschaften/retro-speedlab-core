import numpy as np

BLACK = 32
DARK_SHARE = 0.7
BREAK_FRAMES = 180

Span = tuple[int, int]


def dark_screens(screens: np.ndarray) -> np.ndarray:
    return (screens.max(3) < BLACK).mean((1, 2)) >= DARK_SHARE


def breaks(dark: np.ndarray) -> list[Span]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], dark.astype(np.int8), [0]])))
    return [
        (int(start), int(end))
        for start, end in zip(edges[::2], edges[1::2], strict=True)
        if end - start >= BREAK_FRAMES
    ]


def level_spans(dark: np.ndarray, level_count: int) -> list[Span]:
    found = breaks(dark)
    spans = [(before[1], after[0]) for before, after in zip(found, found[1:], strict=False)]
    if len(spans) != level_count:
        raise RuntimeError(
            f"The video shows {len(spans)} levels between {len(found)} dark score screens, not {level_count}"
        )
    return spans
