import argparse
import importlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import stable_retro
from loguru import logger

HOLD_DECISIONS = 3
PERCENTILES = (50, 90, 99)
TOP_CANDIDATES = 5
RAM_SIZE = 0x800
BYTE_RANGE = 256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the RAM bytes that hold a sprite's screen position.")
    parser.add_argument("--game", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--starts", type=Path, nargs="*", required=True, help="extra savestates to sample from")
    parser.add_argument("--actions", required=True, help="module:ATTRIBUTE of the (actions, frames, buttons) table")
    parser.add_argument("--color", required=True, help="exact sprite color as r,g,b")
    parser.add_argument("--min-area", type=int, required=True, help="blob area that is certainly the sprite")
    parser.add_argument("--max-fill", type=float, required=True, help="reject solid rectangles, e.g. 0.9")
    parser.add_argument("--playfield-bottom", type=int, required=True, help="first HUD row, excluded from the search")
    parser.add_argument("--decisions", type=int, required=True, help="decisions per start")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def is_sprite(blob: np.ndarray, arguments: argparse.Namespace) -> bool:
    _, _, width, height, area = blob
    return area >= arguments.min_area and area < arguments.max_fill * width * height


def sprite_center(frame: np.ndarray, color: np.ndarray, arguments: argparse.Namespace) -> tuple[float, float] | None:
    mask = np.all(frame[: arguments.playfield_bottom] == color, axis=2).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = [stats[index] for index in range(1, count) if is_sprite(stats[index], arguments)]
    if len(blobs) != 1:
        return None
    left, top, width, height, _ = blobs[0]
    return left + width / 2, top + height / 2


def sample(arguments: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    module_name, attribute = arguments.actions.split(":")
    actions = getattr(importlib.import_module(module_name), attribute)
    env = stable_retro.make(arguments.game, state=arguments.state, render_mode=None)
    color = np.array([int(channel) for channel in arguments.color.split(",")])
    rng = random.Random(arguments.seed)
    rams, centers = [], []
    for start in (None, *arguments.starts):
        env.reset(seed=arguments.seed)
        if start is not None:
            env.em.set_state(start.read_bytes())
            env.data.reset()
            env.data.update_ram()
        action = rng.randrange(len(actions))
        for decision in range(arguments.decisions):
            if decision % HOLD_DECISIONS == 0:
                action = rng.randrange(len(actions))
            for buttons in actions[action]:
                frame, _, terminated, truncated, _ = env.step(buttons)
            if terminated or truncated:
                break
            center = sprite_center(frame, color, arguments)
            if center is not None:
                rams.append(env.get_ram()[:RAM_SIZE].astype(np.int64))
                centers.append(center)
    return np.array(rams), np.array(centers)


def fit(values: np.ndarray, target: np.ndarray) -> dict[str, object]:
    offset = float(np.median(target - values))
    errors = np.abs(target - values - offset)
    return {"offset": offset, "error_percentiles": [round(float(p), 2) for p in np.percentile(errors, PERCENTILES)]}


def single_bytes(rams: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    moving = [address for address in range(rams.shape[1]) if rams[:, address].std() > 0]
    correlations = {address: abs(np.corrcoef(rams[:, address], target)[0, 1]) for address in moving}
    best = sorted(correlations, key=correlations.get, reverse=True)[:TOP_CANDIDATES]
    return [{"address": hex(a), "correlation": round(correlations[a], 3), **fit(rams[:, a], target)} for a in best]


def byte_differences(rams: np.ndarray, target: np.ndarray, anchor: int) -> list[dict[str, object]]:
    results = []
    for address in range(rams.shape[1]):
        if address == anchor or rams[:, address].std() == 0:
            continue
        values = (rams[:, address] - rams[:, anchor]) % BYTE_RANGE
        results.append({"formula": f"({hex(address)} - {hex(anchor)}) % 256", **fit(values, target)})
    return sorted(results, key=lambda result: result["error_percentiles"][1])[:TOP_CANDIDATES]


def main() -> None:
    arguments = parse_arguments()
    rams, centers = sample(arguments)
    if len(rams) == 0:
        raise RuntimeError("The sprite color was never seen as exactly one blob; adjust --color or --min-area.")
    report = {"samples": len(rams), "x": single_bytes(rams, centers[:, 0]), "y": single_bytes(rams, centers[:, 1])}
    anchor = int(report["y"][0]["address"], 16)
    report["y_minus_height"] = byte_differences(rams, centers[:, 1], anchor)
    arguments.out.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
