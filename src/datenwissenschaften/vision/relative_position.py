from math import hypot

from datenwissenschaften.vision.detection import Detection

ALIGNED_PIXELS = 8


def describe_target(actor: Detection | None, targets: tuple[Detection, ...]) -> dict[str, object]:
    if actor is None or not targets:
        return {"visible": bool(targets), "direction": "unknown", "distance": None}
    actor_x, actor_y = actor.center
    nearest = min(targets, key=lambda target: hypot(target.center[0] - actor_x, target.center[1] - actor_y))
    delta_x, delta_y = nearest.center[0] - actor_x, nearest.center[1] - actor_y
    return {"visible": True, "direction": _direction(delta_x, delta_y), "distance": round(hypot(delta_x, delta_y))}


def nearest_distance(actor: Detection | None, targets: tuple[Detection, ...]) -> float | None:
    if actor is None or not targets:
        return None
    actor_x, actor_y = actor.center
    return min(hypot(target.center[0] - actor_x, target.center[1] - actor_y) for target in targets)


def _direction(delta_x: float, delta_y: float) -> str:
    vertical = "" if abs(delta_y) < ALIGNED_PIXELS else "up" if delta_y < 0 else "down"
    horizontal = "" if abs(delta_x) < ALIGNED_PIXELS else "left" if delta_x < 0 else "right"
    return "-".join(part for part in (vertical, horizontal) if part) or "here"
