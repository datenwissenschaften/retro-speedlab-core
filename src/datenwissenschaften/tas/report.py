from dataclasses import dataclass
from typing import Any

import msgspec

from datenwissenschaften.environment.levels import level_map
from datenwissenschaften.settings import GameSettings
from datenwissenschaften.tas.journal import Split
from datenwissenschaften.tas.movie import Segment, frames
from datenwissenschaften.tas.runner import SegmentRunner
from datenwissenschaften.tas.stuck import Stuck

SECTION = "speedrun"
START_BUTTON = "START"


@dataclass(slots=True, frozen=True)
class Course:
    levels: dict[str, tuple[str, ...]]
    first: str
    frame_rate: float
    goals: dict[str, str]
    start_button: int
    game: GameSettings


def course_of(runner: SegmentRunner, game: GameSettings) -> Course:
    wrapper, emulator = runner.wrapper, runner.emulator
    buttons = list(emulator.buttons)
    if START_BUTTON not in buttons:
        raise RuntimeError(f"{emulator.gamename} has no {START_BUTTON} button to time the run from: {buttons}")
    goals = {state_cls.__name__: state_cls.goal for state_cls in runner.classes if hasattr(state_cls, "goal")}
    levels = level_map(wrapper.levels, wrapper.state_classes)
    rate = emulator.em.get_screen_rate()
    return Course(levels, runner.order[0], rate, goals, buttons.index(START_BUTTON), game)


def splits(course: Course, segments: list[Segment]) -> list[Split]:
    leaving = {states[-1]: level for level, states in course.levels.items()}
    opened = next((states[0] for states in course.levels.values()), None)
    found, played, origin, previous = [], 0, course.first, 0.0
    for segment in segments:
        played += len(segment.pressed)
        seconds = played / course.frame_rate
        if origin in leaving or (segment.state == opened and not found):
            found.append(Split(leaving[origin] if origin in leaving else course.first, seconds, seconds - previous))
            previous = seconds
        origin = segment.state
    return found


def start_frame(segments: list[Segment], button: int) -> int | None:
    played = 0
    for segment in segments:
        for frame, buttons in enumerate(segment.pressed):
            if buttons[button]:
                return played + frame
        played += len(segment.pressed)
    return None


def timing(course: Course, segments: list[Segment]) -> dict[str, Any]:
    played, start = frames(segments), start_frame(segments, course.start_button)
    return {
        "power_on_seconds": played / course.frame_rate,
        "start_seconds": None if start is None else (played - start) / course.frame_rate,
        "start_frame": start,
    }


def objectives(course: Course, searching: str) -> list[dict[str, str]]:
    found = [{"level": "game", "text": f"Beat the game ({course.game.category})"}]
    found += [
        {"level": "level", "text": f"Beat {level}"} for level, states in course.levels.items() if searching in states
    ]
    if searching in course.goals:
        found.append({"level": "state", "text": course.goals[searching]})
    return found


def game_field(game: GameSettings) -> dict[str, Any]:
    sources = {"speedrun_com": msgspec.to_builtins(game.speedrun_com), "tasvideos": game.tasvideos}
    return {
        "id": game.id,
        "title": game.title,
        "platform": game.platform,
        "category": game.category,
        "sources": sources,
    }


def report(
    course: Course,
    segments: list[Segment],
    found: list[Split],
    firsts: dict[str, float],
    searching: str,
    stuck: Stuck | None,
) -> dict[str, Any]:
    played = frames(segments)
    return {
        "frames": played,
        "seconds": played / course.frame_rate,
        "frame_rate": course.frame_rate,
        "segments": [{"state": segment.state, "frames": len(segment.pressed)} for segment in segments],
        "splits": [{**msgspec.to_builtins(split), "first_seconds": firsts[split.level]} for split in found],
        "levels": [{"name": level, "states": list(states)} for level, states in course.levels.items()],
        "goals": course.goals,
        "opening": course.first,
        "reached": segments[-1].state if segments else course.first,
        "won": bool(segments) and segments[-1].won,
        "searching": searching,
        "stuck": msgspec.to_builtins(stuck),
        "objectives": objectives(course, searching),
        "game": game_field(course.game),
        "timing": timing(course, segments),
    }
