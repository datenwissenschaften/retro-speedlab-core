import base64
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import typer
from speedlab_engine.collection import cartridge_name
from speedlab_engine.console import Console
from speedlab_engine.pad import packed, unpacked

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper, phases
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.speedrun.movie import memory_sample
from datenwissenschaften.speedrun.record import StoredRecord
from datenwissenschaften.workflow.wrapper import wrapper_class

WRAPPER = "speedrun.{slug}:Wrapper"

app = typer.Typer()


class Piece(msgspec.Struct, frozen=True):
    segment: int
    entered: str
    reached: str
    won: bool
    first_frame: int
    frames: int
    start: str
    end: str
    inputs: str


class Pieces(msgspec.Struct, frozen=True):
    game: str
    version: str
    cartridge_version: str
    record: str
    labels: list[str]
    levels: dict[str, str]
    pieces: list[Piece]


class Traced(msgspec.Struct, frozen=True):
    segment: int
    identical: bool
    labels: list[str]
    frames: list[list[int | None]]


def game_wrapper(slug: str) -> type[StateMachineGymWrapper[RamInfo]]:
    return wrapper_class(WRAPPER.format(slug=slug))


def text(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def level_names(wrapper: type[StateMachineGymWrapper[RamInfo]]) -> dict[str, str]:
    return {state.__name__: level for level, states in wrapper.levels for state in states}


def ram_class(wrapper: type[StateMachineGymWrapper[RamInfo]]) -> type[RamInfo]:
    return wrapper.ram_info_cls  # pyright: ignore[reportGeneralTypeIssues]


def labels_of(wrapper: type[StateMachineGymWrapper[RamInfo]]) -> list[str]:
    return [shown.label for shown in ram_class(wrapper).shown_fields().values()]


def packed_inputs(pressed: list[list[int]]) -> bytes:
    return bytes(packed(tuple(bool(button) for button in buttons)) for buttons in pressed)


def record_pieces(collection: Path, slug: str, record: Path) -> Pieces:
    content = record.read_bytes()
    stored = msgspec.msgpack.decode(content, type=StoredRecord)
    wrapper = game_wrapper(slug)
    console = Console(collection, cartridge_name(slug), False)
    route = [phases(wrapper)[0], *(segment.state for segment in stored.segments)]
    starts = [console.em.get_state(), *(segment.emulator_state for segment in stored.segments)]
    first_frames = np.cumsum([0, *(len(segment.pressed) for segment in stored.segments)]).tolist()
    pieces = [
        Piece(
            segment=index,
            entered=route[index],
            reached=segment.state,
            won=segment.won,
            first_frame=first_frames[index],
            frames=len(segment.pressed),
            start=text(starts[index]),
            end=text(segment.emulator_state),
            inputs=text(packed_inputs(segment.pressed)),
        )
        for index, segment in enumerate(stored.segments)
    ]
    return Pieces(
        game=slug,
        version=stored.version,
        cartridge_version=console.version,
        record=hashlib.sha256(content).hexdigest(),
        labels=labels_of(wrapper),
        levels=level_names(wrapper),
        pieces=pieces,
    )


def frame_row(frame: int, ram: RamInfo) -> list[int | None]:
    position = ram.position()
    located = [None, None, None] if position is None else [position.area, position.x, position.y]
    return [frame, *located, *memory_sample(ram).values()]


def traced(console: Console, wrapper: type[StateMachineGymWrapper[RamInfo]], piece: Piece) -> Traced:
    console.em.set_state(base64.b64decode(piece.start))
    rows: list[list[int | None]] = []
    for offset, pressed in enumerate(base64.b64decode(piece.inputs), 1):
        console.step(np.asarray(unpacked(pressed), dtype=np.int8))
        rows.append(frame_row(piece.first_frame + offset, ram_class(wrapper).from_ram(console.get_ram())))
    identical = console.em.get_state() == base64.b64decode(piece.end)
    return Traced(piece.segment, identical, labels_of(wrapper), rows)


def traces(collection: Path, slug: str, lines: Iterator[str]) -> Iterator[Traced]:
    console = Console(collection, cartridge_name(slug), False)
    wrapper = game_wrapper(slug)
    for line in lines:
        yield traced(console, wrapper, msgspec.json.decode(line, type=Piece))


def written(document: Any) -> None:
    sys.stdout.write(json.dumps(msgspec.to_builtins(document), separators=(",", ":")) + "\n")


@app.command()
def pieces(collection: Path, slug: str, record: Path) -> None:
    written(record_pieces(collection, slug, record))


@app.command()
def frames(collection: Path, slug: str) -> None:
    for trace in traces(collection, slug, iter(sys.stdin.readline, "")):
        written(trace)


if __name__ == "__main__":
    app()
