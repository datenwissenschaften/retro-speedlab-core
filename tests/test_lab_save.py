from pathlib import Path

import msgspec
import pytest
import typer

from datenwissenschaften.lab.save import save

KNOWLEDGE = """mechanics:
  - title: Pace
    text: "Every clean landing raises pace up to 8, and hitting a wall resets it."
player: null
enemies: []
powerups: []
levels: []
ram: []
"""
UNQUOTED_COMMAS = KNOWLEDGE.replace(
    '  - title: Pace\n    text: "Every clean landing raises pace up to 8, and hitting a wall resets it."',
    "  - {title: Pace, text: Every clean landing raises pace up to 8, and hitting a wall resets it.}",
)


def test_valid_knowledge_is_saved(tmp_path: Path) -> None:
    draft, target = tmp_path / "draft.yaml", tmp_path / "game" / "knowledge.yaml"
    draft.write_text(KNOWLEDGE, encoding="utf-8")

    save(draft, target)

    assert target.read_text(encoding="utf-8") == KNOWLEDGE


def test_knowledge_the_search_cannot_load_is_refused_and_the_saved_one_stays(tmp_path: Path) -> None:
    draft, target = tmp_path / "draft.yaml", tmp_path / "knowledge.yaml"
    draft.write_text(UNQUOTED_COMMAS, encoding="utf-8")
    target.write_text(KNOWLEDGE, encoding="utf-8")

    with pytest.raises(RuntimeError, match="not valid game knowledge"):
        save(draft, target)

    assert target.read_text(encoding="utf-8") == KNOWLEDGE


def test_hints_the_search_cannot_load_are_refused(tmp_path: Path) -> None:
    draft = tmp_path / "draft.json"
    draft.write_text('[{"state": 3}]', encoding="utf-8")

    with pytest.raises(msgspec.ValidationError):
        save(draft, tmp_path / "hints.json")

    assert not (tmp_path / "hints.json").exists()


def test_only_hints_and_knowledge_are_saved(tmp_path: Path) -> None:
    draft = tmp_path / "draft.yaml"
    draft.write_text(KNOWLEDGE, encoding="utf-8")

    with pytest.raises(typer.BadParameter):
        save(draft, tmp_path / "config.yaml")
