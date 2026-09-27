import json
from pathlib import Path

import pytest

from datenwissenschaften.persistence import JsonDatabase


def test_values_survive_a_reload(tmp_path: Path):
    path = tmp_path / "nested" / "database.json"
    JsonDatabase(path).set("engine-version:Game", "2.10.17")

    database = JsonDatabase(path)

    assert database.contains("engine-version:Game")
    assert database.get("engine-version:Game") == "2.10.17"
    assert json.loads(path.read_text(encoding="utf-8")) == {"engine-version:Game": "2.10.17"}


def test_missing_keys_fail_fast(tmp_path: Path):
    database = JsonDatabase(tmp_path / "database.json")

    assert not database.contains("missing")
    with pytest.raises(KeyError):
        database.get("missing")


def test_delete_removes_the_key_and_tolerates_absent_keys(tmp_path: Path):
    database = JsonDatabase(tmp_path / "database.json")
    database.set("history:Game", {"episodes": 1})

    database.delete("history:Game")
    database.delete("history:Game")

    assert not JsonDatabase(tmp_path / "database.json").contains("history:Game")
