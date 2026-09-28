from pathlib import Path

from datenwissenschaften.states.landmarks import Landmarks


def test_landmarks_keep_the_first_sighting_across_restarts(tmp_path: Path):
    path = tmp_path / "landmarks.json"
    Landmarks(path).remember("door", (-3, 260))
    landmarks = Landmarks(path)

    landmarks.remember("door", (90, 90))

    assert landmarks.recall("door") == (-3, 260)
    assert landmarks.recall("scale") is None


def test_forgotten_landmarks_are_gone_from_disk(tmp_path: Path):
    path = tmp_path / "landmarks.json"
    landmarks = Landmarks(path)
    landmarks.remember("door", (1, 2))

    landmarks.forget()

    assert landmarks.recall("door") is None
    assert Landmarks(path).recall("door") is None
