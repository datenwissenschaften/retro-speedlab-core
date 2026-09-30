from pathlib import Path

from datenwissenschaften.states.landmarks import Landmarks, without_loops


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


def test_a_route_drops_its_loops_and_only_a_shorter_route_replaces_it(tmp_path: Path):
    landmarks = Landmarks(tmp_path / "landmarks.json")
    landmarks.remember("scale", (200, 0))

    landmarks.remember_route("scale", [(0, 0), (50, 0), (50, 50), (52, 2), (100, 0), (150, 0)])
    landmarks.remember_route("scale", [(0, 0), (40, 40), (80, 80), (120, 40), (160, 0), (180, 0)])

    assert Landmarks(tmp_path / "landmarks.json")._route("scale") == [(0, 0), (52, 2), (100, 0), (150, 0)]


def test_the_waypoint_follows_the_route_and_ends_at_the_landmark(tmp_path: Path):
    landmarks = Landmarks(tmp_path / "landmarks.json")
    landmarks.remember("scale", (100, 100))
    straight = landmarks.waypoint("scale", (0, 0))
    landmarks.remember_route("scale", [(0, 0), (0, 30), (0, 60), (0, 100), (60, 100)])

    assert straight == (100, 100)
    assert landmarks.waypoint("scale", (0, 0)) == (0, 60)
    assert landmarks.waypoint("scale", (0, 95)) == (60, 100)
    assert landmarks.waypoint("scale", (80, 100)) == (100, 100)
    assert landmarks.waypoint("door", (0, 0)) is None


def test_small_steps_along_a_route_are_no_loop(tmp_path: Path):
    assert without_loops([(0, 0), (8, 0), (16, 0), (24, 6), (8, 10)]) == [(0, 0), (8, 10)]
    assert without_loops([(0, 0), (8, 0), (16, 0), (24, 0)]) == [(0, 0), (8, 0), (16, 0), (24, 0)]


def test_the_safety_map_counts_danger_per_cell_and_survives_restarts(tmp_path: Path):
    path = tmp_path / "landmarks.json"
    landmarks = Landmarks(path)
    landmarks.safety.mark("danger", (33, 40))
    landmarks.safety.mark("danger", (47, 32))
    landmarks.safety.mark("blocked", (-1, 0))

    restarted = Landmarks(path).safety

    assert restarted.count("danger", (40, 45)) == 2
    assert restarted.count("danger", (48, 45)) == 0
    assert restarted.count("blocked", (-16, 15)) == 1


def test_forgetting_clears_the_safety_map(tmp_path: Path):
    landmarks = Landmarks(tmp_path / "landmarks.json")
    landmarks.safety.mark("danger", (0, 0))

    landmarks.forget()

    assert landmarks.safety.count("danger", (0, 0)) == 0
