import json

from datenwissenschaften.states.facts import Offset, render_facts


def test_offsets_read_as_directions_and_everything_else_stays_json():
    text = render_facts({"lives": 3, "door": Offset(-34, 12), "bell": Offset(0, -5)})

    assert json.loads(text) == {"lives": 3, "door": "34 left, 12 below", "bell": "level, 5 above"}
