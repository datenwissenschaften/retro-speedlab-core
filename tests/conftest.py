from importlib.util import find_spec

import pytest

from datenwissenschaften.experiments import API_KEY

collect_ignore = [] if find_spec("torch") else ["test_policy.py"]


@pytest.fixture(autouse=True)
def untracked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY, raising=False)
