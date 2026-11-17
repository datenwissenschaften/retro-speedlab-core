import pytest

from datenwissenschaften.experiments import API_KEY


@pytest.fixture(autouse=True)
def untracked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY, raising=False)
