import dataclasses
import functools
import json
from pathlib import Path

import httpx
from fakes import write_config

from datenwissenschaften.lab.reports import REPORTS_DIR, publish, publish_reports
from datenwissenschaften.settings import load_config

GAME = "Game-Nes-v0"


def backend(published: list[str], refused: set[str], puts: list[tuple[str, dict]]) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[{"game": GAME, "name": name} for name in published])
        name = request.url.path.rsplit("/", 1)[-1]
        if name in refused:
            return httpx.Response(500)
        puts.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200)

    return httpx.Client(base_url="https://api.test", transport=httpx.MockTransport(answer))


def test_only_reports_the_website_does_not_have_yet_are_published(tmp_path: Path):
    reports = tmp_path / REPORTS_DIR
    reports.mkdir(parents=True)
    for name in ("2026-10-29T1910.md", "2026-10-29T2320.md", "2026-10-29T2350.md"):
        (reports / name).write_text(f"# {name}", encoding="utf-8")
    puts: list[tuple[str, dict]] = []

    publish(backend(["2026-10-29T1910.md"], {"2026-10-29T2320.md"}, puts), GAME, "Game", tmp_path)

    assert puts == [(f"/reports/{GAME}/2026-10-29T2350.md", {"title": "Game", "text": "# 2026-10-29T2350.md"})]


def test_without_the_website_relay_nothing_is_published(tmp_path: Path):
    publish_reports(load_config(write_config(tmp_path)), tmp_path / "missing")


def test_lab_reports_wait_while_the_website_is_down(tmp_path: Path, monkeypatch):
    settings = load_config(write_config(tmp_path))
    live = dataclasses.replace(
        settings,
        ui=dataclasses.replace(settings.ui, enabled=True),
        upload=dataclasses.replace(settings.upload, api_key="key"),
    )
    down = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("down")))
    monkeypatch.setattr(httpx, "Client", functools.partial(httpx.Client, transport=down))

    publish_reports(live, tmp_path)
