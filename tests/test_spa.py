from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from readback.server.api import create_app


class StubService:
    def queue(self) -> dict:
        return {"items": [], "total": 0, "reviewed": 0}

    def stats(self) -> dict:
        return {}

    def context(self, utterance_id: str) -> dict | None:
        return None

    def audio(self, utterance_id: str) -> bytes:
        raise KeyError(utterance_id)

    def record(self, utterance_id: str, payload: dict) -> dict:
        return {}

    def undo(self, utterance_id: str) -> dict | None:
        return None


@pytest.fixture
def web_dir(tmp_path: Path) -> Path:
    (tmp_path / "200.html").write_text("<!doctype html><div id=app></div>")
    (tmp_path / "app.css").write_text("body{margin:0}")
    return tmp_path


def _client(web_dir: Path) -> TestClient:
    return TestClient(create_app(StubService(), web_dir))


def test_serves_existing_asset(web_dir):
    resp = _client(web_dir).get("/app.css")
    assert resp.status_code == 200
    assert resp.text == "body{margin:0}"


def test_deep_link_falls_back_to_200_html(web_dir):
    resp = _client(web_dir).get("/review")
    assert resp.status_code == 200
    assert "id=app" in resp.text


def test_root_serves_spa_shell(web_dir):
    resp = _client(web_dir).get("/")
    assert resp.status_code == 200
    assert "id=app" in resp.text


def test_api_not_shadowed_by_spa_mount(web_dir):
    resp = _client(web_dir).get("/api/queue")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0, "reviewed": 0}
