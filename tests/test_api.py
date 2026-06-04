from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from readback.server.api import create_app


class StubService:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, dict]] = []
        self.undone: list[str] = []

    def queue(self) -> dict:
        return {
            "items": [{"utterance_id": "kagc/1", "shard": 46}],
            "total": 1,
            "reviewed": 0,
        }

    def stats(self) -> dict:
        return {
            "total": 1,
            "reviewed": 0,
            "remaining": 1,
            "by_tier": {"silver": 1, "tail": 0},
        }

    def context(self, utterance_id: str) -> dict | None:
        if utterance_id != "kagc/1":
            return None
        return {"utterance_id": "kagc/1", "base_hyp": "cleared to land"}

    def audio(self, utterance_id: str) -> bytes:
        if utterance_id != "kagc/1":
            raise KeyError(utterance_id)
        return b"RIFF\x00\x00\x00\x00WAVEfmt "

    def record(self, utterance_id: str, payload: dict) -> dict:
        self.recorded.append((utterance_id, payload))
        return {"utterance_id": utterance_id, **payload}

    def undo(self, utterance_id: str) -> dict | None:
        self.undone.append(utterance_id)
        return {"utterance_id": utterance_id, "decision": "accept"}


@pytest.fixture
def client() -> tuple[TestClient, StubService]:
    service = StubService()
    return TestClient(create_app(service)), service


def test_queue_returns_payload(client):
    api, _ = client
    assert api.get("/api/queue").json()["items"][0]["utterance_id"] == "kagc/1"


def test_stats_returns_counts(client):
    api, _ = client
    assert api.get("/api/queue/stats").json()["remaining"] == 1


def test_clip_context_returns_payload(client):
    api, _ = client
    assert api.get("/api/clips/kagc/1").json()["base_hyp"] == "cleared to land"


def test_unknown_clip_is_404(client):
    api, _ = client
    assert api.get("/api/clips/nope").status_code == 404


def test_audio_returns_wav_with_immutable_cache_and_etag(client):
    api, _ = client
    resp = api.get("/api/clips/kagc/1/audio")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["etag"] == '"kagc/1"'
    assert "immutable" in resp.headers["cache-control"]
    assert resp.content.startswith(b"RIFF")


def test_audio_unknown_is_404(client):
    api, _ = client
    assert api.get("/api/clips/nope/audio").status_code == 404


def test_audio_304_on_matching_if_none_match(client):
    api, _ = client
    resp = api.get("/api/clips/kagc/1/audio", headers={"If-None-Match": '"kagc/1"'})
    assert resp.status_code == 304


def test_post_review_records_and_flags_reviewed(client):
    api, service = client
    body = {
        "decision": "correct",
        "transcript": "cleared to taxi",
        "base_hyp": "cleared to land",
    }
    resp = api.post("/api/review/kagc/1", json=body)
    assert resp.json()["reviewed"] is True
    assert service.recorded == [("kagc/1", body)]


def test_delete_review_undoes(client):
    api, service = client
    resp = api.delete("/api/review/kagc/1")
    assert resp.json()["restored"]["decision"] == "accept"
    assert service.undone == ["kagc/1"]
