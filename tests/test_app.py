from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from readback import store
from readback.pipeline.layout import labels_path, meta_path
from readback.server.app import build_app


class FakeAudio:
    def __init__(self, data: dict[tuple[int, str], bytes]) -> None:
        self._data = data

    def wav_bytes(self, shard: int, utterance_id: str) -> bytes:
        if (shard, utterance_id) not in self._data:
            raise KeyError(utterance_id)
        return self._data[(shard, utterance_id)]

    def context(self, shard: int, utterance_id: str) -> dict | None:
        return None

    def clip_start(self, shard: int, utterance_id: str) -> float | None:
        return None


def _label(utterance_id: str, tier: str) -> dict:
    return {
        "utterance_id": utterance_id,
        "transcript": "cleared to land",
        "tier": tier,
        "rover_confidence": 0.9,
        "agreement_score": 0.5,
        "n_models_agree": 2,
        "n_tails": 0,
        "callsign_matched": False,
        "callsign_tail": None,
        "callsign_score": 0.0,
        "snapped": False,
        "voting_text": ["cleared to land"],
        "advisory_disagree": None,
    }


def _setup_run(tmp_path: Path) -> Path:
    store.write_jsonl(
        labels_path(tmp_path, 0),
        [_label("kagc/1", "silver"), _label("kagc/2", "gold")],
    )
    store.write_jsonl(
        meta_path(tmp_path, 0),
        [
            {"utterance_id": "kagc/1", "airport": "kagc", "tails": []},
            {"utterance_id": "kagc/2", "airport": "kagc", "tails": []},
        ],
    )
    return tmp_path


def test_queue_endpoint_reflects_run(tmp_path):
    api = TestClient(build_app(_setup_run(tmp_path), FakeAudio({})))
    items = api.get("/api/queue").json()["items"]
    assert [item["utterance_id"] for item in items] == ["kagc/1"]


def test_audio_endpoint_streams_bytes(tmp_path):
    audio = FakeAudio({(0, "kagc/1"): b"RIFFwav"})
    api = TestClient(build_app(_setup_run(tmp_path), audio))
    resp = api.get("/api/clips/kagc/1/audio")
    assert resp.status_code == 200
    assert resp.content == b"RIFFwav"


def test_review_roundtrip_persists(tmp_path):
    api = TestClient(build_app(_setup_run(tmp_path), FakeAudio({})))
    api.post(
        "/api/review/kagc/1",
        json={"decision": "accept", "transcript": "x", "base_hyp": "y"},
    )
    assert api.get("/api/queue").json()["reviewed"] == 1
