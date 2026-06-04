from __future__ import annotations

from pathlib import Path

import pytest

from readback import store
from readback.pipeline.layout import labels_path, meta_path
from readback.server.state import StudioState


def _label(utterance_id: str, tier: str, **over) -> dict:
    row = {
        "utterance_id": utterance_id,
        "transcript": "cleared to land",
        "tier": tier,
        "rover_confidence": 0.9,
        "agreement_score": 0.5,
        "n_models_agree": 2,
        "n_tails": 1,
        "callsign_matched": False,
        "callsign_tail": None,
        "callsign_score": 0.0,
        "snapped": False,
        "voting_text": ["cleared to land", "cleared to taxi"],
        "advisory_disagree": None,
    }
    row.update(over)
    return row


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


@pytest.fixture
def run(tmp_path: Path) -> Path:
    store.write_jsonl(
        labels_path(tmp_path, 0),
        [
            _label("kagc/1", "silver"),
            _label("kagc/2", "gold"),
            _label("kagc/3", "tail", advisory_disagree=0.7),
        ],
    )
    store.write_jsonl(
        meta_path(tmp_path, 0),
        [
            {"utterance_id": "kagc/1", "airport": "kagc", "tails": ["N1"]},
            {"utterance_id": "kagc/2", "airport": "kagc", "tails": []},
            {"utterance_id": "kagc/3", "airport": "kagc", "tails": ["N3"]},
        ],
    )
    return tmp_path


@pytest.fixture
def empty_audio() -> FakeAudio:
    return FakeAudio({})


def test_queue_lists_only_needs_review(run, empty_audio):
    queue = StudioState(run, empty_audio).queue()
    assert {item["utterance_id"] for item in queue["items"]} == {"kagc/1", "kagc/3"}
    assert queue["total"] == 2
    assert queue["reviewed"] == 0


def test_context_merges_meta(run, empty_audio):
    ctx = StudioState(run, empty_audio).context("kagc/1")
    assert ctx["airport"] == "kagc"
    assert ctx["tails"] == ["N1"]
    assert ctx["base_hyp"] == ctx["transcript"]


def test_context_for_known_clip_and_none_for_unknown(run, empty_audio):
    state = StudioState(run, empty_audio)
    assert state.context("nope") is None
    assert state.context("kagc/2") is not None


def test_context_reflects_effective_edit(run, empty_audio):
    state = StudioState(run, empty_audio)
    state.record(
        "kagc/1", {"decision": "edit", "transcript": "edited text", "base_hyp": "y"}
    )
    ctx = StudioState(run, empty_audio).context("kagc/1")
    assert ctx["reviewed"] is True
    assert ctx["effective_transcript"] == "edited text"
    assert ctx["source"] == "human"


def test_context_includes_conversation_neighbors(run, empty_audio):
    ctx = StudioState(run, empty_audio).context("kagc/1")
    convo = ctx["conversation"]
    assert [c["utterance_id"] for c in convo] == ["kagc/1", "kagc/2", "kagc/3"]
    assert [c["current"] for c in convo] == [True, False, False]


def test_plan_includes_calibration_and_boundary(run, empty_audio):
    plan = StudioState(run, empty_audio).plan(budget=10)
    reasons = {item["reason"] for item in plan["items"]}
    assert reasons <= {"calibration", "boundary"}
    assert plan["total"] >= 1


def test_audio_delegates_with_correct_shard(run):
    state = StudioState(run, FakeAudio({(0, "kagc/1"): b"RIFFwav"}))
    assert state.audio("kagc/1") == b"RIFFwav"
    with pytest.raises(KeyError):
        state.audio("nope")


def test_record_persists_and_marks_reviewed(run, empty_audio):
    row = StudioState(run, empty_audio).record(
        "kagc/1", {"decision": "correct", "transcript": "x", "base_hyp": "y"}
    )
    assert row["utterance_id"] == "kagc/1"
    assert row["shard"] == 0
    assert (run / "reviews" / "shard-00000.jsonl").exists()
    assert StudioState(run, empty_audio).queue()["reviewed"] == 1


def test_undo_reverts_and_rewrites(run, empty_audio):
    state = StudioState(run, empty_audio)
    state.record("kagc/1", {"decision": "accept", "transcript": "x", "base_hyp": "y"})
    assert state.undo("kagc/1") is None
    assert state.queue()["reviewed"] == 0
