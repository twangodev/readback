from __future__ import annotations

from pathlib import Path

from readback import store
from readback.pipeline.layout import labels_path, reviews_path
from readback.trainset import iter_trainset


def _label(utterance_id: str, tier: str, **over) -> dict:
    row = {
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
    row.update(over)
    return row


def _set(utterance_id: str, decision: str, transcript: str, seq: int = 0) -> dict:
    return {
        "utterance_id": utterance_id,
        "action": "set",
        "seq": seq,
        "ts": 1.0,
        "shard": 0,
        "decision": decision,
        "transcript": transcript,
        "base_hyp": "raw",
    }


def test_includes_gold_and_silver_excludes_tail_and_non_speech(tmp_path: Path):
    store.write_jsonl(
        labels_path(tmp_path, 0),
        [
            _label("a", "gold"),
            _label("b", "silver"),
            _label("c", "tail"),
            _label("d", "non_speech"),
        ],
    )
    ids = {row["utterance_id"] for row in iter_trainset(tmp_path)}
    assert ids == {"a", "b"}


def test_human_edit_overrides_transcript_and_promotes_from_tail(tmp_path: Path):
    store.write_jsonl(labels_path(tmp_path, 0), [_label("c", "tail")])
    store.append_jsonl(reviews_path(tmp_path, 0), _set("c", "edit", "fixed text"))
    rows = list(iter_trainset(tmp_path))
    assert len(rows) == 1
    assert rows[0]["transcript"] == "fixed text"
    assert rows[0]["source"] == "human"


def test_rejected_excluded_even_if_silver(tmp_path: Path):
    store.write_jsonl(labels_path(tmp_path, 0), [_label("b", "silver")])
    store.append_jsonl(reviews_path(tmp_path, 0), _set("b", "reject", ""))
    assert list(iter_trainset(tmp_path)) == []


def test_accept_keeps_model_transcript(tmp_path: Path):
    store.write_jsonl(
        labels_path(tmp_path, 0), [_label("a", "gold", transcript="model text")]
    )
    store.append_jsonl(reviews_path(tmp_path, 0), _set("a", "accept", "model text"))
    rows = list(iter_trainset(tmp_path))
    assert rows[0]["transcript"] == "model text"
    assert rows[0]["source"] == "human-confirmed"


def test_reviewed_and_source_distinguish_human_rows(tmp_path: Path):
    store.write_jsonl(
        labels_path(tmp_path, 0), [_label("a", "gold"), _label("c", "tail")]
    )
    store.append_jsonl(reviews_path(tmp_path, 0), _set("c", "edit", "fixed text"))
    rows = {row["utterance_id"]: row for row in iter_trainset(tmp_path)}
    assert rows["c"]["reviewed"] is True and rows["c"]["source"] == "human"
    assert rows["a"]["reviewed"] is False and rows["a"]["source"] == "model"
