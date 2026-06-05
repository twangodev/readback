from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from readback import store
from readback.dataset import SCHEMA, iter_dataset, write_dataset_shards
from readback.fuse.confidence import confidence_score
from readback.pipeline.layout import labels_path, reviews_path


def _label(utterance_id: str, tier: str = "silver", **over) -> dict:
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


def test_publishes_every_clip_unfiltered_with_score(tmp_path: Path):
    store.write_jsonl(
        labels_path(tmp_path, 0),
        [
            _label("a", "gold"),
            _label("b", "silver", agreement_score=0.4, rover_confidence=0.6),
            _label("c", "tail"),
            _label("d", "non_speech", transcript=""),
        ],
    )
    rows = {row["utterance_id"]: row for row in iter_dataset(tmp_path)}
    assert set(rows) == {"a", "b", "c", "d"}  # no tier filtering — one row per clip
    assert rows["b"]["review_status"] == "auto"
    assert rows["b"]["confidence"] == confidence_score(0.4, 0.6, None)
    assert rows["b"]["advisory_disagree"] is None
    assert "tier" not in rows["b"]  # tiers are a consumer choice, not published


def test_human_verdicts_override_confidence_and_transcript(tmp_path: Path):
    store.write_jsonl(
        labels_path(tmp_path, 0),
        [
            _label("a", transcript="model text"),
            _label("b"),
            _label("c"),
        ],
    )
    store.append_jsonl(reviews_path(tmp_path, 0), _set("a", "accept", "model text", 0))
    store.append_jsonl(reviews_path(tmp_path, 0), _set("b", "edit", "fixed text", 1))
    store.append_jsonl(reviews_path(tmp_path, 0), _set("c", "reject", "", 2))
    rows = {row["utterance_id"]: row for row in iter_dataset(tmp_path)}
    assert rows["a"]["review_status"] == "verified" and rows["a"]["confidence"] == 1.0
    assert rows["a"]["transcript"] == "model text"
    assert rows["b"]["review_status"] == "edited" and rows["b"]["confidence"] == 1.0
    assert rows["b"]["transcript"] == "fixed text"
    assert rows["c"]["review_status"] == "rejected" and rows["c"]["confidence"] == 0.0
    assert rows["c"]["transcript"] == ""


def test_human_non_speech_is_a_verified_empty_label(tmp_path: Path):
    store.write_jsonl(labels_path(tmp_path, 0), [_label("a")])
    store.append_jsonl(reviews_path(tmp_path, 0), _set("a", "non_speech", "", 0))
    row = next(iter(iter_dataset(tmp_path)))
    assert row["review_status"] == "non_speech"
    assert row["transcript"] == "" and row["confidence"] == 1.0


def test_uses_stored_confidence_when_label_carries_it(tmp_path: Path):
    store.write_jsonl(labels_path(tmp_path, 0), [_label("a", confidence=0.42)])
    rows = list(iter_dataset(tmp_path))
    assert rows[0]["confidence"] == 0.42


def test_parquet_shards_mirror_source_order_and_schema(tmp_path: Path):
    # two shards, each in a fixed (source) row order that must be preserved
    store.write_jsonl(labels_path(tmp_path, 0), [_label("z"), _label("a"), _label("m")])
    store.write_jsonl(labels_path(tmp_path, 1), [_label("b", transcript="model text")])
    store.append_jsonl(reviews_path(tmp_path, 1), _set("b", "edit", "fixed text", 0))

    shards, rows = write_dataset_shards(tmp_path, tmp_path / "out")
    assert (shards, rows) == (2, 4)

    t0 = pq.read_table(tmp_path / "out" / "shard-00000.parquet")
    assert t0.schema.equals(SCHEMA)  # type-uniform across shards for HF load
    assert t0.column("utterance_id").to_pylist() == ["z", "a", "m"]  # order preserved

    t1 = pq.read_table(tmp_path / "out" / "shard-00001.parquet").to_pylist()
    assert t1[0]["transcript"] == "fixed text" and t1[0]["confidence"] == 1.0
