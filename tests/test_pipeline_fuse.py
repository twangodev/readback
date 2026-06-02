from __future__ import annotations

from readback import store
from readback.pipeline.fuse import fuse_shard, run_fuse
from readback.pipeline.layout import hyps_path, labels_path, meta_path

GOLD_TEXT = "november one seven two sierra papa descend"


def _seed_meta(run, index, rows):
    store.write_jsonl(
        meta_path(run, index),
        [
            {"utterance_id": utt, "airport": "k", "tails": list(tails)}
            for utt, tails in rows
        ],
    )


def _seed_hyps(run, index, model, rows):
    store.write_jsonl(
        hyps_path(run, model, index),
        [
            {"utterance_id": utt, "text": text, "confidence": None, "no_speech": 0.0}
            for utt, text in rows
        ],
    )


def test_fuse_shard_assigns_gold_and_tail(tmp_path):
    _seed_meta(tmp_path, 0, [("g", ("N172SP",)), ("t", ())])
    _seed_hyps(tmp_path, 0, "a", [("g", GOLD_TEXT), ("t", "alpha")])
    _seed_hyps(tmp_path, 0, "b", [("g", GOLD_TEXT), ("t", "bravo charlie")])
    _seed_hyps(
        tmp_path, 0, "c", [("g", GOLD_TEXT), ("t", "delta echo foxtrot golf hotel")]
    )

    assert fuse_shard(tmp_path, ["a", "b", "c"], None, None, 0) == (0, 2, 0)
    labels = store.index_by(labels_path(tmp_path, 0), "utterance_id")
    assert labels["g"]["tier"] == "gold"
    assert labels["g"]["callsign_matched"] is True
    assert labels["t"]["tier"] == "tail"


def test_fuse_shard_skips_utterance_missing_a_voter(tmp_path):
    _seed_meta(tmp_path, 0, [("u0", ()), ("u1", ())])
    _seed_hyps(tmp_path, 0, "a", [("u0", "hello world"), ("u1", "foo bar")])
    _seed_hyps(tmp_path, 0, "b", [("u0", "hello world"), ("u1", "foo bar")])
    _seed_hyps(tmp_path, 0, "c", [("u0", "hello world")])

    assert fuse_shard(tmp_path, ["a", "b", "c"], None, None, 0) == (0, 1, 1)
    assert [r["utterance_id"] for r in store.read_jsonl(labels_path(tmp_path, 0))] == [
        "u0"
    ]


def test_fuse_shard_records_advisory_disagreement(tmp_path):
    _seed_meta(tmp_path, 0, [("u0", ())])
    for model in ("a", "b", "c"):
        _seed_hyps(tmp_path, 0, model, [("u0", "hello world")])
    _seed_hyps(tmp_path, 0, "d", [("u0", "completely different transcript")])

    fuse_shard(tmp_path, ["a", "b", "c"], None, "d", 0)
    label = next(store.read_jsonl(labels_path(tmp_path, 0)))
    assert label["advisory_disagree"] > 0


def test_run_fuse_skips_already_fused_shards(tmp_path):
    _seed_meta(tmp_path, 0, [("u0", ())])
    for model in ("a", "b", "c"):
        _seed_hyps(tmp_path, 0, model, [("u0", "hello world")])

    run_fuse(tmp_path, ["a", "b", "c"], [0], log=lambda _: None)
    logs: list[str] = []
    run_fuse(tmp_path, ["a", "b", "c"], [0], log=logs.append)
    assert any("already fused" in line for line in logs)
