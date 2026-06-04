from __future__ import annotations

from pathlib import Path

from readback import store
from readback.pipeline.layout import labels_path, reviews_path, snapshots_dir
from readback.snapshot import build_snapshot, write_snapshot


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


def _setup(tmp_path: Path) -> Path:
    store.write_jsonl(
        labels_path(tmp_path, 0), [_label("a", "gold"), _label("b", "silver")]
    )
    return tmp_path


def test_digest_is_stable_across_timestamps(tmp_path: Path):
    _setup(tmp_path)
    first = build_snapshot(tmp_path, "v1", ts=1.0)
    second = build_snapshot(tmp_path, "v1", ts=2.0)
    assert first["digest"] == second["digest"]
    assert first["total"] == 2
    assert first["by_tier"] == {"gold": 1, "silver": 1}
    assert first["review_offset"] == 0


def test_digest_changes_after_human_edit(tmp_path: Path):
    _setup(tmp_path)
    baseline = build_snapshot(tmp_path, "v1", ts=1.0)
    store.append_jsonl(
        reviews_path(tmp_path, 0),
        {
            "utterance_id": "b",
            "action": "set",
            "seq": 0,
            "ts": 1.0,
            "shard": 0,
            "decision": "edit",
            "transcript": "new text",
            "base_hyp": "raw",
        },
    )
    edited = build_snapshot(tmp_path, "v2", ts=1.0)
    assert edited["digest"] != baseline["digest"]
    assert edited["review_offset"] == 1


def test_write_snapshot_persists_manifest(tmp_path: Path):
    _setup(tmp_path)
    path = write_snapshot(tmp_path, "v1", tiers=["gold", "silver"], ts=1.0)
    assert path == snapshots_dir(tmp_path) / "v1.json"
    manifest = next(iter(store.read_jsonl(path)))
    assert manifest["name"] == "v1"
    assert manifest["tiers"] == ["gold", "silver"]
