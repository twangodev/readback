from __future__ import annotations

from readback.server.worklist import build_worklist, queue_stats


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
        "voting_text": ["a", "b", "c"],
        "advisory_disagree": None,
    }
    row.update(over)
    return row


def test_only_silver_and_tail_are_queued():
    labels = {
        0: [
            _label("a", "gold"),
            _label("b", "silver"),
            _label("c", "tail"),
            _label("d", "non_speech"),
        ]
    }
    queued = {item["utterance_id"] for item in build_worklist(labels)}
    assert queued == {"b", "c"}


def test_ordered_shard_major():
    labels = {
        1: [_label("s1", "silver")],
        0: [_label("s0", "silver")],
    }
    order = [item["utterance_id"] for item in build_worklist(labels)]
    assert order == ["s0", "s1"]


def test_highest_advisory_disagree_first_within_shard():
    labels = {
        0: [
            _label("low", "silver", advisory_disagree=0.1),
            _label("high", "silver", advisory_disagree=0.8),
            _label("none", "silver", advisory_disagree=None),
        ]
    }
    order = [item["utterance_id"] for item in build_worklist(labels)]
    assert order == ["high", "low", "none"]


def test_ties_broken_by_lowest_agreement_score():
    labels = {
        0: [
            _label("sure", "tail", advisory_disagree=0.5, agreement_score=0.6),
            _label("unsure", "tail", advisory_disagree=0.5, agreement_score=0.2),
        ]
    }
    order = [item["utterance_id"] for item in build_worklist(labels)]
    assert order == ["unsure", "sure"]


def test_queue_item_shape_and_reviewed_flag():
    labels = {7: [_label("x", "silver", advisory_disagree=0.3)]}
    item = build_worklist(labels, reviewed={"x"})[0]
    assert item == {
        "utterance_id": "x",
        "shard": 7,
        "tier": "silver",
        "rover_confidence": 0.9,
        "agreement_score": 0.5,
        "n_models_agree": 2,
        "callsign_matched": False,
        "advisory_disagree": 0.3,
        "reviewed": True,
    }


def test_queue_stats_counts_by_tier_and_reviewed():
    labels = {
        0: [_label("a", "silver"), _label("b", "tail"), _label("c", "gold")],
        1: [_label("d", "silver")],
    }
    stats = queue_stats(labels, reviewed={"a"})
    assert stats == {
        "total": 3,
        "reviewed": 1,
        "remaining": 2,
        "by_tier": {"silver": 2, "tail": 1},
    }
