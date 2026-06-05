from __future__ import annotations

from readback.server.worklist import (
    GOLD_VALIDATION_BUDGET,
    build_review_plan,
    build_worklist,
    queue_stats,
)


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


def test_gold_validation_leads_plan_and_caps_at_budget():
    labels = {
        0: [_label(f"g{i}", "gold") for i in range(GOLD_VALIDATION_BUDGET + 10)]
        + [_label("s", "silver", agreement_score=0.3)]
    }
    plan = build_review_plan(labels, budget=40)
    gold_items = [item for item in plan if item["reason"] == "gold-validation"]
    assert len(gold_items) == GOLD_VALIDATION_BUDGET
    assert all(item["tier"] == "gold" for item in gold_items)
    assert {item["reason"] for item in plan[:GOLD_VALIDATION_BUDGET]} == {
        "gold-validation"
    }


def test_gold_validation_excludes_reviewed_and_terminates():
    gold = [f"g{i}" for i in range(GOLD_VALIDATION_BUDGET)]
    labels = {0: [_label(g, "gold") for g in gold]}
    plan = build_review_plan(labels, budget=40, reviewed=set(gold))
    assert [item for item in plan if item["reason"] == "gold-validation"] == []


def test_gold_validation_set_is_stable_and_deterministic():
    labels = {0: [_label(f"g{i}", "gold") for i in range(GOLD_VALIDATION_BUDGET + 20)]}
    runs = [
        [
            item["utterance_id"]
            for item in build_review_plan(labels, budget=40)
            if item["reason"] == "gold-validation"
        ]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert len(set(runs[0])) == GOLD_VALIDATION_BUDGET


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
