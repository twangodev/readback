from __future__ import annotations

from readback.server.context import build_context, diff_ops


def _ops(base: str, hyp: str) -> list[tuple[str, str]]:
    return [(cell["op"], cell["token"]) for cell in diff_ops(base, hyp)]


def test_identical_is_all_equal():
    assert _ops("cleared to land", "cleared to land") == [
        ("equal", "cleared"),
        ("equal", "to"),
        ("equal", "land"),
    ]


def test_substitution_marked_sub():
    assert _ops("cleared to land", "cleared to taxi") == [
        ("equal", "cleared"),
        ("equal", "to"),
        ("sub", "taxi"),
    ]


def test_insertion_marked_ins():
    assert _ops("cleared land", "cleared to land") == [
        ("equal", "cleared"),
        ("ins", "to"),
        ("equal", "land"),
    ]


def test_deletion_marked_del():
    assert _ops("cleared to land", "cleared land") == [
        ("equal", "cleared"),
        ("del", "to"),
        ("equal", "land"),
    ]


def _label() -> dict:
    return {
        "utterance_id": "kagc/1",
        "transcript": "cleared to land",
        "tier": "silver",
        "rover_confidence": 0.8,
        "agreement_score": 0.5,
        "n_models_agree": 2,
        "n_tails": 1,
        "callsign_matched": True,
        "callsign_tail": "N172SP",
        "callsign_score": 0.9,
        "snapped": False,
        "voting_text": ["cleared to land", "cleared to taxi"],
        "advisory_disagree": 0.2,
    }


def test_build_context_merges_label_and_meta():
    meta = {"airport": "kagc", "tails": ["N172SP"]}
    ctx = build_context(_label(), meta)
    assert ctx["utterance_id"] == "kagc/1"
    assert ctx["base_hyp"] == "cleared to land"
    assert ctx["airport"] == "kagc"
    assert ctx["tails"] == ["N172SP"]
    assert ctx["callsign_tail"] == "N172SP"


def test_build_context_aligns_each_hypothesis_to_base():
    ctx = build_context(_label(), {"airport": "kagc", "tails": []})
    aligned = ctx["aligned_hypotheses"]
    assert [a["text"] for a in aligned] == ["cleared to land", "cleared to taxi"]
    assert [(c["op"], c["token"]) for c in aligned[1]["ops"]] == [
        ("equal", "cleared"),
        ("equal", "to"),
        ("sub", "taxi"),
    ]
