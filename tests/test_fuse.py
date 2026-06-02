from __future__ import annotations

from readback.fuse import Tier, fuse_clip
from readback.models.base import Hypothesis


def _h(text: str, no_speech: float | None = None) -> Hypothesis:
    return Hypothesis(text=text, no_speech=no_speech)


def test_clean_unanimous_no_tails_is_gold():
    label = fuse_clip(
        "u1",
        [_h("cleared to land"), _h("cleared to land"), _h("cleared to land")],
        [],
    )
    assert label.tier == Tier.GOLD.value
    assert label.transcript == "cleared to land"
    assert label.rover_confidence == 1.0


def test_blank_majority_is_non_speech_with_empty_transcript():
    label = fuse_clip("u2", [_h(""), _h(""), _h("uh")], [])
    assert label.tier == Tier.NON_SPEECH.value
    assert label.transcript == ""


def test_high_no_speech_is_non_speech():
    label = fuse_clip(
        "u3",
        [_h("thank you", no_speech=0.9), _h("thanks"), _h("thank you")],
        [],
    )
    assert label.tier == Tier.NON_SPEECH.value


def test_partial_agreement_is_silver():
    label = fuse_clip(
        "u4",
        [
            _h("turn left heading two one zero"),
            _h("turn left heading two one five"),
            _h("turn right heading two one zero"),
        ],
        [],
    )
    assert label.tier == Tier.SILVER.value
    assert label.transcript == "turn left heading two one zero"


def test_present_callsign_not_spoken_blocks_gold():
    label = fuse_clip(
        "u5",
        [_h("cleared to land"), _h("cleared to land"), _h("cleared to land")],
        ["N172SP"],
    )
    assert label.n_tails == 1
    assert not label.callsign_matched
    assert label.tier == Tier.SILVER.value


def test_advisory_disagreement_recorded():
    label = fuse_clip(
        "u6",
        [_h("cleared to land"), _h("cleared to land"), _h("cleared to land")],
        [],
        advisory=[_h("cleared for takeoff")],
    )
    assert label.advisory_disagree is not None
    assert label.advisory_disagree > 0.0
