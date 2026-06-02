from __future__ import annotations

from readback.fuse import Tier, classify


def test_non_speech_takes_priority():
    assert (
        classify(
            non_speech=True,
            n_voters=3,
            n_agree=3,
            agreement_score=1.0,
            callsign_ok=True,
        )
        is Tier.NON_SPEECH
    )


def test_gold_requires_unanimous_and_callsign_ok():
    assert (
        classify(
            non_speech=False,
            n_voters=3,
            n_agree=3,
            agreement_score=1.0,
            callsign_ok=True,
        )
        is Tier.GOLD
    )


def test_unanimous_but_callsign_fails_is_silver():
    assert (
        classify(
            non_speech=False,
            n_voters=3,
            n_agree=3,
            agreement_score=1.0,
            callsign_ok=False,
        )
        is Tier.SILVER
    )


def test_partial_agreement_is_silver():
    assert (
        classify(
            non_speech=False,
            n_voters=3,
            n_agree=2,
            agreement_score=0.55,
            callsign_ok=True,
        )
        is Tier.SILVER
    )


def test_low_agreement_is_tail():
    assert (
        classify(
            non_speech=False,
            n_voters=3,
            n_agree=1,
            agreement_score=0.1,
            callsign_ok=True,
        )
        is Tier.TAIL
    )
