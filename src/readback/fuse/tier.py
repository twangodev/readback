from __future__ import annotations

from enum import StrEnum

SILVER_MIN_AGREEMENT = 0.3


class Tier(StrEnum):
    NON_SPEECH = "non_speech"
    GOLD = "gold"
    SILVER = "silver"
    TAIL = "tail"


def classify(
    *,
    non_speech: bool,
    n_voters: int,
    n_agree: int,
    agreement_score: float,
    callsign_ok: bool,
) -> Tier:
    if non_speech:
        return Tier.NON_SPEECH
    if n_agree == n_voters and callsign_ok:
        return Tier.GOLD
    if agreement_score >= SILVER_MIN_AGREEMENT:
        return Tier.SILVER
    return Tier.TAIL
