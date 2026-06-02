from __future__ import annotations

import airwer

from readback.callsign import snap
from readback.fuse.agreement import agreement
from readback.fuse.rover import rover
from readback.fuse.tier import Tier, classify
from readback.models.base import Hypothesis
from readback.schema import Label

NO_SPEECH_THRESHOLD = 0.6


def fuse_clip(
    utterance_id: str,
    voting: list[Hypothesis],
    tails: list[str],
    *,
    weights: list[float] | None = None,
    advisory: list[Hypothesis] | None = None,
) -> Label:
    voting_text = tuple(h.text for h in voting)
    normalized = [airwer.normalize(text) for text in voting_text]
    blanks = sum(1 for text in normalized if not text)
    acoustic = [h.no_speech for h in voting if h.no_speech is not None]
    non_speech = blanks * 2 >= len(voting) or any(
        p > NO_SPEECH_THRESHOLD for p in acoustic
    )

    agree = agreement(list(voting_text))
    fused = rover(list(voting_text), weights=weights)
    present_tails = [t for t in tails if t]
    snapped = snap(fused.text, present_tails)
    callsign_ok = snapped.match.matched or not present_tails
    tier = classify(
        non_speech=non_speech,
        n_voters=len(voting),
        n_agree=agree.n_agree,
        agreement_score=agree.score,
        callsign_ok=callsign_ok,
    )

    advisory_disagree = None
    if advisory:
        advisory_disagree = 1.0 - airwer.agreement(fused.text, advisory[0].text)

    return Label(
        utterance_id=utterance_id,
        transcript="" if tier is Tier.NON_SPEECH else snapped.text,
        tier=tier.value,
        rover_confidence=fused.confidence,
        agreement_score=agree.score,
        n_models_agree=agree.n_agree,
        n_tails=len(present_tails),
        callsign_matched=snapped.match.matched,
        callsign_tail=snapped.match.tail,
        callsign_score=snapped.match.score,
        snapped=snapped.snapped,
        voting_text=voting_text,
        advisory_disagree=advisory_disagree,
    )
