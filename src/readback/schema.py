from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Label:
    utterance_id: str
    transcript: str
    tier: str
    confidence: float
    rover_confidence: float
    agreement_score: float
    n_models_agree: int
    n_tails: int
    callsign_matched: bool
    callsign_tail: str | None
    callsign_score: float
    snapped: bool
    voting_text: tuple[str, ...]
    advisory_disagree: float | None
