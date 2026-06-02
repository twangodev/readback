from __future__ import annotations

import re
from dataclasses import dataclass

import airwer
from rapidfuzz import fuzz

MATCH_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class CallsignMatch:
    matched: bool
    tail: str | None
    score: float
    n_tails: int


def verify(transcript: str, tails: list[str]) -> CallsignMatch:
    present = [tail for tail in tails if tail]
    spoken_transcript = airwer.normalize(transcript)
    best_score = 0.0
    best_tail = None
    for tail in present:
        for realization in _spoken_realizations(tail):
            score = fuzz.partial_ratio(realization, spoken_transcript) / 100.0
            if score > best_score:
                best_score, best_tail = score, tail
    return CallsignMatch(
        matched=best_score >= MATCH_THRESHOLD,
        tail=best_tail,
        score=best_score,
        n_tails=len(present),
    )


def _spoken_realizations(tail: str) -> list[str]:
    realizations = []
    full = airwer.vocab.expand_callsign(tail)
    if full:
        realizations.append(airwer.normalize(full))
    suffix = re.search(r"\d.*$", tail)
    if suffix:
        anchor = airwer.vocab.expand_callsign(suffix.group(0))
        if anchor:
            realizations.append(airwer.normalize(anchor))
    return [r for r in realizations if r]
