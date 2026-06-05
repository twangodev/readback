from __future__ import annotations


def confidence_score(
    agreement_score: float,
    rover_confidence: float,
    advisory_disagree: float | None,
) -> float:
    signals = [agreement_score, rover_confidence]
    if advisory_disagree is not None:
        signals.append(1.0 - advisory_disagree)
    return sum(signals) / len(signals)
