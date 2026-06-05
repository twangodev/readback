from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from readback import store
from readback.fuse.confidence import confidence_score
from readback.review import iter_effective

REVIEW_STATUS = {
    None: "auto",
    "accept": "verified",
    "edit": "edited",
    "reject": "rejected",
    "non_speech": "non_speech",
}


def _confidence(merged: dict) -> float:
    decision = merged["review_decision"]
    if decision in ("accept", "edit", "non_speech"):
        return 1.0
    if decision == "reject":
        return 0.0
    value = merged.get("confidence")
    if value is None:
        value = confidence_score(
            merged["agreement_score"],
            merged["rover_confidence"],
            merged["advisory_disagree"],
        )
    return value


def iter_dataset(run: Path) -> Iterator[dict]:
    """The published label dataset: one transcript and one confidence score per
    utterance, plus the raw agreement signals so consumers can set their own
    threshold. Human review verdicts override confidence (verified/edited -> 1.0,
    rejected -> 0.0). No tiers: a tier is whatever cutoff a consumer picks."""
    for merged in iter_effective(run):
        yield {
            "utterance_id": merged["utterance_id"],
            "transcript": merged["effective_transcript"],
            "confidence": _confidence(merged),
            "review_status": REVIEW_STATUS[merged["review_decision"]],
            "agreement_score": merged["agreement_score"],
            "rover_confidence": merged["rover_confidence"],
            "advisory_disagree": merged["advisory_disagree"],
            "n_models_agree": merged["n_models_agree"],
            "callsign_matched": merged["callsign_matched"],
            "callsign_tail": merged["callsign_tail"],
        }


def write_dataset(run: Path, out: Path) -> int:
    return store.write_jsonl(out, iter_dataset(run))
