from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from readback import store
from readback.review import iter_effective

DEFAULT_TIERS = ("gold", "silver")
_ACCEPTED = ("accept", "edit")


def iter_trainset(run: Path, tiers: tuple[str, ...] = DEFAULT_TIERS) -> Iterator[dict]:
    selected = set(tiers)
    for merged in iter_effective(run):
        if merged["reviewed"]:
            if merged["review_decision"] not in _ACCEPTED:
                continue
        elif merged["tier"] not in selected:
            continue
        if not merged["effective_transcript"]:
            continue
        yield {
            "utterance_id": merged["utterance_id"],
            "transcript": merged["effective_transcript"],
            "tier": merged["tier"],
            "source": merged["source"],
            "reviewed": merged["reviewed"],
            "agreement_score": merged["agreement_score"],
            "rover_confidence": merged["rover_confidence"],
        }


def write_trainset(run: Path, out: Path, tiers: tuple[str, ...] = DEFAULT_TIERS) -> int:
    return store.write_jsonl(out, iter_trainset(run, tiers))
