from __future__ import annotations

import difflib
from collections.abc import Mapping

from readback.review import ReviewEvent, merge_label

_CONTEXT_KEYS = (
    "utterance_id",
    "transcript",
    "tier",
    "rover_confidence",
    "agreement_score",
    "n_models_agree",
    "n_tails",
    "callsign_matched",
    "callsign_tail",
    "callsign_score",
    "snapped",
    "advisory_disagree",
)


def diff_ops(base: str, hyp: str) -> list[dict]:
    ref, candidate = base.split(), hyp.split()
    cells: list[dict] = []
    matcher = difflib.SequenceMatcher(a=ref, b=candidate, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            cells.extend({"op": "equal", "token": token} for token in candidate[j1:j2])
        elif tag == "insert":
            cells.extend({"op": "ins", "token": token} for token in candidate[j1:j2])
        elif tag == "delete":
            cells.extend({"op": "del", "token": token} for token in ref[i1:i2])
        else:
            cells.extend({"op": "sub", "token": token} for token in candidate[j1:j2])
            unmatched = (i2 - i1) - (j2 - j1)
            if unmatched > 0:
                cells.extend(
                    {"op": "del", "token": token} for token in ref[i2 - unmatched : i2]
                )
    return cells


def build_context(
    label: Mapping,
    meta: Mapping,
    current: ReviewEvent | None = None,
    base: str | None = None,
) -> dict:
    merged = merge_label(dict(label), current)
    base_hyp = label["transcript"] if base is None else base
    voting = list(label["voting_text"])
    context = {key: label[key] for key in _CONTEXT_KEYS}
    context["base_hyp"] = base_hyp
    context["voting_text"] = voting
    context["aligned_hypotheses"] = [
        {"text": hyp, "ops": diff_ops(base_hyp, hyp)} for hyp in voting
    ]
    context["airport"] = meta["airport"]
    context["tails"] = list(meta.get("tails", []))
    context["reviewed"] = merged["reviewed"]
    context["review_decision"] = merged["review_decision"]
    context["effective_transcript"] = merged["effective_transcript"]
    context["source"] = merged["source"]
    return context
