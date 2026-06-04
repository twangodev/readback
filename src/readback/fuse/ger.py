from __future__ import annotations

from dataclasses import dataclass

import airwer

from readback.callsign import snap
from readback.serve.client import GerClient

INSTRUCTION = (
    "You are correcting a single air traffic control radio transmission. "
    "Several speech recognizers produced the candidate transcripts below. "
    "They agree on most words and disagree on a few. "
    "Output one corrected transcript that keeps the agreed words and resolves "
    "the disagreements into fluent ATC phraseology. "
    "Write every number, callsign, and letter in spoken form, all lowercase, "
    "no punctuation. Output only the transcript on a single line."
)
CALLSIGN_DIRECTIVE = (
    "The only aircraft physically present are listed under CALLSIGNS, "
    "written in canonical spoken form. Any callsign you transcribe must be one "
    "of these exact spoken forms. Never invent a callsign that is not listed."
)


@dataclass(frozen=True, slots=True)
class GerResult:
    text: str
    raw: str
    corrected: bool
    callsign_enforced: bool


@dataclass(frozen=True, slots=True)
class LabelLike:
    utterance_id: str
    transcript: str
    voting_text: tuple[str, ...]
    callsign_tail: str | None
    callsign_matched: bool
    tails: tuple[str, ...] = ()


def canonical_callsigns(tails: list[str]) -> list[str]:
    spoken = []
    for tail in tails:
        if not tail:
            continue
        expanded = airwer.normalize(airwer.vocab.expand_callsign(tail) or "")
        if expanded and expanded not in spoken:
            spoken.append(expanded)
    return spoken


def build_prompt(
    hypotheses: list[str],
    fused_text: str,
    callsigns: list[str],
    *,
    instruction: str = INSTRUCTION,
) -> str:
    lines = [instruction]
    spoken = canonical_callsigns(callsigns)
    if spoken:
        lines.append(CALLSIGN_DIRECTIVE)
        lines.append("")
        lines.append("CALLSIGNS")
        lines.extend(f"- {callsign}" for callsign in spoken)
    lines.append("")
    lines.append("CANDIDATES")
    lines.extend(
        f"{index + 1}. {airwer.normalize(text)}"
        for index, text in enumerate(hypotheses)
    )
    lines.append("")
    lines.append("CONSENSUS")
    lines.append(airwer.normalize(fused_text))
    lines.append("")
    lines.append("CORRECTED")
    return "\n".join(lines)


def transcript_grammar(callsigns: list[str]) -> str | None:
    spoken = canonical_callsigns(callsigns)
    if len(spoken) != 1:
        return None
    words = spoken[0].split()
    if not words:
        return None
    callsign_literal = " ".join(words)
    return (
        'root ::= (words " ")? callsign (" " words)?\n'
        f'callsign ::= "{callsign_literal}"\n'
        'words ::= word (" " word)*\n'
        "word ::= [a-z]+\n"
    )


def correct(
    client: GerClient,
    label_like: LabelLike,
    *,
    instruction: str = INSTRUCTION,
) -> GerResult:
    tails = list(label_like.tails)
    prompt = build_prompt(
        list(label_like.voting_text),
        label_like.transcript,
        tails,
        instruction=instruction,
    )
    grammar = transcript_grammar(tails) if label_like.callsign_matched else None
    raw = client.complete(prompt, grammar=grammar)
    cleaned = airwer.normalize(raw)

    enforced = False
    text = cleaned
    if label_like.callsign_matched and label_like.callsign_tail:
        snapped = snap(cleaned, [label_like.callsign_tail])
        text = snapped.text
        enforced = snapped.match.matched
    if not text:
        text = airwer.normalize(label_like.transcript)
    return GerResult(
        text=text,
        raw=raw,
        corrected=text != airwer.normalize(label_like.transcript),
        callsign_enforced=enforced,
    )
