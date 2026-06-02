from __future__ import annotations

from dataclasses import dataclass

import airwer

NULL = ""


@dataclass(frozen=True, slots=True)
class RoverResult:
    text: str
    confidence: float


def rover(hypotheses: list[str], weights: list[float] | None = None) -> RoverResult:
    sequences = [airwer.normalize(h).split() for h in hypotheses]
    if not sequences:
        return RoverResult(text="", confidence=1.0)
    if weights is None:
        weights = [1.0] * len(sequences)

    slots, total_weight = _build_vote_network(sequences, weights)
    chosen = []
    slot_scores = []
    for slot in slots:
        token, weight = max(slot.items(), key=lambda item: item[1])
        slot_scores.append(weight / total_weight)
        if token != NULL:
            chosen.append(token)
    confidence = sum(slot_scores) / len(slot_scores) if slot_scores else 1.0
    return RoverResult(text=" ".join(chosen), confidence=confidence)


def _build_vote_network(
    sequences: list[list[str]], weights: list[float]
) -> tuple[list[dict[str, float]], float]:
    slots = [{token: weights[0]} for token in sequences[0]]
    processed = weights[0]
    for sequence, weight in zip(sequences[1:], weights[1:]):
        reference = [_representative(slot) for slot in slots]
        merged: list[dict[str, float]] = []
        for slot_index, token_index in _align(reference, sequence):
            if slot_index is None:
                if token_index is not None:
                    merged.append({sequence[token_index]: weight, NULL: processed})
            elif token_index is None:
                slot = slots[slot_index]
                slot[NULL] = slot.get(NULL, 0.0) + weight
                merged.append(slot)
            else:
                slot = slots[slot_index]
                word = sequence[token_index]
                slot[word] = slot.get(word, 0.0) + weight
                merged.append(slot)
        slots = merged
        processed += weight
    return slots, processed


def _representative(slot: dict[str, float]) -> str | None:
    token = max(slot.items(), key=lambda item: item[1])[0]
    return token if token != NULL else None


def _align(
    reference: list[str | None], hypothesis: list[str]
) -> list[tuple[int | None, int | None]]:
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = (
                reference[i - 1] is not None and reference[i - 1] == hypothesis[j - 1]
            )
            cost[i][j] = min(
                cost[i - 1][j - 1] + (0 if same else 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        same = (
            i > 0
            and j > 0
            and reference[i - 1] is not None
            and reference[i - 1] == hypothesis[j - 1]
        )
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs
