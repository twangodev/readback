from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import airwer

AGREEMENT_TOLERANCE = 0.2


@dataclass(frozen=True, slots=True)
class AgreementResult:
    score: float
    n_agree: int


def agreement(hypotheses: list[str]) -> AgreementResult:
    n = len(hypotheses)
    if n <= 1:
        return AgreementResult(score=1.0, n_agree=n)

    similarities = []
    edges: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in combinations(range(n), 2):
        similarity = airwer.agreement(hypotheses[i], hypotheses[j])
        similarities.append(similarity)
        if (1.0 - similarity) <= AGREEMENT_TOLERANCE:
            edges[i].add(j)
            edges[j].add(i)

    largest = 1
    for size in range(n, 1, -1):
        if any(
            all(b in edges[a] for a, b in combinations(group, 2))
            for group in combinations(range(n), size)
        ):
            largest = size
            break
    return AgreementResult(score=sum(similarities) / len(similarities), n_agree=largest)
