from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Audio:
    array: np.ndarray
    sample_rate: int


@dataclass(frozen=True, slots=True)
class Hypothesis:
    text: str
    confidence: float | None = None
    word_confidences: tuple[float, ...] | None = None
    no_speech: float | None = None


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]: ...
