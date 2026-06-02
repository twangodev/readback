from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np


def to_mono_float32(samples: np.ndarray) -> np.ndarray:
    collapsed = samples.mean(axis=0) if samples.ndim == 2 else samples
    return collapsed.astype(np.float32, copy=False)


def to_target_sr(audio: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    array = audio.astype(np.float32, copy=False)
    if sample_rate == target_sr:
        return array
    import librosa

    return librosa.resample(array, orig_sr=sample_rate, target_sr=target_sr)


def bucket_by_duration[T](
    items: Sequence[T], key: Callable[[T], float], batch_size: int
) -> Iterator[list[T]]:
    ordered = sorted(items, key=key)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]
