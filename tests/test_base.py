from __future__ import annotations

import numpy as np

from readback.audio import bucket_by_duration, to_mono_float32
from readback.models.base import Audio, Hypothesis, Transcriber


def test_hypothesis_optional_fields_default_to_none():
    hypothesis = Hypothesis(text="cleared to land")
    assert hypothesis.confidence is None
    assert hypothesis.word_confidences is None
    assert hypothesis.no_speech is None


def test_to_mono_float32_collapses_channels():
    stereo = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mono = to_mono_float32(stereo)
    assert mono.shape == (2,)
    assert mono.dtype == np.float32


def test_bucket_by_duration_orders_then_chunks():
    items = [("a", 3.0), ("b", 1.0), ("c", 2.0)]
    batches = list(bucket_by_duration(items, key=lambda item: item[1], batch_size=2))
    assert [item[0] for batch in batches for item in batch] == ["b", "c", "a"]
    assert [len(batch) for batch in batches] == [2, 1]


def test_transcriber_protocol_is_runtime_checkable():
    class Echo:
        name = "echo"

        def transcribe(self, clips, bias_terms=None):
            return [Hypothesis(text="") for _ in clips]

    audio = Audio(array=np.zeros(8, dtype=np.float32), sample_rate=16000)
    echo = Echo()
    assert isinstance(echo, Transcriber)
    assert echo.transcribe([audio])[0].text == ""
