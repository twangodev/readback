from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from readback.data import _decode, _present_tails, parse_shard_spec


def test_parse_shard_spec_expands_ranges_and_dedupes():
    assert parse_shard_spec("0-2, 5, 5, 1") == [0, 1, 2, 5]


def test_parse_shard_spec_ignores_blanks():
    assert parse_shard_spec(" , 3 , ") == [3]


def test_present_tails_drops_empty():
    assert _present_tails(["N1", "", None, "N2"]) == ("N1", "N2")
    assert _present_tails(None) == ()


def test_decode_collapses_stereo_to_mono():
    stereo = np.stack([np.ones(800, np.float32), np.zeros(800, np.float32)], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 16000, format="WAV", subtype="FLOAT")
    audio = _decode(buffer.getvalue())
    assert audio.array.ndim == 1
    assert audio.sample_rate == 16000
    assert np.allclose(audio.array, 0.5)
