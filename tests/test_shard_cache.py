from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from readback.server.shard_cache import ShardAudio


def _wav(seed: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(32, seed, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def _write_shard(path: Path, clips: dict[str, bytes]) -> None:
    uids = list(clips)
    audio = [{"bytes": clips[u], "path": f"{u}.wav"} for u in uids]
    table = pa.table({"utterance_id": pa.array(uids), "audio": pa.array(audio)})
    pq.write_table(table, path)


@pytest.fixture
def shards(tmp_path: Path):
    paths: dict[int, Path] = {}
    payloads: dict[int, dict[str, bytes]] = {}
    for shard in (0, 1, 2):
        clips = {f"u{shard}/{i}": _wav(shard + i / 10) for i in range(3)}
        path = tmp_path / f"shard-{shard:05d}.parquet"
        _write_shard(path, clips)
        paths[shard] = path
        payloads[shard] = clips
    return paths, payloads


def test_returns_raw_wav_bytes_unchanged(shards):
    paths, payloads = shards
    cache = ShardAudio(paths.__getitem__)
    got = cache.wav_bytes(1, "u1/2")
    assert got == payloads[1]["u1/2"]
    array, sr = sf.read(io.BytesIO(got), dtype="float32")
    assert sr == 16000 and len(array) == 32


def test_unknown_utterance_raises_keyerror(shards):
    paths, _ = shards
    cache = ShardAudio(paths.__getitem__)
    with pytest.raises(KeyError):
        cache.wav_bytes(0, "nope")


def test_same_shard_loaded_once(shards):
    paths, _ = shards
    loads: list[int] = []

    def resolve(shard: int) -> Path:
        loads.append(shard)
        return paths[shard]

    cache = ShardAudio(resolve)
    cache.wav_bytes(0, "u0/0")
    cache.wav_bytes(0, "u0/1")
    assert loads == [0]


def test_lru_evicts_oldest_beyond_resident(shards):
    paths, _ = shards
    loads: list[int] = []

    def resolve(shard: int) -> Path:
        loads.append(shard)
        return paths[shard]

    cache = ShardAudio(resolve, resident=2)
    cache.wav_bytes(0, "u0/0")
    cache.wav_bytes(1, "u1/0")
    cache.wav_bytes(2, "u2/0")
    assert cache.resident_shards == [1, 2]
    cache.wav_bytes(0, "u0/0")
    assert loads == [0, 1, 2, 0]


def test_access_refreshes_lru_recency(shards):
    paths, _ = shards
    cache = ShardAudio(paths.__getitem__, resident=2)
    cache.wav_bytes(0, "u0/0")
    cache.wav_bytes(1, "u1/0")
    cache.wav_bytes(0, "u0/1")
    cache.wav_bytes(2, "u2/0")
    assert cache.resident_shards == [0, 2]


def test_context_groups_tracks_by_aircraft(tmp_path: Path):
    point = pa.struct(
        [
            ("t", pa.timestamp("us")),
            ("tail", pa.string()),
            ("aircraft_id", pa.string()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("alt", pa.int32()),
            ("speed", pa.int32()),
            ("heading", pa.int32()),
        ]
    )
    t0 = dt.datetime(2022, 1, 1, 12, 0, 0)

    def pt(tail, aid, lat, lon, sec):
        return {
            "t": t0 + dt.timedelta(seconds=sec),
            "tail": tail,
            "aircraft_id": aid,
            "lat": lat,
            "lon": lon,
            "alt": 1000,
            "speed": 120,
            "heading": 90,
        }

    tracks = [
        [
            pt("N1", "a1", 40.0, -80.0, 0),
            pt("N1", "a1", 40.1, -80.1, 1),
            pt("N2", "a2", 41.0, -81.0, 0),
        ]
    ]
    table = pa.table(
        {
            "utterance_id": pa.array(["u/0"]),
            "audio": pa.array([{"bytes": _wav(0.0), "path": "u.wav"}]),
            "tracks": pa.array(tracks, type=pa.list_(point)),
            "start": pa.array([t0], type=pa.timestamp("us")),
            "end": pa.array([t0 + dt.timedelta(seconds=2)], type=pa.timestamp("us")),
            "duration_s": pa.array([2.0]),
            "n_aircraft": pa.array([2], type=pa.int32()),
            "clip_offset_s": pa.array([0.0]),
        }
    )
    path = tmp_path / "shard-00000.parquet"
    pq.write_table(table, path)

    cache = ShardAudio({0: path}.__getitem__)
    ctx = cache.context(0, "u/0")
    assert ctx is not None
    assert ctx["n_aircraft"] == 2
    assert ctx["duration_s"] == 2.0
    by_tail = {a["tail"]: a for a in ctx["tracks"]}
    assert set(by_tail) == {"N1", "N2"}
    assert len(by_tail["N1"]["points"]) == 2
    assert by_tail["N1"]["points"][0]["lat"] == 40.0
    assert cache.wav_bytes(0, "u/0") != b""
