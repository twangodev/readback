from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

PathResolver = Callable[[int], Path]

_SCALAR_COLUMNS = ["start", "end", "duration_s", "n_aircraft", "clip_offset_s"]
_COLUMNS = ["utterance_id", "audio", "tracks", *_SCALAR_COLUMNS]


def _epoch(value) -> float | None:
    return value.timestamp() if value is not None else None


def _tracks(raw: list[dict] | None) -> list[dict]:
    if not raw:
        return []
    grouped: dict[str, dict] = {}
    for point in raw:
        key = point.get("aircraft_id") or point.get("tail") or ""
        aircraft = grouped.setdefault(
            key,
            {
                "tail": point.get("tail"),
                "aircraft_id": point.get("aircraft_id"),
                "points": [],
            },
        )
        aircraft["points"].append(
            {
                "t": _epoch(point.get("t")),
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "alt": point.get("alt"),
                "speed": point.get("speed"),
                "heading": point.get("heading"),
            }
        )
    for aircraft in grouped.values():
        aircraft["points"].sort(key=lambda p: (p["t"] is None, p["t"]))
    return list(grouped.values())


class ShardAudio:
    def __init__(self, resolve: PathResolver, resident: int = 2) -> None:
        self._resolve = resolve
        self._resident = resident
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._lock = threading.Lock()

    def wav_bytes(self, shard: int, utterance_id: str) -> bytes:
        audio = self._shard(shard)["audio"]
        if utterance_id not in audio:
            raise KeyError(utterance_id)
        return audio[utterance_id]

    def clip_start(self, shard: int, utterance_id: str) -> float | None:
        data = self._shard(shard)
        index = data["index"].get(utterance_id)
        if index is None:
            return None
        return data["scalars"]["start"][index]

    def context(self, shard: int, utterance_id: str) -> dict | None:
        data = self._shard(shard)
        index = data["index"].get(utterance_id)
        if index is None:
            return None
        context = {key: values[index] for key, values in data["scalars"].items()}
        tracks = data["tracks"]
        context["tracks"] = _tracks(tracks[index].as_py()) if tracks is not None else []
        return context

    @property
    def resident_shards(self) -> list[int]:
        return list(self._cache)

    def _shard(self, shard: int) -> dict:
        with self._lock:
            if shard in self._cache:
                self._cache.move_to_end(shard)
                return self._cache[shard]
            loaded = self._load(shard)
            self._cache[shard] = loaded
            while len(self._cache) > self._resident:
                self._cache.popitem(last=False)
            return loaded

    def _load(self, shard: int) -> dict:
        import pyarrow.parquet as pq

        path = self._resolve(shard)
        available = set(pq.read_schema(path).names)
        table = pq.read_table(path, columns=[c for c in _COLUMNS if c in available])
        uids = [str(uid) for uid in table.column("utterance_id").to_pylist()]
        index = {uid: position for position, uid in enumerate(uids)}
        audio: dict[str, bytes] = {}
        if "audio" in available:
            for uid, cell in zip(uids, table.column("audio").to_pylist()):
                audio[uid] = cell["bytes"] if cell else b""
        scalars: dict[str, list] = {}
        for column in _SCALAR_COLUMNS:
            if column in available:
                values = table.column(column).to_pylist()
                if column in ("start", "end"):
                    values = [_epoch(value) for value in values]
                scalars[column] = values
            else:
                scalars[column] = [None] * len(uids)
        tracks = table.column("tracks") if "tracks" in available else None
        return {"audio": audio, "index": index, "scalars": scalars, "tracks": tracks}
