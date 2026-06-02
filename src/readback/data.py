from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from readback.models.base import Audio

DEFAULT_REPO = "twangodev/tartanaviation-atc-adsb-utterances"
_META_COLUMNS = ["utterance_id", "airport", "tails"]
_CLIP_COLUMNS = [*_META_COLUMNS, "audio"]


@dataclass(frozen=True, slots=True)
class Clip:
    utterance_id: str
    audio: Audio
    tails: tuple[str, ...]
    airport: str


@runtime_checkable
class ShardSource(Protocol):
    def list_indices(self) -> list[int]: ...

    def meta(self, index: int) -> list[dict]: ...

    def clips(self, index: int) -> list[Clip]: ...


def shard_filename(index: int) -> str:
    return f"shard-{index:05d}.parquet"


def parse_shard_spec(spec: str) -> list[int]:
    indices: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            low, high = (int(bound) for bound in token.split("-", 1))
            indices.update(range(low, high + 1))
        else:
            indices.add(int(token))
    return sorted(indices)


def _present_tails(raw: list[str] | None) -> tuple[str, ...]:
    return tuple(tail for tail in (raw or []) if tail)


def _meta_row(row: dict) -> dict:
    return {
        "utterance_id": str(row["utterance_id"]),
        "airport": str(row["airport"]),
        "tails": list(_present_tails(row["tails"])),
    }


def _decode(audio_bytes: bytes) -> Audio:
    import soundfile as sf

    array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if array.ndim == 2:
        array = array.mean(axis=1)
    return Audio(
        array=np.ascontiguousarray(array, dtype=np.float32),
        sample_rate=int(sample_rate),
    )


class HfShardSource:
    def __init__(self, repo_id: str = DEFAULT_REPO) -> None:
        self.repo_id = repo_id

    def list_indices(self) -> list[int]:
        from huggingface_hub import HfApi

        siblings = HfApi().dataset_info(self.repo_id).siblings or []
        names = (sibling.rfilename for sibling in siblings)
        return sorted(
            int(name.removeprefix("shard-").removesuffix(".parquet"))
            for name in names
            if name.startswith("shard-") and name.endswith(".parquet")
        )

    def _path(self, index: int) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(self.repo_id, shard_filename(index), repo_type="dataset")
        )

    def meta(self, index: int) -> list[dict]:
        import pyarrow.parquet as pq

        table = pq.read_table(self._path(index), columns=_META_COLUMNS)
        return [_meta_row(row) for row in table.to_pylist()]

    def clips(self, index: int) -> list[Clip]:
        import pyarrow.parquet as pq

        table = pq.read_table(self._path(index), columns=_CLIP_COLUMNS)
        return [
            Clip(
                utterance_id=str(row["utterance_id"]),
                audio=_decode(row["audio"]["bytes"]),
                tails=_present_tails(row["tails"]),
                airport=str(row["airport"]),
            )
            for row in table.to_pylist()
        ]
