from __future__ import annotations

from pathlib import Path


def shard_stem(index: int) -> str:
    return f"shard-{index:05d}"


def meta_path(run: Path, index: int) -> Path:
    return run / "meta" / f"{shard_stem(index)}.jsonl"


def hyps_path(run: Path, model: str, index: int) -> Path:
    return run / "hyps" / model / f"{shard_stem(index)}.jsonl"


def labels_path(run: Path, index: int) -> Path:
    return run / "labels" / f"{shard_stem(index)}.jsonl"


def reviews_path(run: Path, index: int) -> Path:
    return run / "reviews" / f"{shard_stem(index)}.jsonl"


def snapshots_dir(run: Path) -> Path:
    return run / "snapshots"


def discover_indices(run: Path) -> list[int]:
    base = run / "meta"
    if not base.exists():
        return []
    return sorted(
        int(path.stem.removeprefix("shard-")) for path in base.glob("shard-*.jsonl")
    )
