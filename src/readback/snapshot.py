from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path

from readback import store
from readback.pipeline.layout import shard_stem, snapshots_dir
from readback.review import load_review_log, merge_label


def _shard_index(path: Path) -> int:
    return int(path.stem.removeprefix("shard-"))


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _shard_digest(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda row: row["utterance_id"])
    return _digest("\n".join(store.canonical_dumps(row) for row in ordered))


def build_snapshot(
    run: Path, name: str, tiers: list[str] | None = None, ts: float | None = None
) -> dict:
    log = load_review_log(run)
    shards: dict[str, str] = {}
    counts: Counter[str] = Counter()
    total = 0
    for path in sorted((run / "labels").glob("shard-*.jsonl")):
        merged = [
            merge_label(label, log.current(label["utterance_id"]))
            for label in store.read_jsonl(path)
        ]
        shards[shard_stem(_shard_index(path))] = _shard_digest(merged)
        for row in merged:
            counts[row["tier"]] += 1
            total += 1
    digest = _digest(
        "\n".join(f"{key}:{value}" for key, value in sorted(shards.items()))
    )
    return {
        "name": name,
        "ts": time.time() if ts is None else ts,
        "review_offset": log.max_seq(),
        "digest": digest,
        "shards": shards,
        "total": total,
        "by_tier": dict(counts),
        "tiers": tiers,
    }


def write_snapshot(
    run: Path, name: str, tiers: list[str] | None = None, ts: float | None = None
) -> Path:
    manifest = build_snapshot(run, name, tiers=tiers, ts=ts)
    path = snapshots_dir(run) / f"{name}.json"
    store.write_jsonl(path, [manifest])
    return path
