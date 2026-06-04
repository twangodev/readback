from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path


def canonical_dumps(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def index_by(path: Path, key: str) -> dict[str, dict]:
    return {row[key]: row for row in read_jsonl(path)}


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    written = 0
    with staging.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)
    _fsync_dir(path.parent)
    return written


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_recoverable(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    lines = path.read_text().splitlines()
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if position == len(lines) - 1:
                break
            raise
    return rows
