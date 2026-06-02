from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


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
    staging.replace(path)
    return written


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


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
