from __future__ import annotations

import sys
from pathlib import Path

from readback.data import HfShardSource
from readback.models.registry import load_specs
from readback.pipeline.infer import transcribe_shards


def main(argv: list[str]) -> None:
    config, name, run, repo, indices_csv = argv
    indices = [int(token) for token in indices_csv.split(",") if token]
    specs = load_specs(Path(config))
    transcribe_shards(HfShardSource(repo), specs[name], indices, Path(run))


if __name__ == "__main__":
    main(sys.argv[1:])
