from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from readback import store
from readback.fuse import fuse_clip
from readback.models.base import Hypothesis
from readback.pipeline.layout import hyps_path, labels_path, meta_path

Logger = Callable[[str], None]


def _voting(
    hyps_by_model: dict[str, dict], voters: list[str], utterance_id: str
) -> list[Hypothesis]:
    return [
        Hypothesis(
            text=hyps_by_model[model][utterance_id]["text"],
            no_speech=hyps_by_model[model][utterance_id].get("no_speech"),
        )
        for model in voters
    ]


def fuse_shard(
    run: Path,
    voters: list[str],
    weights: list[float] | None,
    advisory: str | None,
    index: int,
) -> tuple[int, int, int]:
    meta = store.index_by(meta_path(run, index), "utterance_id")
    hyps_by_model = {
        model: store.index_by(hyps_path(run, model, index), "utterance_id")
        for model in voters
    }
    advisory_hyps = (
        store.index_by(hyps_path(run, advisory, index), "utterance_id")
        if advisory
        else {}
    )

    rows = []
    skipped = 0
    for utterance_id, entry in meta.items():
        if not all(utterance_id in hyps_by_model[model] for model in voters):
            skipped += 1
            continue
        advisory_hyp = (
            [Hypothesis(text=advisory_hyps[utterance_id]["text"])]
            if utterance_id in advisory_hyps
            else None
        )
        label = fuse_clip(
            utterance_id,
            _voting(hyps_by_model, voters, utterance_id),
            entry.get("tails", []),
            weights=weights,
            advisory=advisory_hyp,
        )
        rows.append(asdict(label))
    store.write_jsonl(labels_path(run, index), rows)
    return index, len(rows), skipped


def run_fuse(
    run: Path,
    voters: list[str],
    indices: list[int],
    *,
    weights: list[float] | None = None,
    advisory: str | None = None,
    workers: int = 1,
    log: Logger = print,
) -> None:
    pending = [index for index in indices if not labels_path(run, index).exists()]
    if not pending:
        log(f"all {len(indices)} shards already fused")
        return
    log(
        f"fusing {len(pending)}/{len(indices)} shards (voters={voters} workers={workers})"
    )
    if workers > 1:
        _fuse_parallel(run, voters, weights, advisory, pending, workers, log)
    else:
        for index in pending:
            _report(log, *fuse_shard(run, voters, weights, advisory, index))


def _fuse_parallel(
    run: Path,
    voters: list[str],
    weights: list[float] | None,
    advisory: str | None,
    pending: list[int],
    workers: int,
    log: Logger,
) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fuse_shard, run, voters, weights, advisory, index)
            for index in pending
        ]
        for future in as_completed(futures):
            _report(log, *future.result())


def _report(log: Logger, index: int, labeled: int, skipped: int) -> None:
    note = f" ({skipped} missing voters)" if skipped else ""
    log(f"  shard {index}: {labeled} labels{note}")
