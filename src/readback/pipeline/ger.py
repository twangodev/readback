from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from readback import store
from readback.fuse.ger import LabelLike, correct
from readback.fuse.tier import Tier
from readback.pipeline.layout import labels_path, meta_path, shard_stem
from readback.serve.client import GerClient

Logger = Callable[[str], None]


def ger_path(run: Path, index: int) -> Path:
    return run / "ger" / f"{shard_stem(index)}.jsonl"


def ger_partial_path(run: Path, index: int) -> Path:
    final = ger_path(run, index)
    return final.with_name(final.name + ".partial")


def _label_like(row: dict, tails: list[str]) -> LabelLike:
    return LabelLike(
        utterance_id=row["utterance_id"],
        transcript=row["transcript"],
        voting_text=tuple(row.get("voting_text", ())),
        callsign_tail=row.get("callsign_tail"),
        callsign_matched=bool(row.get("callsign_matched")),
        tails=tuple(tails),
    )


def ger_shard(run: Path, client: GerClient, index: int) -> tuple[int, int, int]:
    meta = store.index_by(meta_path(run, index), "utterance_id")
    partial = ger_partial_path(run, index)
    done = {row["utterance_id"]: row for row in store.read_jsonl_recoverable(partial)}
    rows = []
    corrected = 0
    for row in store.read_jsonl(labels_path(run, index)):
        if row.get("tier") != Tier.SILVER.value:
            rows.append({**row, "corrected": False, "callsign_enforced": False})
            continue
        utterance_id = row["utterance_id"]
        if utterance_id not in done:
            tails = meta.get(utterance_id, {}).get("tails", [])
            result = correct(client, _label_like(row, tails))
            result_row = {
                "utterance_id": utterance_id,
                "transcript": result.text,
                "corrected": result.corrected,
                "callsign_enforced": result.callsign_enforced,
            }
            store.append_jsonl(partial, result_row)
            done[utterance_id] = result_row
        result_row = done[utterance_id]
        rows.append(
            {
                **row,
                "transcript": result_row["transcript"],
                "corrected": result_row["corrected"],
                "callsign_enforced": result_row["callsign_enforced"],
            }
        )
        corrected += int(result_row["corrected"])
    store.write_jsonl(ger_path(run, index), rows)
    partial.unlink(missing_ok=True)
    return index, len(rows), corrected


def run_ger(
    run: Path,
    indices: list[int],
    *,
    client: GerClient,
    workers: int = 1,
    log: Logger = print,
) -> None:
    pending = [index for index in indices if not ger_path(run, index).exists()]
    if not pending:
        log(f"all {len(indices)} shards already corrected")
        return
    log(f"correcting silver labels in {len(pending)}/{len(indices)} shards")
    if workers > 1:
        _ger_parallel(run, client, pending, workers, log)
    else:
        for index in pending:
            _report(log, *ger_shard(run, client, index))


def _ger_parallel(
    run: Path,
    client: GerClient,
    pending: list[int],
    workers: int,
    log: Logger,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ger_shard, run, client, index) for index in pending]
        for future in as_completed(futures):
            _report(log, *future.result())


def _report(log: Logger, index: int, total: int, corrected: int) -> None:
    log(f"  shard {index}: {corrected} corrected / {total} labels")
