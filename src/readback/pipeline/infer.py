from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from readback import store
from readback.data import DEFAULT_REPO, HfShardSource, ShardSource
from readback.models.base import Hypothesis, Transcriber
from readback.models.registry import ModelSpec, build
from readback.pipeline.layout import hyps_path, meta_path

Logger = Callable[[str], None]
WorkerRunner = Callable[[Path, str, Path, str, "list[list[int]]", Logger], None]


def release_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hyp_row(utterance_id: str, hypothesis: Hypothesis) -> dict:
    row: dict[str, object] = {
        "utterance_id": utterance_id,
        "text": hypothesis.text,
        "confidence": hypothesis.confidence,
        "no_speech": hypothesis.no_speech,
    }
    if hypothesis.nbest is not None:
        row["nbest"] = [[text, score] for text, score in hypothesis.nbest]
    return row


def _write_meta(
    source: ShardSource, indices: list[int], run: Path, log: Logger
) -> None:
    for index in indices:
        path = meta_path(run, index)
        if path.exists():
            continue
        store.write_jsonl(path, source.meta(index))
    log(f"meta ready for {len(indices)} shards")


def transcribe_shards(
    source: ShardSource,
    spec: ModelSpec,
    indices: list[int],
    run: Path,
    *,
    build_model: Callable[[ModelSpec], Transcriber] = build,
    release: Callable[[], None] = release_gpu,
    log: Logger = print,
) -> None:
    pending = [
        index for index in indices if not hyps_path(run, spec.name, index).exists()
    ]
    if not pending:
        log(f"skip {spec.name} (all {len(indices)} shards done)")
        return
    log(f"loading {spec.name} ({len(pending)}/{len(indices)} shards pending)")
    try:
        model = build_model(spec)
        for index in pending:
            clips = source.clips(index)
            hyps = model.transcribe([clip.audio for clip in clips])
            store.write_jsonl(
                hyps_path(run, spec.name, index),
                (_hyp_row(clip.utterance_id, hyp) for clip, hyp in zip(clips, hyps)),
            )
            log(f"  {spec.name} shard {index}: {len(clips)} clips")
        del model
    except Exception as error:
        log(f"FAILED {spec.name}: {type(error).__name__}: {error}")
    release()


def run_infer(
    source: ShardSource,
    specs: dict[str, ModelSpec],
    model_names: list[str],
    indices: list[int],
    run: Path,
    *,
    build_model: Callable[[ModelSpec], Transcriber] = build,
    release: Callable[[], None] = release_gpu,
    log: Logger = print,
) -> None:
    _write_meta(source, indices, run, log)
    for name in model_names:
        transcribe_shards(
            source,
            specs[name],
            indices,
            run,
            build_model=build_model,
            release=release,
            log=log,
        )


def _spawn_workers(
    config_path: Path,
    name: str,
    run: Path,
    repo: str,
    groups: list[list[int]],
    log: Logger,
) -> None:
    import subprocess
    import sys

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "readback.models.infer_worker",
                str(config_path),
                name,
                str(run),
                repo,
                ",".join(str(index) for index in group),
            ]
        )
        for group in groups
    ]
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        log(f"FAILED {name}: replica exit codes {codes}")


def run_infer_parallel(
    config_path: Path,
    model_names: list[str],
    indices: list[int],
    run: Path,
    *,
    replicas: int,
    repo: str = DEFAULT_REPO,
    source: ShardSource | None = None,
    run_workers: WorkerRunner = _spawn_workers,
    log: Logger = print,
) -> None:
    source = source or HfShardSource(repo)
    _write_meta(source, indices, run, log)
    for name in model_names:
        pending = [
            index for index in indices if not hyps_path(run, name, index).exists()
        ]
        if not pending:
            log(f"skip {name} (all {len(indices)} shards done)")
            continue
        groups = [
            group for group in (pending[r::replicas] for r in range(replicas)) if group
        ]
        log(f"{name}: {len(pending)} shards across {len(groups)} replicas")
        run_workers(config_path, name, run, repo, groups, log)
