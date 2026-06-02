from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from readback import store
from readback.data import ShardSource
from readback.models.base import Hypothesis, Transcriber
from readback.models.registry import ModelSpec, build
from readback.pipeline.layout import hyps_path, meta_path

Logger = Callable[[str], None]


def release_gpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hyp_row(utterance_id: str, hypothesis: Hypothesis) -> dict:
    return {
        "utterance_id": utterance_id,
        "text": hypothesis.text,
        "confidence": hypothesis.confidence,
        "no_speech": hypothesis.no_speech,
    }


def _write_meta(
    source: ShardSource, indices: list[int], run: Path, log: Logger
) -> None:
    for index in indices:
        path = meta_path(run, index)
        if path.exists():
            continue
        store.write_jsonl(path, source.meta(index))
    log(f"meta ready for {len(indices)} shards")


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
        pending = [
            index for index in indices if not hyps_path(run, name, index).exists()
        ]
        if not pending:
            log(f"skip {name} (all {len(indices)} shards done)")
            continue
        log(f"loading {name} ({len(pending)}/{len(indices)} shards pending)")
        try:
            model = build_model(specs[name])
            for index in pending:
                clips = source.clips(index)
                hyps = model.transcribe([clip.audio for clip in clips])
                store.write_jsonl(
                    hyps_path(run, name, index),
                    (
                        _hyp_row(clip.utterance_id, hyp)
                        for clip, hyp in zip(clips, hyps)
                    ),
                )
                log(f"  {name} shard {index}: {len(clips)} clips")
            del model
        except Exception as error:
            log(f"FAILED {name}: {type(error).__name__}: {error}")
        release()
