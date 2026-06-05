from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from readback.data import DEFAULT_REPO, HfShardSource, parse_shard_spec
from readback.models.registry import load_specs
from readback.pipeline import run_fuse, run_infer, run_infer_parallel
from readback.pipeline.layout import discover_indices

app = typer.Typer(add_completion=False, help="Ensemble ASR pseudo-labeling for ATC.")


def _comma_list(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


@app.command()
def infer(
    config: Annotated[Path, typer.Option(help="TOML mapping model names to kind/ref.")],
    run: Annotated[Path, typer.Option(help="Run directory for meta/ and hyps/.")],
    shards: Annotated[
        str, typer.Option(help="Index spec like '0-46,150' or 'all'.")
    ] = "all",
    repo: Annotated[
        str, typer.Option(help="HF dataset repo of parquet shards.")
    ] = DEFAULT_REPO,
    models: Annotated[
        str, typer.Option(help="Comma list; default every model in the config.")
    ] = "",
    replicas: Annotated[
        int, typer.Option(help="Data-parallel worker processes per model.")
    ] = 1,
) -> None:
    specs = load_specs(config)
    chosen = _comma_list(models) or list(specs)
    missing = [name for name in chosen if name not in specs]
    if missing:
        raise typer.BadParameter(f"models not in config: {missing}")
    source = HfShardSource(repo)
    indices = source.list_indices() if shards == "all" else parse_shard_spec(shards)
    if replicas > 1:
        run_infer_parallel(config, chosen, indices, run, replicas=replicas, repo=repo)
    else:
        run_infer(source, specs, chosen, indices, run)


@app.command()
def fuse(
    run: Annotated[Path, typer.Option(help="Run directory holding meta/ and hyps/.")],
    voters: Annotated[str, typer.Option(help="Comma list of voting model names.")],
    weights: Annotated[
        str, typer.Option(help="Comma list of weights, aligned with voters.")
    ] = "",
    advisory: Annotated[
        str, typer.Option(help="Non-voting model compared against the fused text.")
    ] = "",
    shards: Annotated[
        str, typer.Option(help="Index spec or 'all' (default: every fused shard).")
    ] = "all",
    workers: Annotated[int, typer.Option(help="Process workers for fusion.")] = 1,
) -> None:
    voter_names = _comma_list(voters)
    weight_values = [float(token) for token in _comma_list(weights)] or None
    if weight_values and len(weight_values) != len(voter_names):
        raise typer.BadParameter("weights must align with voters")
    indices = discover_indices(run) if shards == "all" else parse_shard_spec(shards)
    run_fuse(
        run,
        voter_names,
        indices,
        weights=weight_values,
        advisory=advisory or None,
        workers=workers,
    )


@app.command()
def snapshot(
    run: Annotated[Path, typer.Option(help="Run directory with labels/ and reviews/.")],
    name: Annotated[str, typer.Option(help="Snapshot name (snapshots/<name>.json).")],
    tiers: Annotated[
        str, typer.Option(help="Training-view tiers recorded in the manifest.")
    ] = "",
) -> None:
    from readback.snapshot import write_snapshot

    path = write_snapshot(run, name, tiers=_comma_list(tiers) or None)
    typer.echo(str(path))


@app.command()
def trainset(
    run: Annotated[Path, typer.Option(help="Run directory with labels/ and reviews/.")],
    out: Annotated[Path, typer.Option(help="Output training JSONL manifest.")],
    tiers: Annotated[
        str, typer.Option(help="Auto-label tiers to include; human edits always win.")
    ] = "gold,silver",
) -> None:
    from readback.trainset import write_trainset

    count = write_trainset(run, out, tuple(_comma_list(tiers)))
    typer.echo(f"{count} examples -> {out}")


@app.command()
def publish(
    run: Annotated[Path, typer.Option(help="Run directory with labels/ and reviews/.")],
    out: Annotated[Path, typer.Option(help="Output published-dataset JSONL.")],
) -> None:
    from readback.dataset import write_dataset

    count = write_dataset(run, out)
    typer.echo(f"{count} rows -> {out}")


@app.command()
def ger(
    run: Annotated[Path, typer.Option(help="Run directory with labels/ and meta/.")],
    base_url: Annotated[
        str, typer.Option(help="vLLM OpenAI-compatible base URL.")
    ] = "http://127.0.0.1:8000/v1",
    model: Annotated[
        str, typer.Option(help="Served model id; default the configured Qwen.")
    ] = "",
    shards: Annotated[str, typer.Option(help="Index spec or 'all'.")] = "all",
    workers: Annotated[int, typer.Option(help="Concurrent correction requests.")] = 8,
) -> None:
    from readback.pipeline.ger import run_ger
    from readback.serve.client import VllmClient
    from readback.serve.vllm import DEFAULT_MODEL

    indices = discover_indices(run) if shards == "all" else parse_shard_spec(shards)
    client = VllmClient(model=model or DEFAULT_MODEL, base_url=base_url)
    run_ger(run, indices, client=client, workers=workers)


@app.command()
def serve(
    run: Annotated[Path, typer.Option(help="Run directory with labels/ and meta/.")],
    repo: Annotated[
        str, typer.Option(help="HF dataset repo of parquet shards.")
    ] = DEFAULT_REPO,
    web: Annotated[
        Path | None,
        typer.Option(help="Built SPA directory; defaults to web/build if present."),
    ] = None,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    resident: Annotated[
        int, typer.Option(help="Shards kept resident in the audio LRU.")
    ] = 2,
) -> None:
    import fcntl

    import uvicorn

    from readback.server.app import build_app, hf_audio

    if web is None:
        bundled = Path(__file__).resolve().parents[2] / "web" / "build"
        web = bundled if bundled.exists() else None
    elif not web.exists():
        raise typer.BadParameter(f"web directory not found: {web}")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise typer.BadParameter(
            f"another studio already holds {run / '.lock'}"
        ) from error
    served = build_app(run, hf_audio(repo, resident), web)
    uvicorn.run(served, host=host, port=port)
