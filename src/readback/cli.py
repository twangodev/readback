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
def serve() -> None:
    typer.echo("serve: not implemented yet")
