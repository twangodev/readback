from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="Ensemble ASR pseudo-labeling for ATC.")


@app.command()
def infer() -> None:
    typer.echo("infer: not implemented yet")


@app.command()
def fuse() -> None:
    typer.echo("fuse: not implemented yet")


@app.command()
def serve() -> None:
    typer.echo("serve: not implemented yet")
