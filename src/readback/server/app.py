from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from readback.server.api import create_app
from readback.server.shard_cache import ShardAudio
from readback.server.state import StudioState


def build_app(
    run: Path,
    audio: ShardAudio,
    web_dir: Path | None = None,
    reviewer: str | None = None,
) -> FastAPI:
    return create_app(StudioState(run, audio, reviewer), web_dir)


def hf_audio(repo: str, resident: int = 2) -> ShardAudio:
    from huggingface_hub import hf_hub_download

    from readback.data import shard_filename

    def resolve(shard: int) -> Path:
        return Path(hf_hub_download(repo, shard_filename(shard), repo_type="dataset"))

    return ShardAudio(resolve, resident)
