from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from readback.models.base import Transcriber


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    kind: str
    ref: str
    options: dict = field(default_factory=dict)


def load_specs(path: Path) -> dict[str, ModelSpec]:
    table = tomllib.loads(path.read_text())
    return {
        name: ModelSpec(
            name=name,
            kind=body["kind"],
            ref=body["ref"],
            options={
                key: value for key, value in body.items() if key not in ("kind", "ref")
            },
        )
        for name, body in table.get("models", {}).items()
    }


def build(spec: ModelSpec) -> Transcriber:
    if spec.kind == "parakeet":
        from readback.models.parakeet import ParakeetTranscriber

        return ParakeetTranscriber(
            spec.ref,
            spec.name,
            from_path=spec.options.get("from_path", False),
            batch_size=spec.options.get("batch_size", 64),
            beam_size=spec.options.get("beam_size", 1),
        )
    if spec.kind == "canary":
        from readback.models.canary import CanaryQwenTranscriber

        return CanaryQwenTranscriber(
            spec.ref,
            max_new_tokens=spec.options.get("max_new_tokens", 256),
            batch_size=spec.options.get("batch_size", 16),
        )
    if spec.kind == "whisper":
        from readback.models.whisper_atc import WhisperAtcTranscriber

        return WhisperAtcTranscriber(
            spec.ref,
            batch_size=spec.options.get("batch_size", 16),
            dtype=spec.options.get("dtype", "float16"),
            beam_size=spec.options.get("beam_size", 5),
        )
    raise ValueError(f"unknown model kind: {spec.kind!r}")
