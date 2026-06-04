from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MODEL = "Qwen/Qwen3-35B-A3B-FP8"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


@dataclass(frozen=True, slots=True)
class VllmConfig:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_model_len: int = 4096
    max_num_batched_tokens: int = 2096
    gpu_memory_utilization: float = 0.92
    reasoning_parser: str = "qwen3"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"


def serve_command(config: VllmConfig = VllmConfig()) -> list[str]:
    return [
        "vllm",
        "serve",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        "1",
        "--language-model-only",
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--gpu-memory-utilization",
        f"{config.gpu_memory_utilization}",
        "--reasoning-parser",
        config.reasoning_parser,
        "--enable-prefix-caching",
        *config.extra_args,
    ]


def is_ready(config: VllmConfig = VllmConfig(), *, timeout: float = 2.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(config.health_url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_until_ready(
    config: VllmConfig = VllmConfig(),
    *,
    attempts: int = 60,
    interval: float = 5.0,
) -> bool:
    import time

    for _ in range(attempts):
        if is_ready(config):
            return True
        time.sleep(interval)
    return False
