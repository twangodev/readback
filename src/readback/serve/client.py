from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from readback.serve.vllm import DEFAULT_MODEL, VllmConfig


@runtime_checkable
class GerClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        choices: list[str] | None = None,
        grammar: str | None = None,
    ) -> str: ...


@dataclass(slots=True)
class VllmClient:
    model: str = DEFAULT_MODEL
    base_url: str = VllmConfig().base_url
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 128
    _client: Any = None

    def _openai(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        choices: list[str] | None = None,
        grammar: str | None = None,
    ) -> str:
        extra_body: dict[str, object] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        if choices is not None:
            extra_body["guided_choice"] = choices
        if grammar is not None:
            extra_body["guided_grammar"] = grammar

        client = self._openai()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
        )
        return completion.choices[0].message.content or ""
