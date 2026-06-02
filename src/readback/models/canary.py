from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from readback.audio import to_target_sr
from readback.models.base import Audio, Hypothesis

if TYPE_CHECKING:
    import torch


class CanaryQwenTranscriber:
    name = "canary-qwen"

    def __init__(
        self, model_ref: str, max_new_tokens: int = 256, batch_size: int = 16
    ) -> None:
        import nemo.collections.speechlm2.models as speechlm2
        import torch

        model = speechlm2.SALM.from_pretrained(model_ref)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        locator = getattr(model, "audio_locator_tag", "<audio>")
        self._prompt = f"Transcribe the following: {locator}"

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        if not clips:
            return []
        target_sr = self._model.sampling_rate
        order = sorted(
            range(len(clips)), key=lambda i: len(clips[i].array) / clips[i].sample_rate
        )
        results: list[Hypothesis | None] = [None] * len(clips)
        for start in range(0, len(order), self._batch_size):
            indices = order[start : start + self._batch_size]
            waves = [
                to_target_sr(clips[i].array, clips[i].sample_rate, target_sr)
                for i in indices
            ]
            audios, audio_lens = _collate(waves, self._model.device)
            answer_ids = self._model.generate(
                prompts=[[{"role": "user", "content": self._prompt}] for _ in indices],
                audios=audios,
                audio_lens=audio_lens,
                max_new_tokens=self._max_new_tokens,
            )
            for index, row in zip(indices, answer_ids):
                text = self._model.tokenizer.ids_to_text(row.cpu())
                results[index] = Hypothesis(text=text.strip())
        return [result for result in results if result is not None]


def _collate(waves: list[np.ndarray], device: torch.device):
    import torch

    lengths = torch.tensor([len(wave) for wave in waves], dtype=torch.int64)
    batch = torch.zeros(len(waves), int(lengths.max()), dtype=torch.float32)
    for index, wave in enumerate(waves):
        batch[index, : len(wave)] = torch.from_numpy(
            np.ascontiguousarray(wave, dtype=np.float32)
        )
    return batch.to(device), lengths.to(device)
