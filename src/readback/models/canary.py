from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from readback.models.base import Audio, Hypothesis


class CanaryQwenTranscriber:
    name = "canary-qwen"

    def __init__(self, model_ref: str, max_new_tokens: int = 256) -> None:
        import nemo.collections.speechlm2.models as speechlm2  # ty: ignore[unresolved-import]
        import torch  # ty: ignore[unresolved-import]

        model = speechlm2.SALM.from_pretrained(model_ref)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._tmp_dir = tempfile.mkdtemp(prefix="readback-canary-")
        self._audio_locator = getattr(model, "audio_locator_tag", "<audio>")

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        import soundfile as sf

        results: list[Hypothesis] = []
        for index, clip in enumerate(clips):
            wav_path = Path(self._tmp_dir) / f"clip_{index:08d}.wav"
            sf.write(
                str(wav_path),
                clip.array.astype(np.float32),
                clip.sample_rate,
                subtype="PCM_16",
            )
            answer_ids = self._model.generate(
                prompts=[
                    [
                        {
                            "role": "user",
                            "content": f"Transcribe the following: {self._audio_locator}",
                            "audio": [str(wav_path)],
                        }
                    ]
                ],
                max_new_tokens=self._max_new_tokens,
            )
            text = self._model.tokenizer.ids_to_text(answer_ids[0].cpu())
            results.append(Hypothesis(text=text.strip()))
            wav_path.unlink(missing_ok=True)
        return results
