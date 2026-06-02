from __future__ import annotations

import contextlib

import numpy as np

from readback.models.base import Audio, Hypothesis

TARGET_SR = 16000
MIN_SAMPLES = 1600


@contextlib.contextmanager
def _trusted_unpickling():
    import torch

    original = torch.load

    def load_trusting(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = load_trusting  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        torch.load = original


class ParakeetTranscriber:
    def __init__(
        self,
        model_ref: str,
        name: str,
        *,
        from_path: bool = False,
        batch_size: int = 64,
    ) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        if from_path:
            with _trusted_unpickling():
                model = nemo_asr.models.ASRModel.restore_from(model_ref)
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(model_ref)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        self._model = model
        self.name = name
        self._batch_size = batch_size

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        if not clips:
            return []
        arrays = [
            _pad_to_min(_to_target_sr(clip.array, clip.sample_rate)) for clip in clips
        ]
        order = sorted(range(len(arrays)), key=lambda index: len(arrays[index]))
        outputs = self._model.transcribe(
            audio=[arrays[index] for index in order],
            batch_size=min(self._batch_size, len(arrays)),
            verbose=False,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        results: list[Hypothesis | None] = [None] * len(arrays)
        for index, output in zip(order, outputs):
            results[index] = Hypothesis(text=_text_of(output))
        return [result for result in results if result is not None]


def _to_target_sr(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SR:
        return audio.astype(np.float32)
    import librosa

    return librosa.resample(
        audio.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
    )


def _pad_to_min(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= MIN_SAMPLES:
        return audio
    return np.pad(audio, (0, MIN_SAMPLES - len(audio)))


def _text_of(output: object) -> str:
    text = getattr(output, "text", output)
    return str(text).strip()
