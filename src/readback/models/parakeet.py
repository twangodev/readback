from __future__ import annotations

import contextlib
from math import exp

import numpy as np

from readback.audio import to_target_sr
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
        beam_size: int = 1,
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
        self._beam_size = beam_size
        if beam_size > 1:
            self._enable_beam(beam_size)

    def _enable_beam(self, beam_size: int) -> None:
        import copy

        from omegaconf import open_dict

        cfg = copy.deepcopy(self._model.cfg.decoding)
        with open_dict(cfg):
            cfg.strategy = "malsd_batch"
            cfg.beam.beam_size = beam_size
            cfg.beam.return_best_hypothesis = False
        self._model.change_decoding_strategy(cfg)

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        if not clips:
            return []
        arrays = [
            _pad_to_min(to_target_sr(clip.array, clip.sample_rate, TARGET_SR))
            for clip in clips
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
            results[index] = self._hypothesis(output)
        return [result for result in results if result is not None]

    def _hypothesis(self, output: object) -> Hypothesis:
        if self._beam_size <= 1:
            return Hypothesis(text=_text_of(output), confidence=_confidence_of(output))
        beams = output if isinstance(output, list) else [output]
        nbest = tuple((_text_of(beam), _score_of(beam)) for beam in beams)
        if not nbest:
            return Hypothesis(text="")
        return Hypothesis(
            text=nbest[0][0], confidence=_confidence_of(beams[0]), nbest=nbest
        )


def _pad_to_min(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= MIN_SAMPLES:
        return audio
    return np.pad(audio, (0, MIN_SAMPLES - len(audio)))


def _text_of(output: object) -> str:
    text = getattr(output, "text", output)
    return str(text).strip()


def _score_of(output: object) -> float:
    return float(getattr(output, "score", 0.0))


def _confidence_of(output: object) -> float | None:
    score = getattr(output, "score", None)
    sequence = getattr(output, "y_sequence", None)
    if score is None or sequence is None or len(sequence) == 0:
        return None
    return exp(min(0.0, float(score) / len(sequence)))
