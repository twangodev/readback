from __future__ import annotations

from math import exp

from readback.models.base import Audio, Hypothesis


class WhisperAtcTranscriber:
    name = "whisper-atc"

    def __init__(
        self, model_dir: str, device: str = "cuda", compute_type: str = "float16"
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_dir, device=device, compute_type=compute_type)

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        hotwords = " ".join(bias_terms) if bias_terms else None
        return [self._transcribe_one(clip, hotwords) for clip in clips]

    def _transcribe_one(self, clip: Audio, hotwords: str | None) -> Hypothesis:
        segments, _ = self._model.transcribe(
            clip.array,
            language="en",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            hotwords=hotwords,
        )
        materialized = list(segments)
        text = " ".join(segment.text.strip() for segment in materialized).strip()
        return Hypothesis(
            text=text,
            confidence=_mean_confidence(materialized),
            no_speech=_peak_no_speech(materialized),
        )


def _peak_no_speech(segments: list) -> float | None:
    if not segments:
        return None
    return max(segment.no_speech_prob for segment in segments)


def _mean_confidence(segments: list) -> float | None:
    if not segments:
        return None
    mean_logprob = sum(segment.avg_logprob for segment in segments) / len(segments)
    return exp(mean_logprob)
