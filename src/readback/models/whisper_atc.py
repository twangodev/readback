from __future__ import annotations

from math import exp

from readback.audio import to_target_sr
from readback.models.base import Audio, Hypothesis


class WhisperAtcTranscriber:
    name = "whisper-atc"

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ) -> None:
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer

        self._model = WhisperModel(model_dir, device=device, compute_type=compute_type)
        self._tokenizer = Tokenizer(
            self._model.hf_tokenizer,
            self._model.model.is_multilingual,
            task="transcribe",
            language="en",
        )
        self._batch_size = batch_size
        self._target_sr = self._model.feature_extractor.sampling_rate

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        import numpy as np
        from faster_whisper.audio import pad_or_trim

        if not clips:
            return []
        hotwords = " ".join(bias_terms) if bias_terms else None
        prompt = self._model.get_prompt(
            self._tokenizer, [], without_timestamps=True, hotwords=hotwords
        )
        results: list[Hypothesis] = []
        for start in range(0, len(clips), self._batch_size):
            chunk = clips[start : start + self._batch_size]
            features = np.stack([self._features(clip, pad_or_trim) for clip in chunk])
            encoder_output = self._model.encode(features)
            outputs = self._model.model.generate(
                encoder_output,
                [prompt.copy() for _ in chunk],
                beam_size=5,
                patience=1,
                length_penalty=1,
                max_length=self._model.max_length,
                suppress_blank=True,
                suppress_tokens=[-1],
                return_scores=True,
                return_no_speech_prob=True,
                sampling_temperature=0.0,
            )
            results.extend(_decode(self._tokenizer, outputs))
        return results

    def _features(self, clip: Audio, pad_or_trim):
        waveform = to_target_sr(clip.array, clip.sample_rate, self._target_sr)
        return pad_or_trim(self._model.feature_extractor(waveform)[..., :-1])


def _decode(tokenizer, outputs):
    for result in outputs:
        tokens = result.sequences_ids[0] if result.sequences_ids else []
        seq_len = len(tokens)
        avg_logprob = result.scores[0] * seq_len / (seq_len + 1) if seq_len else 0.0
        yield Hypothesis(
            text=tokenizer.decode(tokens).strip(),
            confidence=exp(avg_logprob),
            no_speech=result.no_speech_prob,
        )
