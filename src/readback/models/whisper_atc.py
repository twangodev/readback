from __future__ import annotations

from math import exp

from readback.audio import to_target_sr
from readback.models.base import Audio, Hypothesis

TARGET_SR = 16000


class WhisperAtcTranscriber:
    name = "whisper-atc"

    def __init__(
        self,
        model_id: str,
        batch_size: int = 16,
        dtype: str = "float16",
        beam_size: int = 5,
    ) -> None:
        import torch
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            WhisperTokenizer,
        )

        self._processor = WhisperProcessor.from_pretrained(model_id)
        tokenizer = WhisperTokenizer.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = (
            WhisperForConditionalGeneration.from_pretrained(
                model_id, dtype=getattr(torch, dtype)
            )
            .to(device)
            .eval()
        )
        self._batch_size = batch_size
        self._beam_size = beam_size
        self._no_speech_id = tokenizer.convert_tokens_to_ids("<|nospeech|>")
        self._prefix = [
            self._model.config.decoder_start_token_id,
            tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        ]

    def transcribe(
        self, clips: list[Audio], bias_terms: list[str] | None = None
    ) -> list[Hypothesis]:
        if not clips:
            return []
        results: list[Hypothesis] = []
        for start in range(0, len(clips), self._batch_size):
            results.extend(
                self._transcribe_batch(clips[start : start + self._batch_size])
            )
        return results

    def _transcribe_batch(self, clips: list[Audio]) -> list[Hypothesis]:
        import torch

        waveforms = [
            to_target_sr(clip.array, clip.sample_rate, TARGET_SR) for clip in clips
        ]
        features = self._processor(
            waveforms, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self._model.device, self._model.dtype)

        with torch.no_grad():
            no_speech = self._no_speech_probs(features)
            generated = self._model.generate(
                input_features=features,
                language="en",
                task="transcribe",
                num_beams=self._beam_size,
                return_dict_in_generate=True,
                output_scores=True,
            )
        texts = self._processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )
        scores = [exp(min(0.0, score)) for score in generated.sequences_scores.tolist()]
        return [
            Hypothesis(text=text.strip(), confidence=confidence, no_speech=prob)
            for text, confidence, prob in zip(texts, scores, no_speech)
        ]

    def _no_speech_probs(self, features) -> list[float]:
        import torch

        prefix = (
            torch.tensor(self._prefix, dtype=torch.long, device=features.device)
            .unsqueeze(0)
            .expand(features.shape[0], -1)
        )
        logits = self._model(input_features=features, decoder_input_ids=prefix).logits
        probs = logits[:, -1].float().softmax(dim=-1)
        return probs[:, self._no_speech_id].tolist()
