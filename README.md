# readback

Ensemble ASR pseudo-labeling for air-traffic-control audio, with ADS-B callsign priors and a human review studio.

Transcribes every utterance in `twangodev/tartanaviation-atc-adsb-utterances` with an ensemble of architecturally diverse ASR models, auto-accepts where independent models and the ADS-B callsign prior agree, and routes the rest to a context-rich SvelteKit review UI. Published labels train the next `rasr-parakeet` checkpoint.

Reuses sibling repos: [`airwer`](../airwer) for ATC normalization + agreement, [`radiotalk-asr`](../radiotalk-asr) for model loading on Blackwell, [`squawk`](../squawk) for the toolchain and dataset.

```bash
uv sync                # installs everything, inference deps (NeMo / faster-whisper / torch) included
uv run readback --help
```
