# readback

Ensemble ASR pseudo-labeling for air-traffic-control audio. Three ASR models transcribe every
utterance in [`twangodev/tartanaviation-atc-adsb-utterances`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb-utterances),
weighted ROVER fuses them, an ADS-B callsign prior snaps the aircraft callsign, and a SvelteKit studio
reviews the uncertain ones. The result is published as
[`twangodev/tartanaviation-atc-labels`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-labels):
531,050 rows, one transcript and one confidence per utterance, joining 1:1 onto the source.

## Install

```bash
uv sync   # inference deps (NeMo, faster-whisper, torch) included
```

## Reproduce the labels

```bash
# 1. ensemble inference over the source shards  ->  data/run/{meta,hyps}
uv run readback infer --config configs/models.example.toml --run data/run

# 2. weighted ROVER fusion + ADS-B callsign snap  ->  data/run/labels
uv run readback fuse --run data/run \
    --voters parakeet-v2,canary-qwen,whisper-atc --weights 1,1,2 --advisory rasr-v1

# 3. (optional) human review studio  ->  data/run/reviews
uv run readback serve --run data/run

# 4. build the upload-ready HF folder (parquet shards mirroring the source + card)
uv run readback publish --run data/run --out out/atc-labels

# 5. push
hf upload twangodev/tartanaviation-atc-labels out/atc-labels . --repo-type dataset
```

Every stage is shard-resumable (`--shards 0-46,150` or `all`). `rasr-v1` is the in-house non-voting
advisory checkpoint; drop `--advisory` if you don't have it. `confidence` is a ranking signal, not a
calibrated probability (see the dataset card).

## Use the labels

```python
from datasets import load_dataset, concatenate_datasets

src = load_dataset("twangodev/tartanaviation-atc-adsb-utterances", split="train")
lab = load_dataset("twangodev/tartanaviation-atc-labels", split="train")
joined = concatenate_datasets([src, lab], axis=1)   # aligned 1:1, same row order
```

## Reuses

[`airwer`](../airwer) (ATC normalization + agreement), [`radiotalk-asr`](../radiotalk-asr) (model
loading on Blackwell), [`squawk`](../squawk) (toolchain + dataset).
