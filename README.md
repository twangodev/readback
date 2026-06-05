# readback

Ensemble ASR pseudo-labeling for Air Traffic Control.

Labels [`tartanaviation-atc-adsb-utterances`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb-utterances)
and publishes [`tartanaviation-atc-labels`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-labels):
531k rows, one transcript and confidence each, 1:1 onto the source.

## Pipeline

| stage | does |
|---|---|
| `infer` | three ASR models over the source shards |
| `fuse` | weighted ROVER + ADS-B callsign snap |
| `serve` | review studio (optional) |
| `publish` | upload-ready parquet shards + card |

```bash
uv sync
uv run readback infer   --config configs/models.example.toml --run data/run
uv run readback fuse    --run data/run --voters parakeet-v2,canary-qwen,whisper-atc --weights 1,1,2 --advisory rasr-v1
uv run readback serve   --run data/run
uv run readback publish --run data/run --out out/atc-labels
hf upload twangodev/tartanaviation-atc-labels out/atc-labels . --repo-type dataset
```

Shard-resumable. `confidence` ranks; it is not calibrated.

## Use

```python
from datasets import load_dataset, concatenate_datasets

src = load_dataset("twangodev/tartanaviation-atc-adsb-utterances", split="train")
lab = load_dataset("twangodev/tartanaviation-atc-labels", split="train")
joined = concatenate_datasets([src, lab], axis=1)   # 1:1, same order
```

Reuses [`airwer`](https://github.com/twangodev/airwer), [`rasr`](https://github.com/twangodev/rasr), [`squawk`](https://github.com/twangodev/squawk).
