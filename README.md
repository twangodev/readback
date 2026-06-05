# readback

Ensemble ASR pseudo-labeling for ATC audio. Three models transcribe
[`tartanaviation-atc-adsb-utterances`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb-utterances),
weighted ROVER and an ADS-B callsign prior fuse them, a studio reviews the uncertain ones, and the
result publishes as [`tartanaviation-atc-labels`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-labels):
531k rows, one transcript and confidence each, 1:1 onto the source.

## Pipeline

| stage | does | writes |
|---|---|---|
| `infer` | three ASR models over the source shards | `meta/`, `hyps/` |
| `fuse` | weighted ROVER + ADS-B callsign snap | `labels/` |
| `serve` | review studio (optional) | `reviews/` |
| `publish` | upload-ready parquet shards + card | `out/` |

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

Reuses [`airwer`](../airwer), [`radiotalk-asr`](../radiotalk-asr), [`squawk`](../squawk).
