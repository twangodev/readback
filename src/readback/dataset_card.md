---
license: cc-by-4.0
language:
  - en
pretty_name: TartanAviation ATC — Ensemble ASR Labels
task_categories:
  - automatic-speech-recognition
size_categories:
  - 100K<n<1M
source_datasets:
  - twangodev/tartanaviation-atc-adsb-utterances
tags:
  - atc
  - air-traffic-control
  - speech-recognition
  - asr
  - aviation
  - pseudo-labels
  - rover
  - ensemble
  - adsb
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/shard-*.parquet
---

# TartanAviation ATC — Ensemble ASR Labels

A derived, **labels-only** companion to
[`twangodev/tartanaviation-atc-adsb-utterances`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb-utterances).
**No audio is included here** — only one machine-produced transcript and a set of
confidence/agreement scores per utterance, keyed by `utterance_id`, designed to join **1:1**
onto the source utterances dataset.

- **Rows:** __N_ROWS__ (100% coverage of the source corpus)
- **Shards:** __N_SHARDS__ parquet files, `data/shard-00000.parquet` … matching the source shard layout and row order
- **One transcript + one confidence per utterance**, plus the raw ensemble signals so you can set your own quality cutoff

## How it joins the source dataset

This is a **direct 1:1 add-on**: shard `data/shard-NNNNN.parquet` here aligns to `shard-NNNNN`
in the source, rows are in the same order, and every row carries the `utterance_id` join key.

```python
from datasets import load_dataset, concatenate_datasets

src = load_dataset("twangodev/tartanaviation-atc-adsb-utterances", split="train")
lab = load_dataset("twangodev/tartanaviation-atc-labels", split="train")

assert len(src) == len(lab)                 # rows are aligned 1:1, same order
joined = concatenate_datasets([src, lab], axis=1)   # fast path: column concat

# safe fallback if you don't want to trust row order — join on the key:
# src_idx = {u: i for i, u in enumerate(src["utterance_id"])}
# joined = concatenate_datasets([src.select([src_idx[u] for u in lab["utterance_id"]]), lab], axis=1)
```

## Schema

One row per utterance; all fields are derived (the source dataset is unchanged).

| Field | Type | Description |
|---|---|---|
| `utterance_id` | string | Join key; matches one source row 1:1. |
| `transcript` | string | Final fused transcript. `""` when no usable speech; reflects a human edit when `review_status == "edited"`. |
| `confidence` | float (0–1) | Single quality score (higher = better). Human review overrides it (see `review_status`); otherwise it is the mean of the available ensemble signals (`agreement_score`, `rover_confidence`, and `1 − advisory_disagree` when present). **Use it for ranking and pick your own cutoff — see the calibration note.** |
| `review_status` | string | `auto`, `verified`, `edited`, `rejected`, or `non_speech` (see below). |
| `agreement_score` | float (0–1) | Inter-model agreement among the voting ASR models. |
| `rover_confidence` | float (0–1) | ROVER's mean per-slot winning-vote fraction. |
| `advisory_disagree` | float (0–1) or null | Disagreement between the fused text and an out-of-vote advisory model (`1 − word agreement`); `null` if unavailable. |
| `n_models_agree` | int | Size of the agreeing plurality among the voting models. |
| `callsign_matched` | bool | Whether an ADS-B-derived callsign was matched and snapped into the transcript. |
| `callsign_tail` | string or null | The matched aircraft tail/callsign when `callsign_matched`, else `null`. |

### `review_status`

| Value | Meaning | `confidence` |
|---|---|---|
| `auto` | Not human-reviewed; transcript + score from the ensemble. **Vast majority of rows.** | computed |
| `verified` | Human accepted the automatic transcript. | `1.0` |
| `edited` | Human corrected the transcript (`transcript` is the human edit). | `1.0` |
| `rejected` | Human judged the transcript unusable. | `0.0` |
| `non_speech` | No usable speech (silence/noise); `transcript == ""`. | `1.0` |

`confidence == 1.0` is used for both human-verified/edited transcripts and confident non-speech —
use `review_status` to distinguish. `confidence == 0.0` only appears for human-`rejected` rows.

## How labels were produced

1. **Ensemble transcription** by three ASR models —
   [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
   [`nvidia/canary-qwen-2.5b`](https://huggingface.co/nvidia/canary-qwen-2.5b), and
   [`jlvdoorn/whisper-large-v3-atco2-asr`](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr).
2. **Weighted ROVER fusion** of the voting hypotheses into one transcript (`rover_confidence`,
   `agreement_score`, `n_models_agree`).
3. **ADS-B callsign snap** — when an aircraft callsign was available from ADS-B, the fused text was
   fuzzy-matched and snapped to the correct spelling (`callsign_matched`, `callsign_tail`).
4. **Advisory cross-check** — an out-of-vote model's disagreement (`advisory_disagree`) folded into
   the confidence as a tie-breaking signal.
5. **Partial human review (~326 clips)** through a review UI; verdicts override the automatic label.

## ⚠️ Confidence calibration caveat

**`confidence` is a reliable *ranking* signal, but its *absolute* calibration is unconfirmed.** The
formula was tuned on the human-reviewed clips, and those were drawn from a **worklist deliberately
enriched for hard/ambiguous cases** — not a random sample. So a given value should not be read as a
literal probability of correctness; use it to **order** utterances and choose a cutoff.

On the *hardest reviewed slice*, human accept-rate was ~91% at `confidence ≥ 0.8` and ~97% at
`≥ 0.9`. Because that slice is harder than the corpus average, true accept-rates on the full dataset
are expected to be **higher** — treat these as conservative lower bounds. The raw signals are
published so you can build your own filter instead of trusting the single scalar.

```python
high_quality  = lab.filter(lambda r: r["confidence"] >= 0.9)
verified_only = lab.filter(lambda r: r["review_status"] in ("verified", "edited"))
```

## Recommended use & limitations

- **Use:** weak/semi-supervised fine-tuning of ATC ASR on a high-confidence subset; bootstrapping;
  error analysis; callsign-recognition studies.
- **Mostly automatic labels** — apart from ~326 reviewed clips, transcripts are machine-generated and
  inherit ensemble errors (noise, overlap, accents, unusual phraseology).
- **Single-corpus coverage** — audio is from a small number of TartanAviation sites (KAGC, KBTP near
  Pittsburgh); not representative of all ATC.
- **No independent ground truth** — TartanAviation ships no reference transcripts; ADS-B callsign is
  the only external signal, and the reviewed clips were not randomly sampled.
- **Empty transcripts** — non-speech rows carry `transcript == ""`; filter for training text.

## Attribution & provenance

This dataset contains **derived text labels only**; **no audio is redistributed**. It is a 1:1 label
add-on to [`twangodev/tartanaviation-atc-adsb-utterances`](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb-utterances).

**Source corpus (CC BY 4.0):** audio and ADS-B derive from the **TartanAviation** dataset by the
AirLab at Carnegie Mellon University — Patrikar, J. et al., *TartanAviation: Image, Speech, and ADS-B
Trajectory Datasets for Terminal Airspace Operations*, CMU KiltHub,
DOI [10.1184/R1/25639599.v1](https://doi.org/10.1184/R1/25639599.v1), licensed CC BY 4.0. The
underlying ATC audio and ADS-B were lawfully recorded as public broadcasts with airport authorization.

**Labels** were generated by the `readback` ensemble pipeline using `nvidia/parakeet-tdt-0.6b-v2`
(CC-BY-4.0), `nvidia/canary-qwen-2.5b` (CC-BY-4.0), and `jlvdoorn/whisper-large-v3-atco2-asr`
(Apache-2.0). Model outputs are not restricted by the source models' licenses; the NVIDIA models are
credited per CC-BY-4.0.

Released under **CC BY 4.0**, consistent with the source corpus.

```bibtex
@misc{tartanaviation_atc_labels,
  title        = {TartanAviation ATC — Ensemble ASR Labels},
  author       = {twangodev},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/twangodev/tartanaviation-atc-labels}},
  note         = {Derived ensemble ASR labels for the TartanAviation ATC corpus; audio not included.}
}
```
