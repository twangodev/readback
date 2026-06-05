from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from readback import store
from readback.fuse.confidence import confidence_score
from readback.review import iter_effective, load_review_log, merge_label

REVIEW_STATUS = {
    None: "auto",
    "accept": "verified",
    "edit": "edited",
    "reject": "rejected",
    "non_speech": "non_speech",
}

SCHEMA = pa.schema(
    [
        ("utterance_id", pa.string()),
        ("transcript", pa.string()),
        ("confidence", pa.float64()),
        ("review_status", pa.string()),
        ("agreement_score", pa.float64()),
        ("rover_confidence", pa.float64()),
        ("advisory_disagree", pa.float64()),
        ("n_models_agree", pa.int64()),
        ("callsign_matched", pa.bool_()),
        ("callsign_tail", pa.string()),
    ]
)


def _confidence(merged: dict) -> float:
    decision = merged["review_decision"]
    if decision in ("accept", "edit", "non_speech"):
        return 1.0
    if decision == "reject":
        return 0.0
    value = merged.get("confidence")
    if value is None:
        value = confidence_score(
            merged["agreement_score"],
            merged["rover_confidence"],
            merged["advisory_disagree"],
        )
    return value


def published_row(merged: dict) -> dict:
    return {
        "utterance_id": merged["utterance_id"],
        "transcript": merged["effective_transcript"],
        "confidence": _confidence(merged),
        "review_status": REVIEW_STATUS[merged["review_decision"]],
        "agreement_score": merged["agreement_score"],
        "rover_confidence": merged["rover_confidence"],
        "advisory_disagree": merged["advisory_disagree"],
        "n_models_agree": merged["n_models_agree"],
        "callsign_matched": merged["callsign_matched"],
        "callsign_tail": merged["callsign_tail"],
    }


def iter_dataset(run: Path) -> Iterator[dict]:
    """The published label dataset: one transcript and one confidence score per
    utterance, plus the raw agreement signals so consumers set their own
    threshold. Human verdicts override confidence (verified/edited/non_speech ->
    1.0, rejected -> 0.0). No tiers: a tier is whatever cutoff a consumer picks."""
    for merged in iter_effective(run):
        yield published_row(merged)


def write_dataset(run: Path, out: Path) -> int:
    return store.write_jsonl(out, iter_dataset(run))


CARD = Path(__file__).with_name("dataset_card.md")


def write_dataset_shards(run: Path, out_dir: Path) -> tuple[int, int]:
    """Write an upload-ready HuggingFace dataset folder: one parquet shard per
    source shard (shard-NNNNN.parquet at the root, mirroring the source layout),
    rows in the source's exact row order, plus a rendered README.md card. The
    labels are a 1:1 add-on to twangodev/tartanaviation-atc-adsb-utterances (join
    on utterance_id). Returns (n_shards, n_rows)."""
    log = load_review_log(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = rows = 0
    for path in sorted((run / "labels").glob("shard-*.jsonl")):
        records = [
            published_row(merge_label(label, log.current(label["utterance_id"])))
            for label in store.read_jsonl(path)
        ]
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        pq.write_table(table, out_dir / f"{path.stem}.parquet")
        shards += 1
        rows += len(records)
    card = (
        CARD.read_text()
        .replace("__N_ROWS__", f"{rows:,}")
        .replace("__N_SHARDS__", str(shards))
    )
    (out_dir / "README.md").write_text(card)
    return shards, rows
