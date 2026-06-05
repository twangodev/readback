from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from itertools import zip_longest

REVIEW_TIERS = ("silver", "tail")
CALIBRATION_TIERS = ("gold", "silver", "tail")
SILVER_CUTOFF = 0.3
BOUNDARY_HALFWIDTH = 0.1
HIGH_DISAGREE = 0.5
CALIBRATION_BINS = 10
DEFAULT_REVIEW_BUDGET = 1000
GOLD_VALIDATION_BUDGET = 50

_ITEM_KEYS = (
    "utterance_id",
    "tier",
    "rover_confidence",
    "agreement_score",
    "n_models_agree",
    "callsign_matched",
    "advisory_disagree",
)


def _priority(row: Mapping) -> tuple[float, float]:
    return -(row.get("advisory_disagree") or 0.0), row["agreement_score"]


def _item(row: Mapping, shard: int, reviewed: Collection[str]) -> dict:
    item = {key: row[key] for key in _ITEM_KEYS}
    item["shard"] = shard
    item["reviewed"] = row["utterance_id"] in reviewed
    return item


def build_worklist(
    labels_by_shard: Mapping[int, list[dict]],
    reviewed: Collection[str] = (),
) -> list[dict]:
    items: list[dict] = []
    for shard in sorted(labels_by_shard):
        rows = [row for row in labels_by_shard[shard] if row["tier"] in REVIEW_TIERS]
        rows.sort(key=_priority)
        items.extend(_item(row, shard, reviewed) for row in rows)
    return items


def _stable_rank(utterance_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(utterance_id.encode(), digest_size=8).digest()
    )


def _plan_item(
    row: Mapping, shard: int, reason: str, reviewed: Collection[str]
) -> dict:
    item = _item(row, shard, reviewed)
    item["reason"] = reason
    return item


def _calibration_stratum(
    pool: list[tuple[int, dict]], budget: int
) -> list[tuple[int, dict]]:
    if budget <= 0:
        return []
    bins: dict[int, list[tuple[int, dict]]] = {}
    for shard, row in pool:
        if row["tier"] not in CALIBRATION_TIERS:
            continue
        index = min(
            CALIBRATION_BINS - 1, max(0, int(row["agreement_score"] * CALIBRATION_BINS))
        )
        bins.setdefault(index, []).append((shard, row))
    queues = [
        sorted(entries, key=lambda sr: _stable_rank(sr[1]["utterance_id"]))
        for _, entries in sorted(bins.items())
    ]
    chosen: list[tuple[int, dict]] = []
    cursor = 0
    while len(chosen) < budget and any(queues):
        queue = queues[cursor % len(queues)]
        if queue:
            chosen.append(queue.pop(0))
        cursor += 1
    return chosen


def _boundary_stratum(
    pool: list[tuple[int, dict]], budget: int, exclude: Collection[str]
) -> list[tuple[int, dict]]:
    if budget <= 0:
        return []
    scored: list[tuple[float, float, int, dict]] = []
    for shard, row in pool:
        if row["utterance_id"] in exclude:
            continue
        disagree = row.get("advisory_disagree") or 0.0
        near_cutoff = abs(row["agreement_score"] - SILVER_CUTOFF) <= BOUNDARY_HALFWIDTH
        high_disagree = row["tier"] in ("gold", "silver") and disagree >= HIGH_DISAGREE
        if near_cutoff or high_disagree:
            scored.append(
                (abs(row["agreement_score"] - SILVER_CUTOFF), -disagree, shard, row)
            )
    scored.sort(
        key=lambda item: (item[0], item[1], _stable_rank(item[3]["utterance_id"]))
    )
    return [(shard, row) for _, _, shard, row in scored[:budget]]


def _gold_validation_stratum(
    pool: list[tuple[int, dict]], budget: int, reviewed: Collection[str]
) -> list[tuple[int, dict]]:
    gold = sorted(
        ((shard, row) for shard, row in pool if row["tier"] == "gold"),
        key=lambda sr: _stable_rank(sr[1]["utterance_id"]),
    )[:budget]
    return [(shard, row) for shard, row in gold if row["utterance_id"] not in reviewed]


def build_review_plan(
    labels_by_shard: Mapping[int, list[dict]],
    budget: int = DEFAULT_REVIEW_BUDGET,
    reviewed: Collection[str] = (),
) -> list[dict]:
    pool = [
        (shard, row)
        for shard in sorted(labels_by_shard)
        for row in labels_by_shard[shard]
        if row["tier"] != "non_speech"
    ]
    gold_validation = _gold_validation_stratum(pool, GOLD_VALIDATION_BUDGET, reviewed)
    gold_ids = {row["utterance_id"] for _, row in gold_validation}
    calibration = [
        (shard, row)
        for shard, row in _calibration_stratum(pool, budget // 2)
        if row["utterance_id"] not in gold_ids
    ]
    chosen_ids = gold_ids | {row["utterance_id"] for _, row in calibration}
    boundary = _boundary_stratum(pool, budget - len(calibration), chosen_ids)
    gold_items = [
        _plan_item(row, shard, "gold-validation", reviewed)
        for shard, row in gold_validation
    ]
    calibration_items = [
        _plan_item(row, shard, "calibration", reviewed) for shard, row in calibration
    ]
    boundary_items = [
        _plan_item(row, shard, "boundary", reviewed) for shard, row in boundary
    ]
    interleaved: list[dict] = []
    for boundary_item, calibration_item in zip_longest(
        boundary_items, calibration_items
    ):
        if boundary_item is not None:
            interleaved.append(boundary_item)
        if calibration_item is not None:
            interleaved.append(calibration_item)
    return gold_items + interleaved


def queue_stats(
    labels_by_shard: Mapping[int, list[dict]],
    reviewed: Collection[str] = (),
) -> dict:
    by_tier = dict.fromkeys(REVIEW_TIERS, 0)
    total = 0
    done = 0
    for rows in labels_by_shard.values():
        for row in rows:
            if row["tier"] not in REVIEW_TIERS:
                continue
            total += 1
            by_tier[row["tier"]] += 1
            if row["utterance_id"] in reviewed:
                done += 1
    return {
        "total": total,
        "reviewed": done,
        "remaining": total - done,
        "by_tier": by_tier,
    }
