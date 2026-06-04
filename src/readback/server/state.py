from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from readback import store
from readback.pipeline.layout import reviews_path
from readback.review import ReviewLog
from readback.server.context import build_context
from readback.server.shard_cache import ShardAudio
from readback.server.worklist import (
    DEFAULT_REVIEW_BUDGET,
    build_review_plan,
    build_worklist,
    queue_stats,
)

CONVERSATION_SPAN = 3


def _shard_index(path: Path) -> int:
    return int(path.stem.removeprefix("shard-"))


class StudioState:
    def __init__(
        self, run: Path, audio: ShardAudio, reviewer: str | None = None
    ) -> None:
        self._run = run
        self._audio = audio
        self._reviewer = reviewer

        self._labels_by_shard: dict[int, list[dict]] = {}
        self._shard_of: dict[str, int] = {}
        self._label_of: dict[str, dict] = {}
        for path in sorted((run / "labels").glob("shard-*.jsonl")):
            shard = _shard_index(path)
            rows = list(store.read_jsonl(path))
            self._labels_by_shard[shard] = rows
            for row in rows:
                self._shard_of[row["utterance_id"]] = shard
                self._label_of[row["utterance_id"]] = row

        self._meta_of: dict[str, dict] = {}
        for path in sorted((run / "meta").glob("shard-*.jsonl")):
            for row in store.read_jsonl(path):
                self._meta_of[row["utterance_id"]] = row

        self._reviews = ReviewLog.from_rows(
            row
            for path in sorted((run / "reviews").glob("shard-*.jsonl"))
            for row in store.read_jsonl_recoverable(path)
        )

    def _reviewed(self) -> set[str]:
        return self._reviews.reviewed_ids()

    def queue(self) -> dict:
        reviewed = self._reviewed()
        items = build_worklist(self._labels_by_shard, reviewed)
        return {"items": items, "total": len(items), "reviewed": len(reviewed)}

    def plan(self, budget: int = DEFAULT_REVIEW_BUDGET) -> dict:
        reviewed = self._reviewed()
        items = build_review_plan(self._labels_by_shard, budget, reviewed)
        done = sum(1 for item in items if item["reviewed"])
        return {"items": items, "total": len(items), "reviewed": done}

    def stats(self) -> dict:
        return queue_stats(self._labels_by_shard, self._reviewed())

    def context(self, utterance_id: str) -> dict | None:
        if utterance_id not in self._label_of or utterance_id not in self._meta_of:
            return None
        context = build_context(
            self._label_of[utterance_id],
            self._meta_of[utterance_id],
            self._reviews.current(utterance_id),
        )
        extra = self._audio.context(self._shard_of[utterance_id], utterance_id)
        if extra:
            context.update(extra)
        context["conversation"] = self._conversation(utterance_id)
        return context

    def _conversation(self, utterance_id: str) -> list[dict]:
        prefix, _, index = utterance_id.rpartition("/")
        if not index.isdigit():
            return []
        position = int(index)
        rows: list[dict] = []
        for offset in range(-CONVERSATION_SPAN, CONVERSATION_SPAN + 1):
            neighbor = f"{prefix}/{position + offset}"
            label = self._label_of.get(neighbor)
            if label is None:
                continue
            rows.append(
                {
                    "utterance_id": neighbor,
                    "transcript": label["transcript"],
                    "tier": label["tier"],
                    "current": neighbor == utterance_id,
                    "start": self._audio.clip_start(self._shard_of[neighbor], neighbor),
                }
            )
        return rows

    def audio(self, utterance_id: str) -> bytes:
        if utterance_id not in self._shard_of:
            raise KeyError(utterance_id)
        return self._audio.wav_bytes(self._shard_of[utterance_id], utterance_id)

    def record(self, utterance_id: str, payload: dict) -> dict:
        shard = self._shard_of[utterance_id]
        event = self._reviews.record(
            utterance_id,
            payload["decision"],
            payload["transcript"],
            payload["base_hyp"],
            shard,
            self._reviewer,
        )
        store.append_jsonl(reviews_path(self._run, shard), asdict(event))
        return asdict(event)

    def undo(self, utterance_id: str) -> dict | None:
        shard = self._shard_of[utterance_id]
        event = self._reviews.undo(utterance_id, shard, self._reviewer)
        if event is not None:
            store.append_jsonl(reviews_path(self._run, shard), asdict(event))
        restored = self._reviews.current(utterance_id)
        return asdict(restored) if restored is not None else None
