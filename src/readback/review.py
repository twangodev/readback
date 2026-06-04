from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path

from readback import store

SET = "set"
UNDO = "undo"


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    utterance_id: str
    action: str
    seq: int
    ts: float
    shard: int
    reviewer: str | None = None
    decision: str = ""
    transcript: str = ""
    base_hyp: str = ""

    @classmethod
    def from_row(cls, row: dict) -> ReviewEvent:
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in row.items() if key in known})


class ReviewLog:
    def __init__(self) -> None:
        self._stacks: dict[str, list[ReviewEvent]] = {}
        self._log: list[ReviewEvent] = []
        self._next_seq = 0

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> ReviewLog:
        log = cls()
        events = sorted(
            (ReviewEvent.from_row(row) for row in rows), key=lambda event: event.seq
        )
        for event in events:
            log._apply(event)
        return log

    def _apply(self, event: ReviewEvent) -> None:
        self._log.append(event)
        self._next_seq = max(self._next_seq, event.seq + 1)
        stack = self._stacks.setdefault(event.utterance_id, [])
        if event.action == SET:
            stack.append(event)
        elif stack:
            stack.pop()

    def record(
        self,
        utterance_id: str,
        decision: str,
        transcript: str,
        base_hyp: str,
        shard: int,
        reviewer: str | None = None,
    ) -> ReviewEvent:
        event = ReviewEvent(
            utterance_id=utterance_id,
            action=SET,
            seq=self._next_seq,
            ts=time.time(),
            shard=shard,
            reviewer=reviewer,
            decision=decision,
            transcript=transcript,
            base_hyp=base_hyp,
        )
        self._apply(event)
        return event

    def undo(
        self, utterance_id: str, shard: int, reviewer: str | None = None
    ) -> ReviewEvent | None:
        if not self._stacks.get(utterance_id):
            return None
        event = ReviewEvent(
            utterance_id=utterance_id,
            action=UNDO,
            seq=self._next_seq,
            ts=time.time(),
            shard=shard,
            reviewer=reviewer,
        )
        self._apply(event)
        return event

    def current(self, utterance_id: str) -> ReviewEvent | None:
        stack = self._stacks.get(utterance_id)
        return stack[-1] if stack else None

    def reviewed_ids(self) -> set[str]:
        return {uid for uid, stack in self._stacks.items() if stack}

    def max_seq(self) -> int:
        return self._next_seq


def merge_label(label: dict, current: ReviewEvent | None) -> dict:
    if current is None:
        return {
            **label,
            "reviewed": False,
            "review_decision": None,
            "effective_transcript": label["transcript"],
            "source": "model",
        }
    if current.decision == "edit":
        transcript, source = current.transcript, "human"
    elif current.decision == "accept":
        transcript, source = label["transcript"], "human-confirmed"
    else:
        transcript, source = "", "human"
    return {
        **label,
        "reviewed": True,
        "review_decision": current.decision,
        "effective_transcript": transcript,
        "source": source,
    }


def load_review_log(run: Path) -> ReviewLog:
    directory = run / "reviews"
    rows = [
        row
        for path in sorted(directory.glob("shard-*.jsonl"))
        for row in store.read_jsonl_recoverable(path)
    ]
    return ReviewLog.from_rows(rows)


def iter_effective(run: Path) -> Iterator[dict]:
    log = load_review_log(run)
    for path in sorted((run / "labels").glob("shard-*.jsonl")):
        for label in store.read_jsonl(path):
            yield merge_label(label, log.current(label["utterance_id"]))
