from __future__ import annotations

from readback.review import ReviewLog, merge_label


def test_record_then_current_returns_event():
    log = ReviewLog()
    log.record("kagc/1", "edit", "cleared to land", "cleared to taxi", 46)
    current = log.current("kagc/1")
    assert current.decision == "edit"
    assert current.transcript == "cleared to land"


def test_history_is_preserved_not_overwritten():
    log = ReviewLog()
    log.record("kagc/1", "edit", "first", "raw", 46)
    log.record("kagc/1", "edit", "second", "raw", 46)
    assert log.current("kagc/1").transcript == "second"
    assert log.max_seq() == 2


def test_undo_restores_prior_event():
    log = ReviewLog()
    log.record("kagc/1", "edit", "first", "raw", 46)
    log.record("kagc/1", "edit", "second", "raw", 46)
    log.undo("kagc/1", 46)
    assert log.current("kagc/1").transcript == "first"


def test_undo_of_only_event_clears_current():
    log = ReviewLog()
    log.record("kagc/1", "accept", "x", "y", 46)
    log.undo("kagc/1", 46)
    assert log.current("kagc/1") is None
    assert "kagc/1" not in log.reviewed_ids()


def test_undo_with_nothing_to_undo_returns_none():
    log = ReviewLog()
    assert log.undo("kagc/1", 46) is None


def test_from_rows_replays_in_seq_order_regardless_of_input_order():
    rows = [
        {
            "utterance_id": "kagc/1",
            "action": "undo",
            "seq": 2,
            "ts": 3.0,
            "shard": 46,
        },
        {
            "utterance_id": "kagc/1",
            "action": "set",
            "seq": 0,
            "ts": 1.0,
            "shard": 46,
            "decision": "edit",
            "transcript": "first",
            "base_hyp": "raw",
        },
        {
            "utterance_id": "kagc/1",
            "action": "set",
            "seq": 1,
            "ts": 2.0,
            "shard": 46,
            "decision": "edit",
            "transcript": "second",
            "base_hyp": "raw",
        },
    ]
    log = ReviewLog.from_rows(rows)
    assert log.current("kagc/1").transcript == "first"
    assert log.max_seq() == 3


def test_reviewed_ids_tracks_current_stacks():
    log = ReviewLog()
    log.record("kagc/1", "accept", "x", "y", 46)
    log.record("kagc/2", "edit", "z", "y", 12)
    log.undo("kagc/1", 46)
    assert log.reviewed_ids() == {"kagc/2"}


def test_merge_label_edit_uses_human_transcript():
    label = {"utterance_id": "kagc/1", "transcript": "model text"}
    event = ReviewLog().record("kagc/1", "edit", "human text", "model text", 46)
    merged = merge_label(label, event)
    assert merged["effective_transcript"] == "human text"
    assert merged["source"] == "human"
    assert merged["reviewed"] is True


def test_merge_label_accept_keeps_model_transcript():
    label = {"utterance_id": "kagc/1", "transcript": "model text"}
    event = ReviewLog().record("kagc/1", "accept", "model text", "model text", 46)
    merged = merge_label(label, event)
    assert merged["effective_transcript"] == "model text"
    assert merged["source"] == "human-confirmed"


def test_merge_label_reject_blanks_transcript():
    label = {"utterance_id": "kagc/1", "transcript": "model text"}
    event = ReviewLog().record("kagc/1", "reject", "", "model text", 46)
    merged = merge_label(label, event)
    assert merged["effective_transcript"] == ""


def test_merge_label_unreviewed_is_model_source():
    label = {"utterance_id": "kagc/1", "transcript": "model text"}
    merged = merge_label(label, None)
    assert merged["reviewed"] is False
    assert merged["effective_transcript"] == "model text"
    assert merged["source"] == "model"
