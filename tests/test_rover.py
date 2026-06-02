from __future__ import annotations

from readback.fuse import rover


def test_unanimous_returns_that_text_at_full_confidence():
    result = rover(["cleared to land", "cleared to land", "cleared to land"])
    assert result.text == "cleared to land"
    assert result.confidence == 1.0


def test_word_level_majority_recovers_callsign():
    result = rover(
        [
            "cessna one seven three seven",
            "skyhawk one seven three seven",
            "cessna one seven three seven",
        ]
    )
    assert result.text == "cessna one seven three seven"


def test_recovers_truth_from_errors_in_different_slots():
    result = rover(
        [
            "turn left heading two one zero",
            "turn left heading two one five",
            "turn right heading two one zero",
        ]
    )
    assert result.text == "turn left heading two one zero"


def test_majority_insertion_is_kept():
    result = rover(["taxi to bravo", "taxi bravo", "taxi to bravo"])
    assert result.text == "taxi to bravo"


def test_minority_insertion_is_dropped():
    result = rover(["taxi bravo", "taxi to bravo", "taxi bravo"])
    assert result.text == "taxi bravo"


def test_number_forms_are_canonicalized_before_voting():
    result = rover(["descend two five zero", "descend 250", "descend two five zero"])
    assert result.text == "descend two five zero"
    assert result.confidence == 1.0


def test_all_empty_yields_empty():
    assert rover(["", "", ""]).text == ""
