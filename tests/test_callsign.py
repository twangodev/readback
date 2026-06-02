from __future__ import annotations

from readback.callsign import snap, verify


def test_matches_spoken_n_number():
    result = verify("november one seven two sierra papa say intentions", ["N172SP"])
    assert result.matched
    assert result.tail == "N172SP"
    assert result.n_tails == 1


def test_matches_on_digit_suffix_when_carrier_word_missed():
    result = verify("traffic four nine nine six turning crosswind", ["RPA4996"])
    assert result.matched


def test_no_match_when_no_present_callsign_spoken():
    result = verify("cleared to land runway two eight", ["DAL1234", "N172SP"])
    assert not result.matched


def test_no_tails_is_unmatched():
    result = verify("november one seven two sierra papa", [])
    assert not result.matched
    assert result.n_tails == 0


def test_snap_leaves_perfect_match_unchanged():
    result = snap("november one seven two sierra papa descend", ["N172SP"])
    assert not result.snapped
    assert result.text == "november one seven two sierra papa descend"


def test_snap_no_match_unchanged():
    result = snap("cleared to land runway two eight", ["DAL1234"])
    assert not result.snapped


def test_snap_corrects_near_miss_callsign():
    result = snap("november one seven two sierra hotel descend", ["N172SP"])
    assert result.match.tail == "N172SP"
    assert "november one seven two sierra papa" in result.text
