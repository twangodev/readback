from __future__ import annotations

from readback.fuse import agreement


def test_unanimous_full_cluster():
    result = agreement(["cleared to land", "cleared to land", "cleared to land"])
    assert result.n_agree == 3
    assert result.score == 1.0


def test_two_of_three_cluster():
    result = agreement(["cleared to land", "cleared to land", "cleared for takeoff"])
    assert result.n_agree == 2


def test_all_disagree():
    result = agreement(["alpha", "bravo charlie", "delta echo foxtrot golf"])
    assert result.n_agree == 1
