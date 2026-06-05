from __future__ import annotations

from readback.fuse.confidence import confidence_score


def test_without_advisory_averages_two_signals():
    assert confidence_score(0.4, 0.6, None) == 0.5


def test_with_advisory_folds_in_one_minus_disagreement():
    assert confidence_score(0.6, 0.6, 0.0) == (0.6 + 0.6 + 1.0) / 3


def test_higher_advisory_disagreement_lowers_confidence():
    assert confidence_score(0.6, 0.6, 0.9) < confidence_score(0.6, 0.6, 0.1)
