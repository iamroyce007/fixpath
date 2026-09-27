import pytest

from app.scoring import compute_score


def test_range_and_determinism():
    s = compute_score(0.8, 0.9, 2)
    assert 0 <= s <= 1
    assert s == compute_score(0.8, 0.9, 2)


def test_fewer_fixes_scores_higher():
    assert compute_score(0.7, 0.7, 0) > compute_score(0.7, 0.7, 5)


def test_clamps_inputs():
    assert compute_score(5, 5, -3) == 1.0
    assert compute_score(-1, -1, 10**6) == pytest.approx(0.0, abs=1e-3)


def test_bad_weights():
    with pytest.raises(ValueError):
        compute_score(0.5, 0.5, 0, weights=(0, 0, 0))
