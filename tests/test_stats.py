import numpy as np
import pytest

from modelpilot.stats import paired_bootstrap


def test_identical_strategies_have_zero_difference():
    a = np.array([1, 0, 1, 1, 0], dtype=float)
    result = paired_bootstrap(a, a)
    assert (result.diff, result.low, result.high) == (0.0, 0.0, 0.0)


def test_difference_is_mean_of_a_minus_b_and_inside_interval():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 2000).astype(float)
    b = np.where(rng.random(2000) < 0.9, a, 1 - a)  # b agrees with a on ~90% of queries
    result = paired_bootstrap(a, b, n_resamples=500)
    assert result.diff == pytest.approx((a - b).mean())
    assert result.low <= result.diff <= result.high


def test_clear_win_gives_interval_above_zero():
    a = np.ones(500)
    b = np.r_[np.ones(400), np.zeros(100)]  # a is right on 100 more queries
    result = paired_bootstrap(a, b)
    assert result.diff == pytest.approx(0.2)
    assert result.low > 0


def test_same_seed_is_reproducible():
    a, b = np.r_[np.ones(50), np.zeros(50)], np.r_[np.zeros(30), np.ones(70)]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_rejects_unpaired_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(3), np.ones(4))
