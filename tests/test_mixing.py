import numpy as np
import pytest

from modelpilot.mixing import mix_fraction, mixed_scores


def test_fraction_matches_target_cost():
    p = mix_fraction(target_cost=0.5, cheap_cost=0.1, best_cost=1.7)
    assert p == pytest.approx(0.25)
    assert (1 - p) * 0.1 + p * 1.7 == pytest.approx(0.5)


def test_fraction_at_the_ends():
    assert mix_fraction(0.1, 0.1, 1.7) == 0.0
    assert mix_fraction(1.7, 0.1, 1.7) == 1.0


def test_fraction_is_clipped_outside_the_range():
    assert mix_fraction(0.05, 0.1, 1.7) == 0.0
    assert mix_fraction(2.0, 0.1, 1.7) == 1.0


def test_fraction_rejects_best_not_more_expensive():
    with pytest.raises(ValueError):
        mix_fraction(0.5, 1.0, 1.0)


def test_mixed_scores_are_the_weighted_average_per_query():
    cheap, best = np.array([0.0, 1.0, 0.0, 0.5]), np.array([1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(mixed_scores(cheap, best, 0.25), [0.25, 1.0, 0.0, 0.625])


def test_mixed_scores_endpoints_equal_the_fixed_models():
    cheap, best = np.array([0.0, 1.0, 0.3]), np.array([1.0, 0.0, 0.9])
    np.testing.assert_allclose(mixed_scores(cheap, best, 0.0), cheap)
    np.testing.assert_allclose(mixed_scores(cheap, best, 1.0), best)


def test_mixed_scores_mean_equals_simulated_random_mix():
    rng = np.random.default_rng(0)
    cheap, best = rng.integers(0, 2, 5000).astype(float), rng.integers(0, 2, 5000).astype(float)
    p = 0.3
    draws = [np.where(rng.random(5000) < p, best, cheap).mean() for _ in range(200)]
    assert mixed_scores(cheap, best, p).mean() == pytest.approx(np.mean(draws), abs=0.002)


def test_mixed_scores_reject_bad_input():
    with pytest.raises(ValueError):
        mixed_scores(np.ones(3), np.ones(4), 0.5)
    with pytest.raises(ValueError):
        mixed_scores(np.ones(3), np.ones(3), 1.5)
