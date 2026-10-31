"""kNN neighbour search, weighting and the routing decision."""

import numpy as np
import pytest

from modelpilot.router import COST_SCALE, KNNRouter, nearest_neighbours, pick_models, weighted_mean

MODELS = ["cheap", "mid", "pricey"]


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_nearest_neighbours_finds_most_similar_rows():
    train = np.stack([unit(1, 0), unit(0, 1), unit(1, 1)])
    indices, similarity = nearest_neighbours(unit(1, 0.1)[None, :], train, k=2)
    assert set(indices[0]) == {0, 2}
    assert similarity.shape == (1, 2)
    assert similarity.max() == pytest.approx(float(unit(1, 0.1) @ unit(1, 0)))


def test_weighted_mean_favours_closer_neighbours():
    values = np.array([[[1.0], [0.0]]])  # 1 query, 2 neighbours, 1 model
    weights = np.array([[0.9, 0.1]])
    assert weighted_mean(values, weights)[0, 0] == pytest.approx(0.9)


def test_weighted_mean_survives_non_positive_similarities():
    values = np.array([[[1.0], [0.0]]])
    result = weighted_mean(values, np.array([[0.0, -0.5]]))
    assert np.isfinite(result).all()


def test_pick_models_lambda_zero_takes_best_score():
    score = np.array([[0.5, 0.7, 0.9]])
    cost = np.array([[0.01, 0.1, 1.0]])
    assert pick_models(score, cost, 0.0, MODELS) == ["pricey"]


def test_pick_models_high_lambda_takes_cheaper_model():
    score = np.array([[0.5, 0.7, 0.9]])
    cost = np.array([[0.01, 0.1, 1.0]])
    assert pick_models(score, cost, 1.0, MODELS) == ["mid"]   # 0.6 beats 0.49 and -0.1
    assert pick_models(score, cost, 10.0, MODELS) == ["cheap"]


def test_pick_models_breaks_ties_by_lower_cost():
    score = np.array([[0.8, 0.8, 0.8]])
    cost = np.array([[0.3, 0.1, 0.2]])
    assert pick_models(score, cost, 0.0, MODELS) == ["mid"]


def test_router_predicts_from_neighbours(results):
    # Queries 0 and 1 sit close together; query 2 is far away.
    train_embeddings = np.stack([unit(1, 0), unit(1, 0.05), unit(0, 1)])
    router = KNNRouter.from_results(results, train_embeddings, k=2)
    score, cost = router.predict(unit(1, 0.02)[None, :])

    assert router.models == MODELS
    # Neighbours are queries 0 and 1: pricey right on both, cheap and mid on one.
    assert score[0, MODELS.index("pricey")] == pytest.approx(1.0)
    assert score[0, MODELS.index("cheap")] == pytest.approx(0.5, abs=0.05)
    assert cost[0, MODELS.index("cheap")] == pytest.approx(0.001 * COST_SCALE)
    assert router.route(unit(1, 0.02)[None, :], lam=0.0) == ["pricey"]
