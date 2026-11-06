"""Learned router: predictions have the right shape and range, and routing follows them."""

import numpy as np
import pandas as pd

from modelpilot.learned import LearnedRouter


def clustered_results(n: int = 200) -> tuple[pd.DataFrame, np.ndarray]:
    """Queries near (1,0) are solved only by 'pricey'; queries near (0,1) by both models."""
    rng = np.random.default_rng(0)
    rows, vectors = [], []
    for q in range(n):
        hard = q % 2 == 0
        vectors.append(np.array([1.0, 0.0] if hard else [0.0, 1.0]) + rng.normal(0, 0.05, 2))
        rows.append({"embedding_id": q, "model_name": "cheap", "performance": 0.0 if hard else 1.0, "cost": 0.0001})
        rows.append({"embedding_id": q, "model_name": "pricey", "performance": 1.0, "cost": 0.01})
    return pd.DataFrame(rows), np.array(vectors)


def test_predictions_have_one_column_per_model_and_valid_range():
    results, vectors = clustered_results()
    score, cost = LearnedRouter.from_results(results, vectors, hidden=16).predict(vectors[:5])
    assert score.shape == cost.shape == (5, 2)
    assert score.min() >= 0 and score.max() <= 1 and cost.min() >= 0


def test_routes_hard_queries_to_pricey_and_easy_ones_to_cheap():
    results, vectors = clustered_results()
    router = LearnedRouter.from_results(results, vectors, hidden=16, alpha=1e-4)
    assert router.route(np.array([[1.0, 0.0], [0.0, 1.0]]), lam=0.01) == ["pricey", "cheap"]  # pricey costs 10 per 1k, cheap 0.1
