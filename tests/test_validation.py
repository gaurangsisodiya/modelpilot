"""Validation split and the rules that pick defaults from it."""

import numpy as np
import pandas as pd

from modelpilot.validation import choose_default_lambda, split_queries, subset_embeddings, wins_clearly


def two_task_results() -> pd.DataFrame:
    """20 queries (10 per task) x 2 models."""
    rows = [
        {"embedding_id": q, "query": f"q{q}", "task_name": "a" if q < 10 else "b", "model_name": m, "performance": 1.0}
        for q in range(20)
        for m in ["x", "y"]
    ]
    return pd.DataFrame(rows)


def test_split_keeps_each_query_on_one_side_and_samples_every_task():
    fit, val = split_queries(two_task_results(), fraction=0.2, seed=0)
    assert set(fit["embedding_id"]).isdisjoint(val["embedding_id"])
    assert val.drop_duplicates("embedding_id")["task_name"].value_counts().to_dict() == {"a": 2, "b": 2}
    assert len(fit) + len(val) == 40


def test_split_is_reproducible():
    a, b = split_queries(two_task_results(), seed=3)[1], split_queries(two_task_results(), seed=3)[1]
    assert a["embedding_id"].tolist() == b["embedding_id"].tolist()


def test_subset_embeddings_keeps_rows_of_chosen_queries_in_order():
    results = two_task_results()
    embeddings = np.arange(20, dtype=float)[:, None]  # row i belongs to embedding_id i
    _, val = split_queries(results, seed=0)
    assert subset_embeddings(embeddings, results, val)[:, 0].tolist() == sorted(val["embedding_id"].unique())


def test_default_lambda_is_cheapest_within_tolerance():
    sweep = pd.DataFrame({"lambda": [0.0, 1.0, 3.0], "avg_score": [0.760, 0.752, 0.740]})
    assert choose_default_lambda(sweep, tolerance=0.01) == 1.0
    assert choose_default_lambda(sweep, tolerance=0.03) == 3.0


def test_wins_clearly_needs_significance_and_similar_cost():
    assert wins_clearly(ci_low=0.001, cost_ratio=1.05)
    assert not wins_clearly(ci_low=-0.001, cost_ratio=0.5)  # not significant
    assert not wins_clearly(ci_low=0.01, cost_ratio=1.28)   # too expensive
