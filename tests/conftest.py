"""Tiny hand-made datasets shared by the tests."""

import pandas as pd
import pytest


@pytest.fixture
def candidates() -> pd.DataFrame:
    """Three models: cheap, mid and pricey (prices in $ per 1M tokens)."""
    return pd.DataFrame(
        {
            "model_name": ["cheap", "mid", "pricey"],
            "input_price_per_1m": [0.1, 0.5, 2.0],
            "output_price_per_1m": [0.1, 1.0, 4.0],
            "service": ["NVIDIA", "Together", "Together"],
            "api_model_id": ["v/cheap", "v/mid", "v/pricey"],
        }
    )


@pytest.fixture
def results() -> pd.DataFrame:
    """Three queries x three models, already with a `cost` column.

    Query 0: everyone right. Query 1: only pricey right. Query 2: nobody right.
    """
    rows = []
    perf = {0: [1, 1, 1], 1: [0, 0, 1], 2: [0, 0, 0]}
    for query_id, scores in perf.items():
        for model, score, cost in zip(["cheap", "mid", "pricey"], scores, [0.001, 0.01, 0.1]):
            rows.append(
                {"embedding_id": query_id, "model_name": model, "performance": float(score), "cost": cost}
            )
    return pd.DataFrame(rows)
