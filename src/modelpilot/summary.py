"""Per-model summary statistics."""

import pandas as pd


def model_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Average score and average cost per query for each model, best score first.

    Expects one row per (query, model) with `performance` and `cost` columns.
    """
    summary = results.groupby("model_name").agg(
        avg_score=("performance", "mean"),
        avg_cost=("cost", "mean"),
        queries=("embedding_id", "nunique"),
    )
    return summary.sort_values(["avg_score", "avg_cost"], ascending=[False, True])
