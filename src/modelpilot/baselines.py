"""Simple routing strategies to compare the router against.

Each strategy returns a Series mapping embedding_id -> chosen model_name.
"""

import pandas as pd


def cheapest_model(summary: pd.DataFrame) -> str:
    """Model with the lowest average cost per query (summary from model_summary)."""
    return summary.sort_values(["avg_cost", "avg_score"], ascending=[True, False]).index[0]


def best_model(summary: pd.DataFrame) -> str:
    """Model with the highest average score; ties broken by lowest cost."""
    return summary.sort_values(["avg_score", "avg_cost"], ascending=[False, True]).index[0]


def always(results: pd.DataFrame, model_name: str) -> pd.Series:
    """Route every query in `results` to the same model."""
    query_ids = results["embedding_id"].unique()
    return pd.Series(model_name, index=query_ids)


def oracle(results: pd.DataFrame) -> pd.Series:
    """Per query, the model with the highest performance; ties broken by lowest cost.

    Uses the true scores, so it is an upper bound no real router can beat.
    """
    ranked = results.sort_values(
        ["embedding_id", "performance", "cost"], ascending=[True, False, True]
    )
    best = ranked.drop_duplicates("embedding_id", keep="first")
    return best.set_index("embedding_id")["model_name"]
