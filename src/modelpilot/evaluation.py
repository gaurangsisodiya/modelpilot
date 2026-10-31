"""Score a routing strategy: which model it picked for each query, and what that cost."""

import pandas as pd


def chosen_rows(results: pd.DataFrame, choices: pd.Series) -> pd.DataFrame:
    """Return the result row of the chosen model for each query.

    `choices` maps embedding_id -> model_name (one entry per query).
    """
    picks = choices.rename("model_name").rename_axis("embedding_id").reset_index()
    rows = picks.merge(results, on=["embedding_id", "model_name"], how="left", validate="one_to_one")
    if rows["performance"].isna().any():
        raise ValueError("Some chosen (query, model) pairs are missing from results")
    return rows


def evaluate(results: pd.DataFrame, choices: pd.Series) -> dict[str, float]:
    """Avg score and total cost of a strategy over all queries in `results`."""
    rows = chosen_rows(results, choices)
    return {
        "avg_score": rows["performance"].mean(),
        "total_cost": rows["cost"].sum(),
        "queries": len(rows),
    }
