"""Simple routing strategies to compare the router against.

Each strategy returns a Series mapping embedding_id -> chosen model_name.
"""

import pandas as pd

from modelpilot.evaluation import evaluate
from modelpilot.summary import model_summary


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


def evaluate_baselines(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Score always-cheapest, always-best and oracle on test (both splits need `cost`).

    Fixed models are chosen from train stats so test data never influences the choice.
    Returns the results table and the oracle's choices.
    """
    train_summary = model_summary(train)
    cheapest = cheapest_model(train_summary)
    best = best_model(train_summary)
    oracle_choices = oracle(test)
    strategies = {
        f"always-cheapest ({cheapest})": always(test, cheapest),
        f"always-best ({best})": always(test, best),
        "oracle": oracle_choices,
    }
    rows = [{"strategy": name, **evaluate(test, choices)} for name, choices in strategies.items()]
    return pd.DataFrame(rows), oracle_choices
