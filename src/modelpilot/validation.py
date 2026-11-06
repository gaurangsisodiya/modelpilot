"""Hold out part of the train split as validation, so tuning never looks at test."""

import numpy as np
import pandas as pd

from modelpilot.embeddings import query_table


def split_queries(
    results: pd.DataFrame, fraction: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split results by query into (fit, validation), taking `fraction` of each task's queries.

    All rows of one query land on the same side, so no query is in both.
    """
    queries = results.drop_duplicates("embedding_id")[["embedding_id", "task_name"]]
    held_out = queries.groupby("task_name").sample(frac=fraction, random_state=seed)["embedding_id"]
    in_val = results["embedding_id"].isin(held_out)
    return results[~in_val], results[in_val]


def subset_embeddings(embeddings: np.ndarray, results: pd.DataFrame, subset: pd.DataFrame) -> np.ndarray:
    """Rows of `embeddings` (aligned with query_table(results)) for the queries in `subset`, same order."""
    keep = query_table(results)["embedding_id"].isin(subset["embedding_id"]).to_numpy()
    return embeddings[keep]


def choose_default_lambda(sweep: pd.DataFrame, tolerance: float = 0.01) -> float:
    """Largest λ whose avg score is within `tolerance` of the best λ in `sweep` (columns lambda, avg_score)."""
    best = sweep["avg_score"].max()
    return float(sweep.loc[sweep["avg_score"] >= best - tolerance, "lambda"].max())


def wins_clearly(ci_low: float, cost_ratio: float, max_cost_ratio: float = 1.1) -> bool:
    """Challenger replaces the default only if its score gain's 95% CI is above 0 and it costs <= 10% more."""
    return ci_low > 0 and cost_ratio <= max_cost_ratio
