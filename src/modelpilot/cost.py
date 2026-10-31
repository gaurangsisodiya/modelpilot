"""Dollar cost of running a query on a model."""

import pandas as pd

TOKENS_PER_PRICE_UNIT = 1_000_000


def query_cost(
    input_tokens: float,
    output_tokens: float,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    """Dollar cost of one call, given token counts and per-1M-token prices."""
    return (
        input_tokens * input_price_per_1m + output_tokens * output_price_per_1m
    ) / TOKENS_PER_PRICE_UNIT


def add_cost(results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Join model prices onto results and add a `cost` column (dollars per row)."""
    prices = candidates[["model_name", "input_price_per_1m", "output_price_per_1m"]]
    merged = results.merge(prices, on="model_name", how="left", validate="many_to_one")
    if merged["input_price_per_1m"].isna().any():
        raise ValueError("Some results have a model_name with no price in candidates")
    merged["cost"] = query_cost(
        merged["input_tokens"],
        merged["output_tokens"],
        merged["input_price_per_1m"],
        merged["output_price_per_1m"],
    )
    return merged
