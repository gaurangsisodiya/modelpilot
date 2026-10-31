"""Load the xRouteBench routing data from Hugging Face."""

import pandas as pd
from datasets import load_dataset

DATASET = "ulab-ai/xRouteBench"


def load_candidates() -> pd.DataFrame:
    """Return the 18 candidate models with prices, service and API id."""
    return load_dataset(DATASET, "llm_candidates")["train"].to_pandas()


def load_split(split: str) -> pd.DataFrame:
    """Return one split ("train" or "test"): one row per (query, model) pair."""
    return load_dataset(DATASET, "llmrouter_generic")[split].to_pandas()
