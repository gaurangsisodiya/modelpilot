"""Load the xRouteBench routing data from Hugging Face."""

import pandas as pd
from datasets import load_dataset

DATASET = "ulab-ai/xRouteBench"
# The dataset commit every committed result was computed from; pinned so a dataset update can't change them.
REVISION = "ea4b6e1b29d9a734f55f0a637baf326bad6aa681"


def load_candidates() -> pd.DataFrame:
    """Return the 18 candidate models with prices, service and API id."""
    return load_dataset(DATASET, "llm_candidates", revision=REVISION)["train"].to_pandas()


def load_split(split: str) -> pd.DataFrame:
    """Return one split ("train" or "test"): one row per (query, model) pair."""
    return load_dataset(DATASET, "llmrouter_generic", revision=REVISION)[split].to_pandas()
