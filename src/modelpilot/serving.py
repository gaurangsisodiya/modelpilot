"""Decide which model actually serves a request, given what NVIDIA has available."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ServedModel:
    """The model a request is sent to, and whether it replaced the routed choice."""

    model_name: str
    api_model_id: str
    fallback: bool


def fallback_model(candidates: pd.DataFrame, summary: pd.DataFrame, available_ids: set[str]) -> str:
    """Available candidate with the highest train avg score (ties: lower avg cost).

    `summary` is model_summary(train); `candidates` has model_name and api_model_id.
    """
    listed = candidates.loc[candidates["api_model_id"].isin(available_ids), "model_name"]
    ranked = summary.loc[summary.index.isin(listed)].sort_values(
        ["avg_score", "avg_cost"], ascending=[False, True]
    )
    if ranked.empty:
        raise RuntimeError("None of the candidate models are available on NVIDIA")
    return ranked.index[0]


def choose_served_model(
    routed_model: str, candidates: pd.DataFrame, summary: pd.DataFrame, available_ids: set[str]
) -> ServedModel:
    """Use the routed model if NVIDIA serves it; otherwise the best available fallback."""
    api_ids = candidates.set_index("model_name")["api_model_id"]
    if api_ids[routed_model] in available_ids:
        return ServedModel(routed_model, api_ids[routed_model], fallback=False)
    fallback = fallback_model(candidates, summary, available_ids)
    return ServedModel(fallback, api_ids[fallback], fallback=True)
