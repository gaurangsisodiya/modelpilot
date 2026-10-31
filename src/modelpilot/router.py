"""kNN router: predict each model's score and cost from similar train queries."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Costs are compared in dollars per 1,000 queries so lambda stays in a readable range.
COST_SCALE = 1000


def per_query_matrix(results: pd.DataFrame, column: str, models: list[str]) -> np.ndarray:
    """Pivot results into shape (queries, models): row = embedding_id order, column = models order."""
    table = results.pivot(index="embedding_id", columns="model_name", values=column)
    return table.sort_index()[models].to_numpy()


def nearest_neighbours(queries: np.ndarray, train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and similarities of the k most similar train rows for each query.

    Both inputs must be unit-length vectors, so the dot product is the cosine similarity.
    Returns two arrays of shape (queries, k).
    """
    similarity = queries @ train.T
    indices = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    return indices, np.take_along_axis(similarity, indices, axis=1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mean over the neighbour axis, weighting each neighbour by its similarity.

    `values` has shape (queries, k, models); `weights` has shape (queries, k).
    Negative similarities are clipped to a tiny positive weight so the sum is never zero.
    """
    weights = np.clip(weights, 1e-6, None)[:, :, None]
    return (values * weights).sum(axis=1) / weights.sum(axis=1)


def pick_models(
    predicted_score: np.ndarray, predicted_cost: np.ndarray, lam: float, models: list[str]
) -> list[str]:
    """For each query, the model maximising predicted_score - lam * predicted_cost.

    Ties (equal to 9 decimal places) go to the model with lower predicted cost.
    Both arrays have shape (queries, models); cost is in dollars per 1,000 queries.
    """
    utility = np.round(predicted_score - lam * predicted_cost, 9)
    # lexsort sorts by the last key first: highest utility, then lowest cost.
    order = np.lexsort((predicted_cost, -utility), axis=1)
    return [models[i] for i in order[:, 0]]


@dataclass
class KNNRouter:
    """Remembers train embeddings plus each model's score and cost on every train query."""

    models: list[str]
    train_embeddings: np.ndarray
    score_matrix: np.ndarray
    cost_matrix: np.ndarray
    k: int = 10

    @classmethod
    def from_results(cls, train: pd.DataFrame, train_embeddings: np.ndarray, k: int = 10) -> "KNNRouter":
        """Build a router from train results (with `cost`) and their query embeddings."""
        models = sorted(train["model_name"].unique())
        return cls(
            models=models,
            train_embeddings=train_embeddings,
            score_matrix=per_query_matrix(train, "performance", models),
            cost_matrix=per_query_matrix(train, "cost", models) * COST_SCALE,
            k=k,
        )

    def predict(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted (score, cost) per model: similarity-weighted mean over the k nearest train queries."""
        neighbours, similarity = nearest_neighbours(query_embeddings, self.train_embeddings, self.k)
        return (
            weighted_mean(self.score_matrix[neighbours], similarity),
            weighted_mean(self.cost_matrix[neighbours], similarity),
        )

    def route(self, query_embeddings: np.ndarray, lam: float) -> list[str]:
        """Chosen model name for each query embedding."""
        predicted_score, predicted_cost = self.predict(query_embeddings)
        return pick_models(predicted_score, predicted_cost, lam, self.models)
