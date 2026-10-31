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


def nearest_neighbours(queries: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k most similar train rows for each query (cosine similarity).

    Both inputs must be unit-length vectors, so the dot product is the cosine similarity.
    """
    similarity = queries @ train.T
    return np.argpartition(-similarity, k - 1, axis=1)[:, :k]


def pick_models(
    predicted_score: np.ndarray, predicted_cost: np.ndarray, lam: float, models: list[str]
) -> list[str]:
    """For each query, the model maximising predicted_score - lam * predicted_cost.

    Both arrays have shape (queries, models); cost is in dollars per 1,000 queries.
    """
    utility = predicted_score - lam * predicted_cost
    return [models[i] for i in utility.argmax(axis=1)]


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
        """Predicted (score, cost) per model: the mean over the k nearest train queries."""
        neighbours = nearest_neighbours(query_embeddings, self.train_embeddings, self.k)
        return self.score_matrix[neighbours].mean(axis=1), self.cost_matrix[neighbours].mean(axis=1)

    def route(self, query_embeddings: np.ndarray, lam: float) -> list[str]:
        """Chosen model name for each query embedding."""
        predicted_score, predicted_cost = self.predict(query_embeddings)
        return pick_models(predicted_score, predicted_cost, lam, self.models)
