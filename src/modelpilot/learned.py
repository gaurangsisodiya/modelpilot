"""Learned router: regressors predict each model's score and cost straight from the embedding."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from modelpilot.router import COST_SCALE, per_query_matrix, pick_models


@dataclass
class LearnedRouter:
    """MLP for score (all models at once), Ridge for cost; same utility decision as kNN."""

    models: list[str]
    score_model: MLPRegressor
    cost_model: Ridge

    @classmethod
    def from_results(
        cls,
        train: pd.DataFrame,
        train_embeddings: np.ndarray,
        hidden: int = 128,
        alpha: float = 1e-4,
        cost_alpha: float = 1.0,
        seed: int = 0,
    ) -> "LearnedRouter":
        """Fit on train results (with `cost`) and their embeddings (rows in embedding_id order).

        hidden = units in the MLP's one hidden layer; alpha = its L2 penalty;
        cost_alpha = Ridge penalty for the cost model.
        """
        models = sorted(train["model_name"].unique())
        score = per_query_matrix(train, "performance", models)
        cost = per_query_matrix(train, "cost", models) * COST_SCALE
        score_model = MLPRegressor(
            hidden_layer_sizes=(hidden,), alpha=alpha, early_stopping=True, max_iter=300, random_state=seed
        ).fit(train_embeddings, score)
        cost_model = Ridge(alpha=cost_alpha).fit(train_embeddings, cost)
        return cls(models=models, score_model=score_model, cost_model=cost_model)

    def predict(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted (score, cost) per model, shape (queries, models); score clipped to [0, 1], cost to >= 0."""
        score = np.clip(self.score_model.predict(query_embeddings), 0.0, 1.0)
        cost = np.clip(self.cost_model.predict(query_embeddings), 0.0, None)
        return score.astype(np.float64), cost.astype(np.float64)

    def route(self, query_embeddings: np.ndarray, lam: float) -> list[str]:
        """Chosen model name for each query embedding."""
        predicted_score, predicted_cost = self.predict(query_embeddings)
        return pick_models(predicted_score, predicted_cost, lam, self.models)
