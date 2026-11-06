"""Tune kNN and the learned router on a validation slice of train; the test split is not loaded."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import chosen_rows, evaluate
from modelpilot.learned import LearnedRouter
from modelpilot.router import COST_SCALE, KNNRouter, per_query_matrix
from modelpilot.stats import paired_bootstrap
from modelpilot.validation import choose_default_lambda, split_queries, subset_embeddings, wins_clearly

OUT = Path("results/v6_learned")
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
K_GRID = [5, 10, 20, 50]
HIDDEN_GRID = [64, 256]
ALPHA_GRID = [1e-4, 1e-2, 1.0, 10.0]
COST_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0]


def mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error over every (query, model) cell."""
    return float(((predicted - actual) ** 2).mean())


def main() -> None:
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    embeddings = split_embeddings("train", train, load_encoder())
    fit, val = split_queries(train)
    fit_emb, val_emb = subset_embeddings(embeddings, train, fit), subset_embeddings(embeddings, train, val)
    val_ids = query_table(val)["embedding_id"]
    print(f"fit: {fit['embedding_id'].nunique()} queries, validation: {val['embedding_id'].nunique()} queries\n")

    models = sorted(train["model_name"].unique())
    true_score = per_query_matrix(val, "performance", models)
    true_cost = per_query_matrix(val, "cost", models) * COST_SCALE

    # 1. Settings, picked by how well each predicts validation score (and cost) per model.
    tuning = []
    for k in K_GRID:
        score, cost = KNNRouter.from_results(fit, fit_emb, k=k).predict(val_emb)
        tuning.append({"router": "knn", "setting": f"k={k}", "score_mse": mse(score, true_score), "cost_mse": mse(cost, true_cost)})
    for hidden in HIDDEN_GRID:
        for alpha in ALPHA_GRID:
            score, cost = LearnedRouter.from_results(fit, fit_emb, hidden=hidden, alpha=alpha).predict(val_emb)
            tuning.append({"router": "mlp", "setting": f"hidden={hidden} alpha={alpha:g}", "hidden": hidden, "alpha": alpha,
                           "score_mse": mse(score, true_score), "cost_mse": mse(cost, true_cost)})
    tuning = pd.DataFrame(tuning)
    knn_rows, mlp_rows = tuning[tuning["router"] == "knn"], tuning[tuning["router"] == "mlp"]
    best_k = int(knn_rows.loc[knn_rows["score_mse"].idxmin(), "setting"].removeprefix("k="))
    best_mlp = mlp_rows.loc[mlp_rows["score_mse"].idxmin()]

    cost_rows = []
    for cost_alpha in COST_ALPHA_GRID:
        router = LearnedRouter.from_results(fit, fit_emb, hidden=int(best_mlp["hidden"]), alpha=best_mlp["alpha"], cost_alpha=cost_alpha)
        _, cost = router.predict(val_emb)
        cost_rows.append({"router": "ridge cost", "setting": f"cost_alpha={cost_alpha:g}", "cost_alpha": cost_alpha, "cost_mse": mse(cost, true_cost)})
    cost_rows = pd.DataFrame(cost_rows)
    best_cost_alpha = float(cost_rows.loc[cost_rows["cost_mse"].idxmin(), "cost_alpha"])
    tuning = pd.concat([tuning, cost_rows], ignore_index=True)

    print(tuning[["router", "setting", "score_mse", "cost_mse"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nchosen: kNN k={best_k}; MLP hidden={int(best_mlp['hidden'])} alpha={best_mlp['alpha']:g}; Ridge cost_alpha={best_cost_alpha:g}\n")

    # 2. λ sweep on validation with the chosen settings, plus a paired bootstrap at each λ.
    routers = {
        "knn": KNNRouter.from_results(fit, fit_emb, k=best_k),
        "mlp": LearnedRouter.from_results(fit, fit_emb, hidden=int(best_mlp["hidden"]), alpha=best_mlp["alpha"], cost_alpha=best_cost_alpha),
    }
    sweep = []
    for lam in LAMBDAS:
        picked = {name: pd.Series(r.route(val_emb, lam), index=val_ids) for name, r in routers.items()}
        rows = {name: chosen_rows(val, c).set_index("embedding_id").sort_index() for name, c in picked.items()}
        boot = paired_bootstrap(rows["mlp"]["performance"].to_numpy(), rows["knn"]["performance"].to_numpy())
        for name, choices in picked.items():
            sweep.append({"router": name, "lambda": lam, **evaluate(val, choices), "top_model": choices.mode()[0]})
        sweep.append({"router": "mlp - knn", "lambda": lam, "avg_score": boot.diff, "ci_low": boot.low, "ci_high": boot.high,
                      "total_cost": rows["mlp"]["cost"].sum() / rows["knn"]["cost"].sum()})
    sweep = pd.DataFrame(sweep)

    OUT.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(OUT / "validation_tuning.csv", index=False)
    sweep.to_csv(OUT / "validation_sweep.csv", index=False)
    print("validation λ sweep ('mlp - knn': avg_score = score difference with 95% CI, total_cost = cost ratio)\n")
    print(sweep.drop(columns="queries").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Decisions, written down before test is ever looked at.
    default_lambda = choose_default_lambda(sweep[sweep["router"] == "knn"])
    at_default = sweep[(sweep["router"] == "mlp - knn") & (sweep["lambda"] == default_lambda)].iloc[0]
    decision = {
        "knn_k": best_k,
        "mlp_hidden": int(best_mlp["hidden"]),
        "mlp_alpha": float(best_mlp["alpha"]),
        "cost_alpha": best_cost_alpha,
        "default_lambda": default_lambda,
        "mlp_minus_knn_at_default": {"diff": at_default["avg_score"], "ci_low": at_default["ci_low"],
                                     "ci_high": at_default["ci_high"], "cost_ratio": at_default["total_cost"]},
        "default_router": "mlp" if wins_clearly(at_default["ci_low"], at_default["total_cost"]) else "knn",
    }
    (OUT / "validation_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(f"\ndecision: {json.dumps(decision, indent=2)}")
    print(f"\nSaved {OUT / 'validation_tuning.csv'}, {OUT / 'validation_sweep.csv'} and {OUT / 'validation_decision.json'}")


if __name__ == "__main__":
    main()
