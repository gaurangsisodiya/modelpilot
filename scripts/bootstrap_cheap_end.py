"""Paired bootstrap at the cheap end: LLMRouter knn/svm (alpha=0.8) vs ModelPilot λ=10 and always-cheapest."""

import json
from pathlib import Path

import pandas as pd

from modelpilot.baselines import always, cheapest_model
from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import chosen_rows
from modelpilot.router import KNNRouter
from modelpilot.stats import paired_bootstrap
from modelpilot.summary import model_summary

V4 = Path("results/v4_llmrouter")
LLMROUTER_RUNS = {"LLMRouter knnrouter α=0.8": "knnrouter_a0.8_b0.2", "LLMRouter svmrouter α=0.8": "svmrouter_a0.8_b0.2"}
ROUTER_LAMBDA = 10.0
N_RESAMPLES = 1000


def llmrouter_choices(name: str, queries: pd.DataFrame) -> pd.Series:
    """LLMRouter's exported routing (query text -> model) as embedding_id -> model."""
    routing = pd.read_csv(V4 / "routing" / f"{name}.csv")
    merged = routing.merge(queries, on="query", how="left", validate="one_to_one")
    if merged["embedding_id"].isna().any():
        raise ValueError(f"{name}: some routed queries do not match a test query")
    return merged.set_index("embedding_id")["routed_model"]


def check_against_saved(name: str, rows: pd.DataFrame) -> None:
    """Re-routed decisions must reproduce the pipeline's saved aggregate result."""
    saved = json.loads((V4 / "raw" / "minilm" / f"llmrouter_generic_{name}.json").read_text())
    score, cost = rows["performance"].mean(), rows["cost"].mean()
    if abs(score - saved["avg_performance"]) > 1e-4 or abs(cost - saved["avg_price_cost"]) > 1e-8:
        raise ValueError(f"{name}: re-routed {score:.4f}/{cost:.8f} != saved {saved['avg_performance']}/{saved['avg_price_cost']}")


def main() -> None:
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)
    queries = query_table(test)

    encoder = load_encoder()
    router = KNNRouter.from_results(train, split_embeddings("train", train, encoder), k=10)
    test_embeddings = split_embeddings("test", test, encoder)
    cheapest = cheapest_model(model_summary(train))

    strategies = {
        f"ModelPilot kNN λ={ROUTER_LAMBDA:g}": pd.Series(router.route(test_embeddings, ROUTER_LAMBDA), index=queries["embedding_id"]),
        f"always-cheapest ({cheapest})": always(test, cheapest),
    }
    for label, name in LLMROUTER_RUNS.items():
        strategies[label] = llmrouter_choices(name, queries)

    rows = {label: chosen_rows(test, choices).set_index("embedding_id").sort_index() for label, choices in strategies.items()}
    for label, name in LLMROUTER_RUNS.items():
        check_against_saved(name, rows[label])

    results = []
    for a in LLMROUTER_RUNS:
        for b in [k for k in strategies if k not in LLMROUTER_RUNS]:
            boot = paired_bootstrap(rows[a]["performance"].to_numpy(), rows[b]["performance"].to_numpy(), n_resamples=N_RESAMPLES)
            results.append(
                {
                    "a": a, "b": b,
                    "avg_score_a": rows[a]["performance"].mean(), "avg_score_b": rows[b]["performance"].mean(),
                    "score_diff": boot.diff, "ci_low": boot.low, "ci_high": boot.high,
                    "total_cost_a": rows[a]["cost"].sum(), "total_cost_b": rows[b]["cost"].sum(),
                }
            )
    table = pd.DataFrame(results)
    table.to_csv(V4 / "bootstrap_cheap_end.csv", index=False)

    lines = [
        f"Paired bootstrap, {N_RESAMPLES} resamples over {len(queries)} test queries. Difference = A − B in avg score.",
        "",
        "| A | B | Avg score A | Avg score B | Difference (95% CI) | Cost A ($) | Cost B ($) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| {r.a} | {r.b} | {r.avg_score_a:.3f} | {r.avg_score_b:.3f} | "
            f"{r.score_diff:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.total_cost_a:.3f} | {r.total_cost_b:.3f} |"
        )
    markdown = "\n".join(lines) + "\n"
    (V4 / "bootstrap_cheap_end.md").write_text(markdown)
    print("re-routed LLMRouter decisions match the saved pipeline results\n")
    print(markdown)


if __name__ == "__main__":
    main()
