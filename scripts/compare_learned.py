"""Final, one-time test comparison: kNN vs learned router with the settings fixed on validation.

Reads results/v6_learned/validation_decision.json (from tune_routers.py); nothing here is tuned.
Both routers are refit on the full train split (fit + validation) before routing test.
"""

import json
from pathlib import Path

import pandas as pd

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import chosen_rows, evaluate
from modelpilot.learned import LearnedRouter
from modelpilot.router import KNNRouter
from modelpilot.stats import paired_bootstrap

OUT = Path("results/v6_learned")
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def main() -> None:
    decision = json.loads((OUT / "validation_decision.json").read_text())
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)
    encoder = load_encoder()
    train_emb, test_emb = split_embeddings("train", train, encoder), split_embeddings("test", test, encoder)
    test_ids = query_table(test)["embedding_id"]

    routers = {
        f"kNN k={decision['knn_k']}": KNNRouter.from_results(train, train_emb, k=decision["knn_k"]),
        f"MLP hidden={decision['mlp_hidden']}": LearnedRouter.from_results(
            train, train_emb, hidden=decision["mlp_hidden"], alpha=decision["mlp_alpha"], cost_alpha=decision["cost_alpha"]
        ),
    }
    knn_name, mlp_name = routers

    sweep, picks = [], {}
    for lam in LAMBDAS:
        for name, router in routers.items():
            choices = pd.Series(router.route(test_emb, lam), index=test_ids)
            picks[name, lam] = choices
            sweep.append({"router": name, "lambda": lam, **evaluate(test, choices), "top_model": choices.mode()[0]})
    sweep = pd.DataFrame(sweep)

    lam = decision["default_lambda"]
    rows = {n: chosen_rows(test, picks[n, lam]).set_index("embedding_id").sort_index() for n in routers}
    boot = paired_bootstrap(rows[mlp_name]["performance"].to_numpy(), rows[knn_name]["performance"].to_numpy())
    cost_ratio = rows[mlp_name]["cost"].sum() / rows[knn_name]["cost"].sum()

    OUT.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUT / "test_sweep.csv", index=False)
    lines = [
        f"Test split ({len(test_ids)} queries), run once with settings fixed on validation; default λ = {lam:g}.",
        "",
        "| λ | " + " | ".join(f"{n} avg score | {n} cost ($)" for n in routers) + " |",
        "|---:|" + "---:|---:|" * len(routers),
    ]
    for value in LAMBDAS:
        cells = []
        for name in routers:
            r = sweep[(sweep["router"] == name) & (sweep["lambda"] == value)].iloc[0]
            cells += [f"{r.avg_score:.3f}", f"{r.total_cost:.3f}"]
        lines.append(f"| {value:g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Paired bootstrap at λ = {lam:g} (MLP − kNN avg score, 1,000 resamples): "
        f"{boot.diff:+.4f} [{boot.low:+.4f}, {boot.high:+.4f}]; MLP cost / kNN cost = {cost_ratio:.2f}.",
        f"Default router chosen on validation: {decision['default_router']}.",
    ]
    markdown = "\n".join(lines) + "\n"
    (OUT / "test_comparison.md").write_text(markdown)
    print(markdown)
    print(f"Saved {OUT / 'test_sweep.csv'} and {OUT / 'test_comparison.md'}")


if __name__ == "__main__":
    main()
