"""Two checks on test, nothing retuned: router vs a random cheap/best mix at equal cost, and per-task wins.

Routers: v3 (kNN k=10) and final (kNN k=20), both fit on the full train split, at the
existing sweep's λ values. Fixed models (always-cheapest, always-best) are picked on train.
Writes results/v7_checks/{random_mix,per_task}.{csv,md}.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modelpilot.baselines import always, best_model, cheapest_model
from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import chosen_rows
from modelpilot.mixing import mix_fraction, mixed_scores
from modelpilot.router import KNNRouter
from modelpilot.stats import paired_bootstrap
from modelpilot.summary import model_summary

OUT = Path("results/v7_checks")
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
TASK_LAMBDAS = [0.0, 1.0]
VERSIONS = {"v3 (k=10)": (10, Path("results/v3/router_sweep.csv")), "final (k=20)": (20, Path("results/v6_learned/test_sweep.csv"))}


def rows_for(test: pd.DataFrame, choices: pd.Series) -> pd.DataFrame:
    """Chosen result rows, one per query, in embedding_id order (so strategies pair up)."""
    return chosen_rows(test, choices).set_index("embedding_id").sort_index()


def check_saved(saved_path: Path, version: str, lam: float, rows: pd.DataFrame) -> None:
    """The re-routed sweep must reproduce the saved result exactly (nothing changed)."""
    saved = pd.read_csv(saved_path)
    if "router" in saved:
        saved = saved[saved["router"].str.startswith("kNN")]
    row = saved[saved["lambda"] == lam].iloc[0]
    if abs(rows["performance"].mean() - row["avg_score"]) > 1e-9 or abs(rows["cost"].sum() - row["total_cost"]) > 1e-9:
        raise ValueError(f"{version} λ={lam}: re-run differs from {saved_path}")


def main() -> None:
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)
    encoder = load_encoder()
    train_emb, test_emb = split_embeddings("train", train, encoder), split_embeddings("test", test, encoder)
    test_ids = query_table(test)["embedding_id"]

    summary = model_summary(train)
    cheap_name, best_name = cheapest_model(summary), best_model(summary)
    cheap, best = rows_for(test, always(test, cheap_name)), rows_for(test, always(test, best_name))
    cheap_cost, best_cost = cheap["cost"].sum(), best["cost"].sum()

    mix_rows, task_rows = [], []
    for version, (k, saved_path) in VERSIONS.items():
        router = KNNRouter.from_results(train, train_emb, k=k)
        for lam in LAMBDAS:
            rows = rows_for(test, pd.Series(router.route(test_emb, lam), index=test_ids))
            check_saved(saved_path, version, lam, rows)
            cost = rows["cost"].sum()
            p = mix_fraction(cost, cheap_cost, best_cost)
            mix = mixed_scores(cheap["performance"].to_numpy(), best["performance"].to_numpy(), p)
            boot = paired_bootstrap(rows["performance"].to_numpy(), mix)
            mix_rows.append({
                "version": version, "lambda": lam, "router_avg_score": rows["performance"].mean(), "router_cost": cost,
                "mix_share_best": p, "mix_cost": (1 - p) * cheap_cost + p * best_cost, "mix_avg_score": mix.mean(),
                "diff": boot.diff, "ci_low": boot.low, "ci_high": boot.high, "router_wins": boot.low > 0,
            })
            if lam not in TASK_LAMBDAS:
                continue
            for task, ids in rows.groupby("task_name").groups.items():
                r, b = rows.loc[ids, "performance"].to_numpy(), best.loc[ids, "performance"].to_numpy()
                tb = paired_bootstrap(r, b)
                task_rows.append({
                    "version": version, "lambda": lam, "task": task, "queries": len(ids),
                    "router_avg_score": r.mean(), "best_avg_score": b.mean(),
                    "diff": tb.diff, "ci_low": tb.low, "ci_high": tb.high, "router_wins": tb.low > 0,
                })

    OUT.mkdir(parents=True, exist_ok=True)
    mix_table, task_table = pd.DataFrame(mix_rows), pd.DataFrame(task_rows)
    mix_table.to_csv(OUT / "random_mix.csv", index=False)
    task_table.to_csv(OUT / "per_task.csv", index=False)

    lines = [
        f"Router vs a random mix of always-cheapest ({cheap_name}, ${cheap_cost:.3f}) and always-best ({best_name}, ${best_cost:.3f})",
        "with the same total cost, on the test split (3,729 queries). Diff = router − mix avg score, paired bootstrap 95% CI",
        "(1,000 resamples, seed 0). The mix's score is its expected score over random assignments. ✓ = CI above 0.",
        "",
        "| Version | λ | Router avg score | Router cost ($) | Mix share to best | Mix avg score | Diff [95% CI] | Router wins |",
        "|---|---:|---:|---:|---:|---:|---|:---:|",
    ]
    for r in mix_table.rename(columns={"lambda": "lam"}).itertuples():
        note = " (router below cheapest cost; mix = always-cheapest)" if r.router_cost < cheap_cost else ""
        lines.append(
            f"| {r.version} | {r.lam:g} | {r.router_avg_score:.3f} | {r.router_cost:.3f} | {r.mix_share_best:.1%}{note} | "
            f"{r.mix_avg_score:.3f} | {r.diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {'✓' if r.router_wins else ''} |"
        )
    (OUT / "random_mix.md").write_text("\n".join(lines) + "\n")

    lines = [
        f"Avg score per task on the test split: router vs always-best ({best_name}). Diff = router − always-best,",
        "paired bootstrap 95% CI within the task (1,000 resamples, seed 0). ✓ = router wins (CI above 0).",
        "Small tasks (aime 13, human_eval 16, 50-query tasks) have wide intervals.",
        "",
        "| Version | λ | Task | Queries | Router | Always-best | Diff [95% CI] | Router wins |",
        "|---|---:|---|---:|---:|---:|---|:---:|",
    ]
    for r in task_table.rename(columns={"lambda": "lam"}).itertuples():
        lines.append(
            f"| {r.version} | {r.lam:g} | {r.task} | {r.queries} | {r.router_avg_score:.3f} | {r.best_avg_score:.3f} | "
            f"{r.diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {'✓' if r.router_wins else ''} |"
        )
    (OUT / "per_task.md").write_text("\n".join(lines) + "\n")
    print((OUT / "random_mix.md").read_text())
    print((OUT / "per_task.md").read_text())


if __name__ == "__main__":
    main()
