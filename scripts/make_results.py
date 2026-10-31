"""Write the results table (markdown) and cost-vs-score chart for a router sweep."""

import argparse
from pathlib import Path

import pandas as pd

from modelpilot.baselines import evaluate_baselines
from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.report import plot_cost_vs_score, results_table, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("results/v3/router_sweep.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)

    baselines, _ = evaluate_baselines(train, test)
    sweep = pd.read_csv(args.sweep)

    args.out.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(args.out / "baselines.csv", index=False)

    best_cost = baselines.loc[baselines["strategy"].str.startswith("always-best"), "total_cost"].iloc[0]
    markdown = to_markdown(results_table(baselines, sweep), reference_cost=best_cost)
    (args.out / "results.md").write_text(markdown)

    plot_cost_vs_score(baselines, sweep, args.out / "cost_vs_score.png")

    print(markdown)
    print(f"Wrote {args.out / 'results.md'}, {args.out / 'cost_vs_score.png'}, {args.out / 'baselines.csv'}")


if __name__ == "__main__":
    main()
