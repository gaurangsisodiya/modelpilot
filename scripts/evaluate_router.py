"""Evaluate the kNN router on the test split for several lambda values."""

import argparse
from pathlib import Path

import pandas as pd

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import evaluate
from modelpilot.router import KNNRouter

LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/latest"), help="folder for router_sweep.csv")
    args = parser.parse_args()

    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)

    encoder = load_encoder()
    router = KNNRouter.from_results(train, split_embeddings("train", train, encoder), k=10)
    test_embeddings = split_embeddings("test", test, encoder)
    test_ids = query_table(test)["embedding_id"]

    rows = []
    for lam in LAMBDAS:
        choices = pd.Series(router.route(test_embeddings, lam), index=test_ids)
        rows.append({"lambda": lam, **evaluate(test, choices), "top_model": choices.mode()[0]})

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "router_sweep.csv", index=False)

    print("kNN router (k=10) on test split\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {args.out / 'router_sweep.csv'}")


if __name__ == "__main__":
    main()
