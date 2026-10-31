"""Evaluate the kNN router on the test split for several lambda values."""

import pandas as pd

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import evaluate
from modelpilot.router import KNNRouter

LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def main() -> None:
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

    print("kNN router (k=10) on test split\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
