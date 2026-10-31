"""Evaluate the baseline strategies on the test split."""

import pandas as pd

from modelpilot.baselines import always, best_model, cheapest_model, oracle
from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.evaluation import evaluate
from modelpilot.summary import model_summary


def main() -> None:
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)

    # Fixed models are chosen from train stats so test data never influences the choice.
    train_summary = model_summary(train)
    cheapest = cheapest_model(train_summary)
    best = best_model(train_summary)

    strategies = {
        f"always-cheapest ({cheapest})": always(test, cheapest),
        f"always-best ({best})": always(test, best),
        "oracle": oracle(test),
    }

    rows = [{"strategy": name, **evaluate(test, choices)} for name, choices in strategies.items()]
    table = pd.DataFrame(rows).set_index("strategy")

    print("Test split baselines\n")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nOracle picks per model:")
    print(strategies["oracle"].value_counts().to_string())


if __name__ == "__main__":
    main()
