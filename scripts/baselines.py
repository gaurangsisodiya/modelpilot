"""Evaluate the baseline strategies on the test split."""

from modelpilot.baselines import evaluate_baselines
from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split


def main() -> None:
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)

    table, oracle_choices = evaluate_baselines(train, test)

    print("Test split baselines\n")
    print(table.set_index("strategy").to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nOracle picks per model:")
    print(oracle_choices.value_counts().to_string())


if __name__ == "__main__":
    main()
