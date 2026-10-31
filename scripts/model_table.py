"""Print each model's average score and average cost per query on the train split."""

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.summary import model_summary


def main() -> None:
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)

    table = model_summary(train)
    table = table.join(candidates.set_index("model_name")["service"])
    table["avg_cost_per_1k_queries"] = table["avg_cost"] * 1000

    print("Train split: per-model avg score and avg cost (sorted by avg score)\n")
    print(
        table[["service", "avg_score", "avg_cost_per_1k_queries", "queries"]].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()
