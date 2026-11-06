"""Put LLMRouter's results next to ModelPilot's router and baselines (results/v4_llmrouter)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from modelpilot.baselines import always, evaluate_baselines
from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.evaluation import evaluate

V4 = Path("results/v4_llmrouter")


def load_llmrouter(raw_dir: Path) -> pd.DataFrame:
    """One row per LLMRouter result JSON; raw/<embeddings>/<dataset>_<router>_a<a>_b<b>.json."""
    rows = []
    for path in sorted(raw_dir.glob("*/*.json")):
        result = json.loads(path.read_text())
        rows.append(
            {
                "source": "LLMRouter",
                "router": result["router"],
                "embeddings": path.parent.name,
                "setting": f"alpha={result['alpha']}",
                "alpha": result["alpha"],
                "avg_score": result["avg_performance"],
                "total_cost": result["avg_price_cost"] * result["num_queries"],
                "queries": result["num_queries"],
                "top_model": max(result["routing_distribution"], key=result["routing_distribution"].get),
            }
        )
    return pd.DataFrame(rows)


def load_modelpilot(sweep_path: Path, baselines: pd.DataFrame) -> pd.DataFrame:
    """ModelPilot kNN λ sweep (v3) plus the three baselines, in the same columns."""
    sweep = pd.read_csv(sweep_path)
    router = pd.DataFrame(
        {
            "source": "ModelPilot",
            "router": "modelpilot_knn",
            "embeddings": "minilm",
            "setting": [f"lambda={lam:g}" for lam in sweep["lambda"]],
            "avg_score": sweep["avg_score"],
            "total_cost": sweep["total_cost"],
            "queries": sweep["queries"],
            "top_model": sweep["top_model"],
        }
    )
    base = baselines.assign(source="baseline", router=baselines["strategy"], embeddings="-", setting="-")
    return pd.concat([router, base.drop(columns="strategy")], ignore_index=True)


def check_replay(llmrouter: pd.DataFrame, test: pd.DataFrame) -> None:
    """LLMRouter's fixed-model baselines must score exactly like our evaluate() on that model."""
    for row in llmrouter[llmrouter["router"].isin(["largest_llm", "smallest_llm"])].itertuples():
        ours = evaluate(test, always(test, row.top_model))
        if abs(ours["avg_score"] - row.avg_score) > 1e-4 or abs(ours["total_cost"] - row.total_cost) > 1e-4:
            raise ValueError(f"{row.router} ({row.top_model}) does not match ModelPilot's evaluation: {ours}")
        print(f"replay check ok: {row.router} -> {row.top_model} avg score {row.avg_score:.4f}, cost ${row.total_cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("results/v3/router_sweep.csv"))
    args = parser.parse_args()

    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)

    llmrouter = load_llmrouter(V4 / "raw")
    check_replay(llmrouter, test)
    baselines, _ = evaluate_baselines(train, test)
    table = pd.concat([load_modelpilot(args.sweep, baselines), llmrouter], ignore_index=True)

    table.to_csv(V4 / "comparison.csv", index=False)
    print(table.drop(columns=["alpha", "queries"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {V4 / 'comparison.csv'}")


if __name__ == "__main__":
    main()
