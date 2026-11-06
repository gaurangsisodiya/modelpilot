"""Build a copy of LLMRouter's input data whose train labels break ties by avg score.

LLMRouter labels each train query with groupby("query")["performance"].idxmax(), which
returns the first tied row, i.e. whichever model the file lists first (qwen2.5-7b-instruct).
Here each query's rows are reordered by the model's train avg score, highest first, so
the same unchanged idxmax picks the best-on-average model among those tied. Test data,
embeddings and the candidate pool are symlinked unchanged.
"""

import argparse
from pathlib import Path

import pandas as pd

PIPELINE = Path.home() / "code" / "LLMRouter" / "benchmark_pipeline"
DATASET = "llmrouter_generic"


def reorder_by_avg_score(train: pd.DataFrame) -> pd.DataFrame:
    """Within each query, sort rows by the model's train avg score (highest first)."""
    avg_score = train.groupby("model_name")["performance"].mean()
    ranked = train.assign(_avg=train["model_name"].map(avg_score))
    ranked = ranked.sort_values(["embedding_id", "_avg"], ascending=[True, False], kind="mergesort")
    return ranked.drop(columns="_avg").reset_index(drop=True)


def link(target: Path, link_path: Path) -> None:
    """Create a symlink unless it already exists."""
    if not link_path.exists():
        link_path.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PIPELINE / "data_minilm")
    parser.add_argument("--out", type=Path, default=PIPELINE / "data_minilm_modified_labels")
    args = parser.parse_args()

    src, dst = args.source / DATASET, args.out / DATASET
    dst.mkdir(parents=True, exist_ok=True)

    train = pd.read_json(src / "routing_data_train.jsonl", lines=True)
    reordered = reorder_by_avg_score(train)
    reordered.to_json(dst / "routing_data_train.jsonl", orient="records", lines=True)
    print(f"train: {len(reordered)} rows reordered -> {dst / 'routing_data_train.jsonl'}")
    print("first model listed per query now:", reordered.groupby("embedding_id")["model_name"].first().value_counts().to_dict())

    for name in ("routing_data_test.jsonl", "query_embeddings_train.pt", "query_embeddings_test.pt"):
        link((src / name).resolve(), dst / name)
    link((args.source / "llm_candidates").resolve(), args.out / "llm_candidates")


if __name__ == "__main__":
    main()
