"""Write our MiniLM embeddings in LLMRouter's benchmark_pipeline layout.

Creates <out>/llmrouter_generic/ with the routing jsonl files (symlinked from LLMRouter's
own download) and query_embeddings_{train,test}.pt, plus <out>/llm_candidates/ with the
pool file and MiniLM embeddings of each model's description. LLMRouter expects tensor
row i to be the query with embedding_id == i.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from modelpilot.data import load_split
from modelpilot.embeddings import embed, load_encoder, query_table, split_embeddings

PIPELINE = Path.home() / "code" / "LLMRouter" / "benchmark_pipeline"
DATASET = "llmrouter_generic"


def export_split(split: str, out_dir: Path, encoder) -> None:
    """Save one split's embeddings as a float32 tensor indexed by embedding_id."""
    results = load_split(split)
    ids = query_table(results)["embedding_id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError(f"{split}: embedding_id is not 0..N-1, row order would not match")
    vectors = split_embeddings(split, results, encoder)
    torch.save(torch.from_numpy(vectors.astype(np.float32)), out_dir / f"query_embeddings_{split}.pt")
    print(f"{split}: {vectors.shape} -> {out_dir / f'query_embeddings_{split}.pt'}")


def export_candidates(source: Path, out_dir: Path, encoder) -> None:
    """Copy the pool file and embed each model's description (used by graphrouter)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "default_llm.json", out_dir / "default_llm.json")
    pool = json.loads((source / "default_llm.json").read_text())
    names = list(pool)
    vectors = embed([pool[name].get("feature", name) for name in names], encoder)
    embeddings = {name: vectors[i].tolist() for i, name in enumerate(names)}
    (out_dir / "default_llm_embeddings.json").write_text(json.dumps(embeddings))
    print(f"llm_candidates: {len(names)} descriptions -> {out_dir / 'default_llm_embeddings.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PIPELINE / "data_minilm")
    args = parser.parse_args()

    source = PIPELINE / "data"
    dataset_dir = args.out / DATASET
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        link = dataset_dir / f"routing_data_{split}.jsonl"
        if not link.exists():
            link.symlink_to(source / DATASET / f"routing_data_{split}.jsonl")

    encoder = load_encoder()
    for split in ("train", "test"):
        export_split(split, dataset_dir, encoder)
    export_candidates(source / "llm_candidates", args.out / "llm_candidates", encoder)


if __name__ == "__main__":
    main()
