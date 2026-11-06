"""Save LLMRouter's per-query routing decisions from already-trained models.

run_pipeline.py writes only aggregate metrics, so a paired comparison needs each test
query's chosen model. This reloads a saved model and calls the pipeline's own
route_test_queries(); it trains nothing. Runs in LLMRouter's Python 3.10 environment:
    ~/code/LLMRouter/.venv/bin/python scripts/llmrouter/export_routing.py \
        --data-dir ~/code/LLMRouter/benchmark_pipeline/data_minilm \
        --routers knnrouter,svmrouter --alpha 0.8 --out results/v4_llmrouter/routing
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PIPELINE = Path.home() / "code" / "LLMRouter" / "benchmark_pipeline"
DATASET = "llmrouter_generic"
# Routers whose route_test_queries() branch needs only the saved model file, not the trained object.
MODEL_FILE_ROUTERS = {"knnrouter", "svmrouter"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--routers", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(PIPELINE))
    import run_pipeline as rp

    rp.DATA_DIR = str(args.data_dir.resolve())
    rp.POOL_DIR = str(args.data_dir.resolve() / "llm_candidates")
    beta = round(1 - args.alpha, 1)
    args.out.mkdir(parents=True, exist_ok=True)

    for router in args.routers.split(","):
        if router not in MODEL_FILE_ROUTERS:
            raise SystemExit(f"{router}: only {sorted(MODEL_FILE_ROUTERS)} are supported")
        cfg = rp.build_config(router, DATASET, args.alpha, beta)
        routing = pd.DataFrame(rp.route_test_queries(router, None, cfg, DATASET, "cpu"))
        path = args.out / f"{router}_a{args.alpha}_b{beta}.csv"
        routing.to_csv(path, index=False)
        print(f"{router} alpha={args.alpha}: {len(routing)} queries -> {path}")


if __name__ == "__main__":
    main()
