"""Run LLMRouter's benchmark_pipeline on llmrouter_generic for a chosen embedding set.

Runs inside LLMRouter's own Python 3.10 environment, not ModelPilot's:
    ~/code/LLMRouter/.venv/bin/python scripts/llmrouter/run_llmrouter_pipeline.py \
        --data-dir ~/code/LLMRouter/benchmark_pipeline/data_minilm --text-dim 384 \
        --results-dir results/v4_llmrouter/raw/minilm

It imports the pipeline's run_pipeline.py and only redirects where it reads data and
writes results (and mfrouter's text_dim, which must match the embedding size). Training,
routing and evaluation are LLMRouter's own code, called through its run_job().
"""

import argparse
import sys
import time
from pathlib import Path

PIPELINE = Path.home() / "code" / "LLMRouter" / "benchmark_pipeline"
DATASET = "llmrouter_generic"
# CPU-feasible local routers. Excluded: routerdc (fine-tunes mDeBERTa, hard-coded to CUDA),
# causallm_router (LoRA on Llama-2-7b + vLLM, GPU only), API routers (live LLM calls).
ROUTERS = [
    "knnrouter", "svmrouter", "mlprouter", "mfrouter", "elorouter", "graphrouter",
    "hybrid_llm", "largest_llm", "smallest_llm", "gmtrouter", "personalizedrouter",
]
ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--text-dim", type=int, required=True, help="embedding size: 384 MiniLM, 1024 Qwen3")
    parser.add_argument("--routers", default=",".join(ROUTERS))
    parser.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    args = parser.parse_args()

    sys.path.insert(0, str(PIPELINE))
    import run_pipeline as rp

    rp.DATA_DIR = str(args.data_dir.resolve())
    rp.POOL_DIR = str(args.data_dir.resolve() / "llm_candidates")
    rp.RESULTS_DIR = str(args.results_dir.resolve())
    rp.ROUTER_HPARAMS["mfrouter"]["text_dim"] = args.text_dim

    for alpha in (float(a) for a in args.alphas.split(",")):
        beta = round(1 - alpha, 1)
        for router in args.routers.split(","):
            start = time.time()
            ok = rp.run_job(router, DATASET, alpha, beta, device="cpu")
            print(f"[time]  {router} alpha={alpha}: {'ok' if ok else 'FAILED'} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
