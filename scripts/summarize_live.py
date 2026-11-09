"""Summarise the live GuideLLM runs (results/v5_live): speed, cost per 1k requests, expected avg score."""

import argparse
import json
from pathlib import Path

import pandas as pd

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import load_encoder, query_table, split_embeddings
from modelpilot.evaluation import evaluate
from modelpilot.gateway_config import load_config
from modelpilot.live import read_gateway_log, window_costs
from modelpilot.router import DEFAULT_K, KNNRouter, per_query_matrix

SETUPS = ["small", "large", "auto"]


def expected_scores(lam: float) -> dict[str, float]:
    """Offline avg score on the full xRouteBench test split with only the two gateway models."""
    candidates = load_candidates()
    config = load_config(candidates)
    small, large = config.backends["small"].router_model, config.backends["large"].router_model
    train = add_cost(load_split("train"), candidates)
    test = add_cost(load_split("test"), candidates)
    pair_test = test[test["model_name"].isin([small, large])]
    encoder = load_encoder()
    router = KNNRouter.from_results(
        train[train["model_name"].isin([small, large])], split_embeddings("train", train, encoder), k=DEFAULT_K
    )
    routed = pd.Series(router.route(split_embeddings("test", test, encoder), lam), index=query_table(test)["embedding_id"])
    score = per_query_matrix(pair_test, "performance", [small, large]).mean(axis=0)
    return {"small": float(score[0]), "large": float(score[1]), "auto": evaluate(pair_test, routed)["avg_score"]}


def stat(metric: dict, key: str) -> float:
    """mean / median / p95 of a GuideLLM metric over successful requests."""
    values = metric["successful"]
    return values["percentiles"]["p95"] if key == "p95" else values[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-price-per-hour", type=float, default=None, help="RunPod $/hour of the pod")
    parser.add_argument("--gpu", default="", help="GPU name, for the table caption")
    parser.add_argument("--raw", type=Path, default=Path("results/v5_live/raw"))
    args = parser.parse_args()
    out = args.raw.parent

    records = read_gateway_log(args.raw / "gateway.jsonl")
    auto_lambdas = {r["lambda"] for r in records if r["requested"] == "auto"}
    lam = auto_lambdas.pop() if len(auto_lambdas) == 1 else 1.0
    expected = expected_scores(lam)

    rows = []
    for setup in SETUPS:
        path = args.raw / f"guidellm_{setup}.json"
        if not path.exists():
            continue
        for bench in json.loads(path.read_text())["benchmarks"]:
            m = bench["metrics"]
            rows.append({
                "setup": {"small": "all-small", "large": "all-large", "auto": f"routed λ={lam:g}"}[setup],
                "concurrency": bench["config"]["strategy"]["streams"],
                "requests": m["request_totals"]["successful"],
                "errors": m["request_totals"]["errored"],
                "req_per_s": stat(m["requests_per_second"], "mean"),
                "output_tok_per_s": stat(m["output_tokens_per_second"], "mean"),
                "ttft_ms_median": stat(m["time_to_first_token_ms"], "median"),
                "ttft_ms_p95": stat(m["time_to_first_token_ms"], "p95"),
                "latency_s_median": stat(m["request_latency"], "median"),
                "latency_s_p95": stat(m["request_latency"], "p95"),
                **window_costs(records, setup, bench["start_time"], bench["end_time"], args.gpu_price_per_hour),
                "expected_avg_score": expected[setup],
            })
    table = pd.DataFrame(rows)
    table.to_csv(out / "live_results.csv", index=False)

    gpu = f"{args.gpu} at ${args.gpu_price_per_hour}/h" if args.gpu_price_per_hour is not None else "GPU price not given"
    lines = [
        f"Live benchmark through the gateway ({gpu}). Expected avg score: offline, full test split, two models only.",
        "Token cost = xRouteBench prices for both models; live cost = Together tokens + GPU time for the window.",
        "",
        "| Setup | Conc. | Req/s | Out tok/s | TTFT med / p95 (ms) | Latency med / p95 (s) | Route ms (median) | Share large | Token cost / 1k ($) | Live cost / 1k ($) | Expected avg score | Errors |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| {r.setup} | {r.concurrency} | {r.req_per_s:.2f} | {r.output_tok_per_s:.0f} | "
            f"{r.ttft_ms_median:.0f} / {r.ttft_ms_p95:.0f} | {r.latency_s_median:.2f} / {r.latency_s_p95:.2f} | "
            f"{r.route_ms_median:.0f} | {r.share_large:.0%} | {r.token_cost_per_1k:.3f} | {r.live_cost_per_1k:.3f} | {r.expected_avg_score:.3f} | {r.errors} |"
        )
    markdown = "\n".join(lines) + "\n"
    (out / "live_results.md").write_text(markdown)
    print(markdown)
    print(f"Saved {out / 'live_results.csv'} and {out / 'live_results.md'}")


if __name__ == "__main__":
    main()
