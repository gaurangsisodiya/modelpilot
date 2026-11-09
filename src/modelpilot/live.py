"""Cost of a live benchmark run, from the gateway's per-request log."""

import json
import statistics
from datetime import datetime
from pathlib import Path


def read_gateway_log(path: Path) -> list[dict]:
    """Gateway log lines, each with an added `epoch` (request start, seconds)."""
    records = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        record["epoch"] = datetime.fromisoformat(record["time"]).timestamp()
        records.append(record)
    return records


def window_costs(
    records: list[dict], setup: str, start: float, end: float, gpu_price_per_hour: float | None
) -> dict:
    """Requests of one benchmark window (same setup, start <= time <= end) and their cost per 1k requests.

    token_cost_per_1k: both backends at xRouteBench prices (comparable to the offline numbers).
    live_cost_per_1k: what you actually pay; Together tokens for large, plus GPU time for the
    whole window when the setup uses the self-hosted small model (the GPU bills even when idle).
    """
    done = [r for r in records if r["requested"] == setup and start <= r["epoch"] <= end and r["status"] == 200]
    n = len(done)
    if n == 0:
        return {"logged_requests": 0, "share_large": float("nan"), "route_ms_median": float("nan"),
                "token_cost_per_1k": float("nan"), "live_cost_per_1k": float("nan")}
    large_cost = sum(r["cost_usd"] for r in done if r["backend"] == "large")
    uses_gpu = setup in ("small", "auto")
    if uses_gpu and gpu_price_per_hour is None:
        gpu_cost = float("nan")
    else:
        gpu_cost = gpu_price_per_hour * (end - start) / 3600 if uses_gpu else 0.0
    return {
        "logged_requests": n,
        "share_large": sum(r["backend"] == "large" for r in done) / n,
        "route_ms_median": statistics.median(r["route_s"] for r in done) * 1000,
        "token_cost_per_1k": sum(r["cost_usd"] for r in done) / n * 1000,
        "live_cost_per_1k": (large_cost + gpu_cost) / n * 1000,
    }
