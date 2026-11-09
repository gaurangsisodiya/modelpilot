"""Cost per 1k requests for a live benchmark window."""

import json
import math

import pytest

from modelpilot.live import read_gateway_log, window_costs


def rec(epoch: float, requested: str, backend: str, cost: float, status: int = 200) -> dict:
    return {"epoch": epoch, "requested": requested, "backend": backend, "cost_usd": cost, "status": status, "route_s": 0.05}


def test_routed_window_counts_large_tokens_plus_gpu_time():
    records = [rec(10, "auto", "small", 0.001), rec(20, "auto", "large", 0.003), rec(30, "auto", "large", 0.003),
               rec(40, "auto", "small", 0.001, status=502),  # failed: not counted
               rec(50, "small", "small", 0.001),             # other setup: not counted
               rec(500, "auto", "large", 0.003)]             # outside the window
    result = window_costs(records, "auto", start=0, end=360, gpu_price_per_hour=1.0)
    assert result["logged_requests"] == 3
    assert result["share_large"] == pytest.approx(2 / 3)
    assert result["route_ms_median"] == pytest.approx(50)
    assert result["token_cost_per_1k"] == pytest.approx(0.007 / 3 * 1000)
    assert result["live_cost_per_1k"] == pytest.approx((0.006 + 1.0 * 360 / 3600) / 3 * 1000)  # $0.10 of GPU


def test_all_large_pays_no_gpu():
    result = window_costs([rec(1, "large", "large", 0.002)], "large", 0, 3600, gpu_price_per_hour=1.0)
    assert result["live_cost_per_1k"] == pytest.approx(2.0)


def test_missing_gpu_price_gives_nan_live_cost_for_small():
    result = window_costs([rec(1, "small", "small", 0.001)], "small", 0, 10, gpu_price_per_hour=None)
    assert math.isnan(result["live_cost_per_1k"]) and result["token_cost_per_1k"] == pytest.approx(1.0)


def test_empty_window():
    assert window_costs([], "auto", 0, 10, 1.0)["logged_requests"] == 0


def test_read_gateway_log_adds_epoch(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"time": "2026-11-09T12:00:00.000+00:00", "requested": "auto"}) + "\n")
    assert read_gateway_log(path)[0]["epoch"] == pytest.approx(1794225600.0)
