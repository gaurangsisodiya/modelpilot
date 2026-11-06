"""Gateway routing, streaming, logging and errors, against the fake backend in-process (no network)."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from modelpilot.fake_backend import create_fake_backend
from modelpilot.gateway import create_app, last_user_text
from modelpilot.gateway_config import Backend, GatewayConfig

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "what is two plus two"}]


def config(tmp_path) -> GatewayConfig:
    return GatewayConfig(
        backends={
            "small": Backend("small", "qwen2.5-7b-instruct", "http://small/v1", "Qwen/Qwen2.5-7B-Instruct", "EMPTY", 0.20, 0.20),
            "large": Backend("large", "gpt-oss-120b", "http://large/v1", "openai/gpt-oss-120b", "key", 0.15, 0.60),
        },
        lam=1.0,
        log_path=tmp_path / "gateway.jsonl",
    )


def gateway(tmp_path, choose=None, transport=None) -> tuple[TestClient, list]:
    """Gateway wired to one fake backend (both hosts reach it); records what the chooser was called with."""
    calls = []

    def default_choose(text: str, lam: float) -> str:
        calls.append((text, lam))
        return "large"

    http = httpx.AsyncClient(transport=transport or httpx.ASGITransport(app=create_fake_backend(delay_seconds=0.0)))
    app = create_app(config(tmp_path), choose or default_choose, http)
    return TestClient(app), calls


def log_lines(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "gateway.jsonl").read_text().splitlines()]


def test_auto_routes_with_chooser_and_logs_tokens_and_cost(tmp_path):
    client, calls = gateway(tmp_path)
    with client:
        reply = client.post("/v1/chat/completions", json={"model": "auto", "messages": MESSAGES, "max_tokens": 10})
    assert reply.status_code == 200
    assert reply.headers["x-modelpilot-backend"] == "large"
    assert reply.json()["model"] == "openai/gpt-oss-120b"  # backend saw the real model id
    assert calls == [("what is two plus two", 1.0)]
    record = log_lines(tmp_path)[0]
    assert (record["backend"], record["status"], record["completion_tokens"]) == ("large", 200, 10)
    assert record["cost_usd"] == pytest.approx((record["prompt_tokens"] * 0.15 + 10 * 0.60) / 1e6)
    assert record["latency_s"] >= 0


def test_forced_backend_skips_router(tmp_path):
    def must_not_route(text: str, lam: float) -> str:
        raise AssertionError("router called for a forced backend")

    client, _ = gateway(tmp_path, choose=must_not_route)
    with client:
        reply = client.post("/v1/chat/completions", json={"model": "small", "messages": MESSAGES})
    assert reply.json()["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert log_lines(tmp_path)[0]["lambda"] is None


def test_lambda_field_reaches_router(tmp_path):
    client, calls = gateway(tmp_path)
    with client:
        client.post("/v1/chat/completions", json={"model": "auto", "messages": MESSAGES, "lambda": 7})
    assert calls[0][1] == 7.0


def test_unknown_model_is_rejected(tmp_path):
    client, _ = gateway(tmp_path)
    with client:
        reply = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": MESSAGES})
    assert reply.status_code == 400
    assert "auto" in reply.json()["error"]["message"]


def test_streaming_relays_chunks_and_logs_usage(tmp_path):
    client, _ = gateway(tmp_path)
    with client:
        with client.stream("POST", "/v1/chat/completions",
                           json={"model": "large", "messages": MESSAGES, "max_tokens": 5, "stream": True}) as reply:
            lines = [line for line in reply.iter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in map(lambda l: json.loads(l[5:]), lines[:-1]) if c["choices"])
    assert text == "token " * 5
    record = log_lines(tmp_path)[0]
    assert record["stream"] and record["completion_tokens"] == 5 and record["ttft_s"] <= record["latency_s"]


def test_unreachable_backend_gives_502_and_is_logged(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, _ = gateway(tmp_path, transport=httpx.MockTransport(refuse))
    with client:
        reply = client.post("/v1/chat/completions", json={"model": "small", "messages": MESSAGES})
    assert reply.status_code == 502
    assert log_lines(tmp_path)[0]["status"] == 502


def test_models_lists_routing_choices(tmp_path):
    client, _ = gateway(tmp_path)
    with client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["auto", "small", "large"]


def test_fake_backend_waits_its_delay():
    client = TestClient(create_fake_backend(delay_seconds=0.2))
    start = time.perf_counter()
    reply = client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 3})
    assert time.perf_counter() - start >= 0.2
    assert reply.json()["usage"]["completion_tokens"] == 3


def test_last_user_text_handles_content_parts():
    messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "hi"},
                {"role": "user", "content": [{"type": "text", "text": "new"}, {"type": "image_url", "image_url": {}}]}]
    assert last_user_text(messages) == "new"
    assert last_user_text([]) == ""
