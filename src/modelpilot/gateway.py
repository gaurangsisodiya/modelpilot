"""OpenAI-compatible gateway: POST /v1/chat/completions, routed between a small and a large backend.

`model` in the request picks the backend: "auto" (route with kNN), "small" or "large".
An optional extra field "lambda" overrides the default cost weight for "auto".
Every request is logged as one JSON line (backend, latency, tokens, cost).

Run from the repo root:
    uv run uvicorn modelpilot.gateway:app --port 8000
"""

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from modelpilot.cost import add_cost, query_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import embed, load_encoder, split_embeddings
from modelpilot.gateway_config import Backend, GatewayConfig, load_config
from modelpilot.router import DEFAULT_K, KNNRouter

Chooser = Callable[[str, float], str]  # (query text, λ) -> "small" or "large"
TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def load_chooser(config: GatewayConfig, candidates) -> Chooser:
    """kNN router trained on xRouteBench train, restricted to the two backends' models."""
    train = add_cost(load_split("train"), candidates)
    by_model = {b.router_model: b.name for b in config.backends.values()}
    encoder = load_encoder()
    router = KNNRouter.from_results(
        train[train["model_name"].isin(by_model)], split_embeddings("train", train, encoder), k=DEFAULT_K
    )

    def choose(text: str, lam: float) -> str:
        return by_model[router.route(embed([text], encoder), lam)[0]]

    return choose


def last_user_text(messages: list[dict]) -> str:
    """Text of the last user message (string content, or the text parts of a list)."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return content or ""
    return ""


def log_request(path: Path, record: dict) -> None:
    """Append one JSON line per request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def finish_record(record: dict, backend: Backend, usage: dict, status: int, started: float) -> dict:
    """Fill in latency, tokens and cost once the backend has answered."""
    prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    record.update(
        status=status,
        latency_s=round(time.perf_counter() - started, 4),
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=query_cost(prompt, completion, backend.input_price_per_1m, backend.output_price_per_1m),
    )
    return record


def error(status: int, message: str) -> JSONResponse:
    """OpenAI-style error body."""
    return JSONResponse({"error": {"message": message, "type": "gateway_error"}}, status_code=status)


def create_app(
    config: GatewayConfig | None = None, choose: Chooser | None = None, client: httpx.AsyncClient | None = None
) -> FastAPI:
    """Build the gateway. Anything not passed in (tests pass all three) is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg, chooser, http = config, choose, client
        if cfg is None or chooser is None:
            candidates = load_candidates()
            cfg = cfg or load_config(candidates)
            chooser = chooser or load_chooser(cfg, candidates)
        http = http or httpx.AsyncClient(timeout=TIMEOUT)
        app.state.config, app.state.choose, app.state.client = cfg, chooser, http
        yield
        if client is None:
            await http.aclose()

    app = FastAPI(title="ModelPilot gateway", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        names = ["auto", *app.state.config.backends]
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "modelpilot"} for n in names]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        cfg: GatewayConfig = request.app.state.config
        try:
            body = await request.json()
            lam = float(body.pop("lambda", cfg.lam))
        except (ValueError, TypeError):
            return error(400, "body must be JSON with a numeric 'lambda' if given")

        requested = body.get("model", "auto")
        started = time.perf_counter()
        if requested == "auto":
            name = await run_in_threadpool(request.app.state.choose, last_user_text(body.get("messages", [])), lam)
        elif requested in cfg.backends:
            name = requested
        else:
            return error(400, f"model must be 'auto', 'small' or 'large', got {requested!r}")

        backend = cfg.backends[name]
        body["model"] = backend.model_id
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "requested": requested,
            "backend": name,
            "model_id": backend.model_id,
            "lambda": lam if requested == "auto" else None,
            "stream": bool(body.get("stream")),
            "route_s": round(time.perf_counter() - started, 4),
        }
        headers = {"Authorization": f"Bearer {backend.api_key}"}
        http: httpx.AsyncClient = request.app.state.client
        url = f"{backend.base_url}/chat/completions"

        if body.get("stream"):
            # Ask for token usage in the last chunk so the log has token counts.
            body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
            return await relay_stream(http, url, body, headers, backend, record, started, cfg.log_path)

        try:
            reply = await http.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            log_request(cfg.log_path, finish_record(record, backend, {}, 502, started))
            return error(502, f"{name} backend unreachable: {exc.__class__.__name__}")
        try:
            payload = reply.json()
        except ValueError:
            payload = {"error": {"message": reply.text, "type": "backend_error"}}
        log_request(cfg.log_path, finish_record(record, backend, payload.get("usage") or {}, reply.status_code, started))
        return JSONResponse(payload, status_code=reply.status_code, headers={"x-modelpilot-backend": name})

    return app


async def relay_stream(
    http: httpx.AsyncClient, url: str, body: dict, headers: dict, backend: Backend,
    record: dict, started: float, log_path: Path,
) -> Response:
    """Pass the backend's server-sent events through line by line, timing the first chunk."""
    try:
        reply = await http.send(http.build_request("POST", url, json=body, headers=headers), stream=True)
    except httpx.HTTPError as exc:
        log_request(log_path, finish_record(record, backend, {}, 502, started))
        return error(502, f"{backend.name} backend unreachable: {exc.__class__.__name__}")
    if reply.status_code != 200:
        content = await reply.aread()
        await reply.aclose()
        log_request(log_path, finish_record(record, backend, {}, reply.status_code, started))
        return Response(content, status_code=reply.status_code, media_type="application/json")

    async def events() -> AsyncIterator[str]:
        usage: dict = {}
        try:
            async for line in reply.aiter_lines():
                if line.startswith("data:") and line != "data: [DONE]":
                    record.setdefault("ttft_s", round(time.perf_counter() - started, 4))
                    try:
                        usage = json.loads(line.removeprefix("data:")).get("usage") or usage
                    except ValueError:
                        pass
                yield line + "\n"  # aiter_lines drops the newline; blank lines separate events
        finally:
            await reply.aclose()
            log_request(log_path, finish_record(record, backend, usage, 200, started))

    return StreamingResponse(events(), media_type="text/event-stream", headers={"x-modelpilot-backend": backend.name})


app = create_app()
