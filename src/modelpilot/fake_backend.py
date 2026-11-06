"""Fake OpenAI-compatible backend: waits a fixed delay, then returns `max_tokens` dummy tokens.

Lets the gateway (and GuideLLM) run end to end without a GPU or API key.
    FAKE_DELAY_SECONDS=0.5 uv run uvicorn modelpilot.fake_backend:app --port 8001
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULT_MAX_TOKENS = 16
MAX_TOKENS_CAP = 4096
TOKEN = "token "


def create_fake_backend(delay_seconds: float = 0.5) -> FastAPI:
    """App answering /v1/chat/completions after `delay_seconds` (time to first token when streaming)."""
    app = FastAPI(title="Fake OpenAI backend")

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": "fake", "object": "model", "owned_by": "modelpilot"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        n = min(int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS), MAX_TOKENS_CAP)
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}
        base = {"id": "fake-1", "created": int(time.time()), "model": body.get("model", "fake")}
        await asyncio.sleep(delay_seconds)

        if not body.get("stream"):
            message = {"role": "assistant", "content": TOKEN * n}
            return JSONResponse({**base, "object": "chat.completion", "usage": usage,
                                 "choices": [{"index": 0, "message": message, "finish_reason": "length"}]})

        def chunk(choices: list, **extra) -> str:
            return f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', 'choices': choices, **extra})}\n\n"

        async def events() -> AsyncIterator[str]:
            for _ in range(n):
                yield chunk([{"index": 0, "delta": {"content": TOKEN}, "finish_reason": None}])
            yield chunk([{"index": 0, "delta": {}, "finish_reason": "length"}])
            if (body.get("stream_options") or {}).get("include_usage"):
                yield chunk([], usage=usage)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_fake_backend(float(os.environ.get("FAKE_DELAY_SECONDS", "0.5")))
