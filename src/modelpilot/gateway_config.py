"""Gateway settings from .env: the two backends, their prices, the default λ and the log file."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from modelpilot.router import DEFAULT_LAMBDA

# (env prefix, xRouteBench model name, default base URL, default API model id, env var holding the key)
BACKEND_DEFAULTS = {
    "small": ("SMALL", "qwen2.5-7b-instruct", "http://localhost:8001/v1", "Qwen/Qwen2.5-7B-Instruct", "SMALL_API_KEY"),
    "large": ("LARGE", "gpt-oss-120b", "https://api.together.xyz/v1", "openai/gpt-oss-120b", "TOGETHER_API_KEY"),
}


@dataclass(frozen=True)
class Backend:
    """One OpenAI-compatible server the gateway can send a request to."""

    name: str  # "small" or "large"
    router_model: str  # model_name in xRouteBench: used for routing and for the price
    base_url: str
    model_id: str
    api_key: str
    input_price_per_1m: float
    output_price_per_1m: float


@dataclass(frozen=True)
class GatewayConfig:
    backends: dict[str, Backend]
    lam: float
    log_path: Path


def load_config(candidates: pd.DataFrame) -> GatewayConfig:
    """Read .env; prices always come from the xRouteBench candidates so live and offline costs agree."""
    load_dotenv()
    prices = candidates.set_index("model_name")
    backends = {}
    for name, (prefix, default_model, default_url, default_id, key_var) in BACKEND_DEFAULTS.items():
        router_model = os.environ.get(f"{prefix}_ROUTER_MODEL", default_model)
        if router_model not in prices.index:
            raise ValueError(f"{prefix}_ROUTER_MODEL={router_model!r} is not an xRouteBench model")
        backends[name] = Backend(
            name=name,
            router_model=router_model,
            base_url=os.environ.get(f"{prefix}_BASE_URL", default_url).rstrip("/"),
            model_id=os.environ.get(f"{prefix}_MODEL_ID", default_id),
            api_key=os.environ.get(key_var, "EMPTY"),
            input_price_per_1m=float(prices.loc[router_model, "input_price_per_1m"]),
            output_price_per_1m=float(prices.loc[router_model, "output_price_per_1m"]),
        )
    return GatewayConfig(
        backends=backends,
        lam=float(os.environ.get("ROUTER_LAMBDA", DEFAULT_LAMBDA)),
        log_path=Path(os.environ.get("GATEWAY_LOG", "logs/gateway.jsonl")),
    )
