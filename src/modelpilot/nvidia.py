"""Thin client for NVIDIA's OpenAI-compatible API."""

import os

from dotenv import load_dotenv
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 1


def make_client() -> OpenAI:
    """OpenAI client pointed at NVIDIA, with the key read from .env (never logged)."""
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set; add it to .env")
    return OpenAI(
        base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES
    )


def available_model_ids(client: OpenAI) -> set[str]:
    """Model ids NVIDIA currently serves (GET /v1/models)."""
    return {model.id for model in client.models.list()}


def chat(client: OpenAI, api_model_id: str, prompt: str, max_tokens: int) -> str:
    """Send one user message and return the reply text."""
    completion = client.chat.completions.create(
        model=api_model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""
