"""FastAPI app: POST /route picks a model, POST /chat routes and calls it on NVIDIA.

Run from the repo root (the embedding cache lives in ./cache):
    uv run uvicorn modelpilot.api:app
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass

import openai
import pandas as pd
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from modelpilot.cost import add_cost
from modelpilot.data import load_candidates, load_split
from modelpilot.embeddings import embed, load_encoder, split_embeddings
from modelpilot.nvidia import available_model_ids, chat, make_client
from modelpilot.router import KNNRouter, pick_models
from modelpilot.serving import choose_served_model
from modelpilot.summary import model_summary

DEFAULT_LAMBDA = 1.0  # chosen on validation (Phase 8)
KNN_K = 20  # chosen on validation (Phase 8)


@dataclass
class AppState:
    """Everything loaded once at startup and shared by all requests."""

    candidates: pd.DataFrame
    summary: pd.DataFrame
    encoder: SentenceTransformer
    router: KNNRouter
    client: OpenAI
    available_ids: set[str]


def load_state() -> AppState:
    """Load data, embeddings, router and NVIDIA client (takes a few seconds)."""
    candidates = load_candidates()
    train = add_cost(load_split("train"), candidates)
    encoder = load_encoder()
    client = make_client()
    return AppState(
        candidates=candidates,
        summary=model_summary(train),
        encoder=encoder,
        router=KNNRouter.from_results(train, split_embeddings("train", train, encoder), k=KNN_K),
        client=client,
        available_ids=available_model_ids(client),
    )


state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load state on startup."""
    global state
    state = load_state()
    yield


app = FastAPI(title="ModelPilot", lifespan=lifespan)


class RouteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=1)
    lam: float = Field(DEFAULT_LAMBDA, alias="lambda", ge=0, description="cost weight")


class ChatRequest(RouteRequest):
    max_tokens: int = Field(512, ge=1, le=4096)


class RouteResponse(BaseModel):
    model: str
    service: str
    api_model_id: str
    predicted_score: float
    predicted_cost_per_1k_queries: float


class ChatResponse(BaseModel):
    routed_model: str
    served_model: str
    api_model_id: str
    fallback: bool
    response: str


def route_query(query: str, lam: float) -> RouteResponse:
    """Embed one query and pick the model with the best predicted score - lam * cost."""
    vector = embed([query], state.encoder)
    predicted_score, predicted_cost = state.router.predict(vector)
    model = pick_models(predicted_score, predicted_cost, lam, state.router.models)[0]
    i = state.router.models.index(model)
    info = state.candidates.set_index("model_name").loc[model]
    return RouteResponse(
        model=model,
        service=info["service"],
        api_model_id=info["api_model_id"],
        predicted_score=predicted_score[0, i],
        predicted_cost_per_1k_queries=predicted_cost[0, i],
    )


@app.post("/route")
def route(request: RouteRequest) -> RouteResponse:
    """Return the model the router would pick for this query."""
    return route_query(request.query, request.lam)


@app.post("/chat")
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Route the query, then answer it with that model on NVIDIA (or the best available fallback)."""
    routed = route_query(request.query, request.lam)
    served = choose_served_model(routed.model, state.candidates, state.summary, state.available_ids)
    try:
        reply = chat(state.client, served.api_model_id, request.query, request.max_tokens)
    except openai.APIError as error:
        raise HTTPException(status_code=502, detail=f"NVIDIA API error: {error.__class__.__name__}")
    return ChatResponse(
        routed_model=routed.model,
        served_model=served.model_name,
        api_model_id=served.api_model_id,
        fallback=served.fallback,
        response=reply,
    )
