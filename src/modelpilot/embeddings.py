"""Turn query text into vectors with sentence-transformers, cached on disk."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("cache")

# Every query is "[System Instruction]\n<task template>\n\n[User Query]\n<question>".
USER_QUERY_MARKER = "[User Query]\n"
# Labels that open the question part, e.g. "## Question:\n", "Question: ", "Task: ".
LEADING_LABEL = re.compile(r"^\s*(?:##\s*Question:|Question:|Task:|Complete the following function:)\s*")


def load_encoder() -> SentenceTransformer:
    """Load the MiniLM sentence encoder on CPU (downloads once, ~90 MB)."""
    return SentenceTransformer(ENCODER_NAME, device="cpu")


def question_text(query: str) -> str:
    """Strip the shared instruction template and leading labels, keeping only the question.

    Options, passages and code stay: they are part of the question. Text without the
    marker is returned with only labels stripped.
    """
    question = query.split(USER_QUERY_MARKER, 1)[-1]
    while (stripped := LEADING_LABEL.sub("", question, count=1)) != question:
        question = stripped
    return question.strip()


def embed(texts: list[str], encoder: SentenceTransformer) -> np.ndarray:
    """Encode texts into unit-length vectors, shape (len(texts), 384)."""
    return encoder.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )


def query_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per query (embedding_id, query), sorted by embedding_id."""
    queries = results[["embedding_id", "query"]].drop_duplicates("embedding_id")
    return queries.sort_values("embedding_id").reset_index(drop=True)


def split_embeddings(split: str, results: pd.DataFrame, encoder: SentenceTransformer) -> np.ndarray:
    """Embeddings for every query in a split, row i = i-th query of query_table()."""
    path = CACHE_DIR / f"{split}_embeddings.npy"
    if path.exists():
        return np.load(path)
    questions = [question_text(q) for q in query_table(results)["query"]]
    vectors = embed(questions, encoder)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, vectors)
    return vectors
