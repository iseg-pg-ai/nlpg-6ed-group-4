"""Hybrid retrieval over pgvector.

Retrieves the most relevant document chunks for a query by combining
vector-cosine search with full-text search, with an optional LLM-based
re-ranking pass. Results are cached per ``(query, top_k, rerank_model)``.
"""

import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

import helpers.db_utils as db
from helpers.lm_studio_utils import LMStudioModelEmbedder
from reranker import rerank

load_dotenv(override=True)  # Load environment variables

_RETRIEVAL_CACHE: dict[str, list[str]] = {}


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrieverConfig:
    """Immutable, slot-optimized configuration for the retriever stage.

    Attributes:
        embedding_model_name: Model used to embed the query before searching.
        top_k: Default number of chunks to return when no explicit limit is given.
    """

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    top_k: int = int(os.getenv("TOP_K", "3"))


def retrieve_context(
    query: str,
    top_k: int | None = None,
    config: RetrieverConfig | None = None,
    rerank_model: str | None = None,
) -> list[str]:
    """Retrieve the most relevant document chunks for a query.

    Embeds the query, runs a hybrid vector + full-text search, optionally
    re-ranks the results with an LLM, and returns the top chunks. Results are
    cached keyed by ``query:top_k:rerank_model``.

    Args:
        query: The user query to search for.
        top_k: Number of chunks to return; defaults to ``config.top_k``.
        config: Retriever settings; uses environment-derived defaults if None.
        rerank_model: If provided, re-rank results with this LLM model.

    Returns:
        A list of chunk contents, most relevant first. Empty on database failure.

    Raises:
        psycopg.Error: If the hybrid search query itself fails.
    """
    cache_key = f"{query}:{top_k}:{rerank_model}"
    if cache_key in _RETRIEVAL_CACHE:
        return _RETRIEVAL_CACHE[cache_key]

    cfg = config or RetrieverConfig()
    limit = top_k if top_k is not None else cfg.top_k

    embedder = LMStudioModelEmbedder(cfg.embedding_model_name)
    query_embedding = embedder.embed(query)

    try:
        with db.get_db_connection() as conn, conn.cursor() as cur:
            results = db.search_chunks_hybrid(cur, query_embedding, query, limit)
    except psycopg.Error as err:
        print(f"[ERROR] Database failure during context retrieval: {err}")
        results = []

    if results and rerank_model:
        results = rerank(query, results, rerank_model)

    _RETRIEVAL_CACHE[cache_key] = results
    return results


def clear_cache() -> None:
    """Clear all cached retrieval results."""
    _RETRIEVAL_CACHE.clear()


def retrieval_test(test_query: str, config: RetrieverConfig | None = None) -> None:
    """Run an interactive retrieval test that prints the top contexts.

    Args:
        test_query: The query to test with.
        config: Retriever settings; uses environment-derived defaults if None.
    """
    cfg = config or RetrieverConfig()

    embedder = LMStudioModelEmbedder(cfg.embedding_model_name)
    embedder.load()

    print(f"Searching for {test_query!r}...")

    results = retrieve_context(test_query, config=cfg)

    print("\n--- Top Retrieved Contexts ---")
    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}]:\n{res}")


if __name__ == "__main__":
    retrieval_test("What is the main topic of the text?")
