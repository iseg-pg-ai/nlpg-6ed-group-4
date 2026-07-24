import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

import helpers.db_utils as db
from helpers.lm_studio_utils import LMStudioModelEmbedder

load_dotenv()  # Load environment variables


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrieverConfig:
    """Immutable, slot-optimized configuration for the context retriever."""

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    top_k: int = int(os.getenv("TOP_K", "3"))


def retrieve_context(query: str, top_k: int | None = None, config: RetrieverConfig | None = None) -> list[str]:
    """Convert a query into an embedding and retrieve matching context chunks from PostgreSQL."""
    cfg = config or RetrieverConfig()
    limit = top_k if top_k is not None else cfg.top_k

    embedder = LMStudioModelEmbedder(cfg.embedding_model_name)
    query_embedding = embedder.embed(query)

    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                return db.search_chunks(cur, query_embedding, limit)
    except psycopg.Error as err:
        print(f"[ERROR] Database failure during context retrieval: {err}")
        return []


def retrieval_test(test_query: str, config: RetrieverConfig | None = None) -> None:
    """Test the retrieval pipeline with a sample query."""
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
