import atexit
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Self

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg_pool import ConnectionPool

load_dotenv(override=True)  # Load environment variables

type Vector = Sequence[float] | list[float]  # PEP 695 Native Type Alias (Python 3.12+)

_pool: ConnectionPool | None = None


def _close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


atexit.register(_close_pool)


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        cfg = DatabaseConfig.from_env()
        with psycopg.connect(cfg.conninfo, autocommit=True) as tmp_conn, tmp_conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        _pool = ConnectionPool(
            cfg.conninfo,
            min_size=1,
            max_size=4,
            open=True,
            configure=lambda conn: register_vector(conn),
            kwargs={"autocommit": False},
        )
    return _pool


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Immutable, slot-optimized database configuration."""

    user: str | None = os.getenv("POSTGRES_USER")
    password: str | None = os.getenv("POSTGRES_PASSWORD")
    host: str | None = os.getenv("POSTGRES_HOST")
    port: str | None = os.getenv("POSTGRES_PORT")
    dbname: str | None = os.getenv("POSTGRES_DB")
    vector_dim: int = int(os.getenv("VECTOR_DIM", "384"))

    @classmethod
    def from_env(cls) -> Self:
        return cls()

    @property
    def conninfo(self) -> str:
        """Generate a libpq-compatible connection string."""
        opts = {
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
        }
        return " ".join(f"{k}={v}" for k, v in opts.items() if v is not None)


# --- SQL Statements ---
CREATE_TABLE_SQL: Final = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    source_file VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dim})
);
"""

INSERT_SQL: Final = """
INSERT INTO document_chunks (source_file, content, embedding)
VALUES (%s, %s, %s::vector);
"""

DROP_SQL: Final = "DROP TABLE IF EXISTS document_chunks CASCADE;"

CHECK_FILE_SQL: Final = """
SELECT EXISTS(
    SELECT 1 FROM document_chunks WHERE source_file = %s
);
"""

CREATE_INDEX_SQL: Final = """
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

ADD_TSVECTOR_SQL: Final = """
ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
"""

CREATE_FTS_INDEX_SQL: Final = """
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
ON document_chunks
USING GIN (content_tsv);
"""

SEARCH_VECTOR_SQL: Final = """
SELECT content, embedding <=> %s::vector AS distance
FROM document_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

SEARCH_FTS_SQL: Final = """
SELECT content, ts_rank(content_tsv, plainto_tsquery('english', %s)) AS score
FROM document_chunks
WHERE content_tsv @@ plainto_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s;
"""


@contextmanager
def get_db_connection(config: DatabaseConfig | None = None) -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool."""
    if config is not None:
        conn = psycopg.connect(config.conninfo)
        register_vector(conn)
        try:
            yield conn
        finally:
            conn.close()
    else:
        with _get_pool().connection() as conn:
            yield conn


def setup_database(conn: psycopg.Connection, vector_dim: int | None = None) -> None:
    """Create the pgvector extension and table schema if they do not exist."""
    dim = vector_dim or DatabaseConfig.from_env().vector_dim

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        query = sql.SQL(CREATE_TABLE_SQL).format(dim=sql.Literal(dim))
        cur.execute(query)
        cur.execute(CREATE_INDEX_SQL)
        cur.execute(ADD_TSVECTOR_SQL)
        cur.execute(CREATE_FTS_INDEX_SQL)

    conn.commit()
    print("[SYSTEM] Database schema ensured.")


def clear_database(conn: psycopg.Connection) -> None:
    """Drop the document_chunks table to start fresh."""
    with conn.cursor() as cur:
        cur.execute(DROP_SQL)

    conn.commit()
    print("[SYSTEM] Database table dropped (cleared).")


def insert_chunk(cur: psycopg.Cursor, file_name: str, chunk: str, embedding: Vector) -> None:
    """Insert a single document chunk and its vector embedding."""
    cur.execute(INSERT_SQL, (file_name, chunk, embedding))


def insert_chunks_batch(cur: psycopg.Cursor, records: Sequence[tuple[str, str, Vector]]) -> None:
    """Batch-insert multiple document chunks for higher throughput."""
    cur.executemany(INSERT_SQL, records)


def search_chunks_vector(cur: psycopg.Cursor, query_embedding: Vector, top_k: int) -> list[tuple[str, float]]:
    """Perform a cosine distance search (`<=>`) and return (content, distance) pairs."""
    cur.execute(SEARCH_VECTOR_SQL, (query_embedding, query_embedding, top_k))
    return [(row[0], row[1]) for row in cur.fetchall()]


def search_chunks_fts(cur: psycopg.Cursor, query_text: str, top_k: int) -> list[tuple[str, float]]:
    """Perform a full-text search and return (content, score) pairs."""
    cur.execute(SEARCH_FTS_SQL, (query_text, query_text, top_k))
    return [(row[0], row[1]) for row in cur.fetchall()]


def search_chunks_hybrid(
    cur: psycopg.Cursor,
    query_embedding: Vector,
    query_text: str,
    top_k: int,
    fts_k: int | None = None,
) -> list[str]:
    """Hybrid search combining vector cosine distance and full-text search via RRF."""
    vec_k = top_k
    fts_k = fts_k or top_k

    vec_results = search_chunks_vector(cur, query_embedding, vec_k)
    fts_results = search_chunks_fts(cur, query_text, fts_k)

    content_scores: dict[str, float] = {}
    for rank, (content, _) in enumerate(vec_results):
        content_scores[content] = content_scores.get(content, 0.0) + 1.0 / (60 + rank)

    for rank, (content, _) in enumerate(fts_results):
        content_scores[content] = content_scores.get(content, 0.0) + 1.0 / (60 + rank)

    ranked = sorted(content_scores, key=content_scores.__getitem__, reverse=True)
    return ranked[:top_k]


def is_file_ingested(conn: psycopg.Connection, file_name: str) -> bool:
    """Check if chunks from the given file already exist in the database."""
    with conn.cursor() as cur:
        cur.execute(CHECK_FILE_SQL, (file_name,))
        result = cur.fetchone()
        return bool(result[0]) if result else False
