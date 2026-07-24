import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Self

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql

load_dotenv()

type Vector = Sequence[float] | list[float]  # PEP 695 Native Type Alias (Python 3.12+)


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

SEARCH_SQL: Final = """
SELECT content 
FROM document_chunks 
ORDER BY embedding <=> %s::vector 
LIMIT %s;
"""


def get_db_connection(config: DatabaseConfig | None = None) -> psycopg.Connection:
    """Establish a connection to the PostgreSQL database using psycopg v3."""
    cfg = config or DatabaseConfig.from_env()
    conn = psycopg.connect(cfg.conninfo, autocommit=False)
    register_vector(conn)

    return conn


def setup_database(conn: psycopg.Connection, vector_dim: int | None = None) -> None:
    """Create the pgvector extension and table schema if they do not exist."""
    dim = vector_dim or DatabaseConfig.from_env().vector_dim

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Safe query composition for table attributes
        query = sql.SQL(CREATE_TABLE_SQL).format(dim=sql.Literal(dim))
        cur.execute(query)

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


def search_chunks(cur: psycopg.Cursor, query_embedding: Vector, top_k: int) -> list[str]:
    """Perform a cosine distance search (`<=>`) and return top_k chunk contents."""
    cur.execute(SEARCH_SQL, (query_embedding, top_k))
    return [row[0] for row in cur.fetchall()]


def is_file_ingested(conn: psycopg.Connection, file_name: str) -> bool:
    """Check if chunks from the given file already exist in the database."""
    with conn.cursor() as cur:
        cur.execute(CHECK_FILE_SQL, (file_name,))
        result = cur.fetchone()
        return bool(result[0]) if result else False
