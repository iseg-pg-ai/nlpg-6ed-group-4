"""Document ingestion into pgvector.

**Cycle role — Stage 1 of 4 (INGEST).** Reads PDF/TXT documents from the data
directory, splits them into sentence-aligned chunks, embeds the chunks in
batches via LM Studio, and stores them in the pgvector database. Includes a
failsafe for huge unpunctuated blocks commonly found in PDFs.

This stage produces the corpus that every downstream stage consumes:
``document_chunks`` rows (source_file + content + embedding) queried later by
the retriever. The cycle continues with RETRIEVE (``retriever.py``).
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pypdf import PdfReader  # Modern, actively maintained successor to PyPDF2

import helpers.db_utils as db
from helpers.lm_studio_utils import LMStudioModelEmbedder

load_dotenv(override=True)  # Load environment variables

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Immutable, slot-optimized ingestion configuration.

    Attributes:
        data_dir: Directory containing the source documents.
        embedding_model_name: Embedding model used to vectorize chunks.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Number of characters carried over between chunks.
    """

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Split text into sentence-aligned chunks with overlap.

    Each chunk contains only whole sentences, never truncated mid-word. Massive
    unpunctuated blocks (a single "sentence" longer than ``chunk_size``) are
    sliced by characters as a failsafe.

    Args:
        text: The raw document text to split.
        chunk_size: Maximum chunk length in characters.
        overlap: Number of characters shared with the previous chunk.

    Yields:
        Consecutive chunks of the input text.

    Raises:
        ValueError: If ``overlap`` is not strictly less than ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    sentences = _SENTENCE_PATTERN.split(text.replace("\n", " "))
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return

    pending: list[str] = []
    pending_len = 0

    for sentence in sentences:
        # PDF FAILSAFE: Catch massive "sentences"
        if len(sentence) > chunk_size:
            if pending:  # Flush any good sentences we were holding
                yield " ".join(pending)
                pending.clear()
                pending_len = 0

            # Brutally slice massive strings by characters
            for i in range(0, len(sentence), chunk_size - overlap):
                yield sentence[i : i + chunk_size]
            continue

        sentence_len = len(sentence) + 1

        if pending and pending_len + sentence_len > chunk_size:
            yield " ".join(pending)

            # Build overlap from trailing sentences of the yielded chunk
            overlap_buf: list[str] = []
            overlap_chars = 0
            for s in reversed(pending):
                s_len = len(s) + 1
                if overlap_chars + s_len > overlap:
                    break
                overlap_buf.insert(0, s)
                overlap_chars += s_len

            pending = overlap_buf
            pending_len = overlap_chars

        pending.append(sentence)
        pending_len += sentence_len

    if pending:
        yield " ".join(pending)


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract all text from a PDF file, page by page.

    Joins the extracted page texts with newlines to avoid merging words across
    page boundaries. On failure, logs the error and returns an empty string.

    Args:
        file_path: Path to the PDF file to read.

    Returns:
        The concatenated page text, or an empty string if the file cannot be
        read or contains no extractable text.
    """
    try:
        reader = PdfReader(file_path)
        # Join extracted page text with newlines to avoid merged word boundaries
        return "\n".join(page_text for page in reader.pages if (page_text := page.extract_text()))
    except (OSError, ValueError) as err:
        print(f"[ERROR] Failed reading PDF {file_path.name!r}: {err}")
        return ""


def run_ingestion(reset_db: bool = True, config: IngestConfig | None = None) -> None:
    """Read, chunk, embed, and ingest text/PDF documents into PostgreSQL.

    Loads the embedder once, optionally clears the database, ensures the schema
    exists, and processes every supported file in the data directory. Files
    already ingested are skipped unless the database is reset.

    Args:
        reset_db: If True, drop the existing table before ingesting.
        config: Ingestion settings; uses environment-derived defaults if None.
    """
    cfg = config or IngestConfig()
    print("Starting ingestion process...")

    if not cfg.data_dir.exists():
        print(f"[ERROR] Data directory {cfg.data_dir!r} does not exist.")
        return

    supported_extensions: Final = {".txt", ".pdf"}
    all_files = [
        path for path in cfg.data_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_extensions
    ]

    if not all_files:
        print(f"[SYSTEM] No .txt or .pdf files found in {cfg.data_dir!r}.")
        return

    # Load the embedder ONCE for the entire ingestion run
    embedder = LMStudioModelEmbedder(cfg.embedding_model_name)
    embedder.load()

    with db.get_db_connection() as conn:
        if reset_db:
            # Fresh start: drop any previous corpus so the run is reproducible.
            db.clear_database(conn)

        # Ensure the schema + indexes exist before writing any chunk.
        db.setup_database(conn)

        with conn.cursor() as cur:
            for file_path in all_files:
                file_name = file_path.name

                # Incremental mode: skip files already present in the corpus.
                if not reset_db and db.is_file_ingested(conn, file_name):
                    print(f"Skipping {file_name!r}: Already ingested.")
                    continue

                print(f"Processing {file_name!r}...")

                # STAGE 1a: EXTRACT — raw document → plain text.
                try:
                    match file_path.suffix.lower():
                        case ".pdf":
                            text = extract_text_from_pdf(file_path)
                        case ".txt":
                            text = file_path.read_text(encoding="utf-8", errors="replace")
                        case _:
                            print(f"Skipping unsupported file type: {file_name!r}")
                            continue
                except (OSError, ValueError, UnicodeDecodeError) as err:
                    print(f"[ERROR] Could not read file {file_name!r}: {err}")
                    continue

                if not text.strip():
                    print(f"[WARNING] No readable text found in {file_name!r}. Skipping.")
                    continue

                # STAGE 1b: CHUNK — text → sentence-aligned chunks (~500 chars).
                chunks = [c for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap) if c.strip()]

                if not chunks:
                    continue

                # STAGE 1c: EMBED — all chunks in one batched API call.
                embeddings = embedder.embed_batch(chunks)
                # STAGE 1d: PERSIST — (source_file, chunk, embedding) → pgvector.
                records = [(file_name, chunk, emb) for chunk, emb in zip(chunks, embeddings)]
                db.insert_chunks_batch(cur, records)

            conn.commit()

    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion(reset_db=True)
