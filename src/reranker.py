from helpers.lm_studio_utils import LMStudioModel


def rerank(query: str, chunks: list[str], model_name: str | None = None) -> list[str]:
    """Re-rank chunks by asking an LLM to score each for relevance to the query."""
    if len(chunks) <= 1:
        return chunks

    llm = LMStudioModel(model_name) if model_name else None

    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        score = _score_chunk(query, chunk, llm)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored]


def _score_chunk(query: str, chunk: str, llm: LMStudioModel | None) -> float:
    """Score a single chunk for relevance to the query on a 0.0–1.0 scale."""
    if llm is None:
        return _lexical_score(query, chunk)

    prompt = (
        f"On a scale from 0.0 to 1.0, how relevant is the following text to the query?\n\n"
        f"Query: {query}\n\n"
        f"Text: {chunk[:500]}\n\n"
        f"Return only a number between 0.0 and 1.0, nothing else."
    )

    try:
        response = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        score = float(raw)
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError, Exception):
        return _lexical_score(query, chunk)


def _lexical_score(query: str, chunk: str) -> float:
    """Fallback lexical overlap score when no LLM is available."""
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    if not query_words or not chunk_words:
        return 0.0
    intersection = query_words & chunk_words
    union = query_words | chunk_words
    return len(intersection) / len(union)
