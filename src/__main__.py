"""Pipeline entrypoint.

Runs the current production workflow: ingest the documents into pgvector and
then evaluate the RAG pipeline with DeepEval + MLflow. This drives the 4-stage
cycle INGEST → RETRIEVE → GUARDRAILS/RAG → EVALUATE (see the module docstrings
of ``ingest``, ``retriever``, ``guardrails``/``rag``, and ``evaluate`` for each
stage's role). Retriever, RAG, and guardrails stages are importable but their
direct invocations are commented out and only exercised indirectly through
``evaluate``.
"""

import evaluate

# import guardrails
import ingest

# import rag
# import retriever


def main():
    """Run the ingestion stage followed by the evaluation stage.

    Resets the database before ingesting so each run starts from a clean state,
    then evaluates all golden-set queries across the configured models and
    temperatures, logging results to MLflow.
    """
    ingest.run_ingestion(reset_db=True)  # ingest grabs data from /data and pushes to PGVector

    # retriever.retrieval_test("What is the main topic of the text?")  # retriever handles PGVector similarity search

    # rag.test_rag("What is the main topic of the text?")  # rag connects context to ministral/qwen

    # guardrails.run_tests_sync()  # guardrails handles NeMo Guardrails logic

    evaluate.run_evaluation()  # evaluate handles DeepEval + MLflow script for reporting


if __name__ == "__main__":
    main()
