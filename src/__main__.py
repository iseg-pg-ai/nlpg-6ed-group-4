import evaluate

# import guardrails
import ingest

# import rag
# import retriever


def main():
    ingest.run_ingestion(reset_db=True)  # ingest grabs data from /data and pushes to PGVector

    # retriever.retrieval_test("What is the main topic of the text?")  # retriever handles PGVector similarity search

    # rag.test_rag("What is the main topic of the text?")  # rag connects context to ministral/qwen

    # guardrails.run_tests_sync()  # guardrails handles NeMo Guardrails logic

    evaluate.run_evaluation()  # evaluate handles DeepEval + MLflow script for reporting


if __name__ == "__main__":
    main()
