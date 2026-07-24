import evaluate as evaluate
import guardrails as guardrails
import ingest as ingest
import rag as rag
import retriever as retriever


def main():
    # ingest grabs data from /data and pushes to PGVector
    ingest.run_ingestion(reset_db=True)

    # retriever handles PGVector similarity search
    retriever.retrieval_test("What is the main topic of the text?")

    # rag connects context to ministral/qwen
    rag.test_rag("What is the main topic of the text?")

    # guardrails handles NeMo Guardrails logic
    guardrails.run_tests_sync()

    # evaluate handles DeepEval + MLflow script for reporting
    evaluate.run_evaluation()


if __name__ == "__main__":
    main()
