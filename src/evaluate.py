import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import mlflow
import requests
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from golden_set import EVALUATION_DATA
from guardrails import GuardrailsPipeline
from helpers.eval_utils import LMStudioJudge, NLPLexicalOverlapMetric
from helpers.lm_studio_utils import LMStudioModel, LMStudioModelEmbedder

load_dotenv()


class BatchItem(TypedDict):
    """Structured result dictionary for Phase 1 generation output."""

    query: str
    expected_output: str
    answer: str
    context: list[str | Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalConfig:
    """Immutable, slot-optimized configuration for the evaluation pipeline."""

    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_a: str = os.getenv("MODEL_A", "ministral-3-3b-instruct-2512")
    model_b: str = os.getenv("MODEL_B", "qwen3.5-2b")
    judge_model: str = os.getenv("JUDGE_MODEL", "llama-3.2-1b-instruct@q6_k")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    temperatures: tuple[float, ...] = (0.0, 0.5)
    experiment_name: str = "NLPG_RAG_Evaluation"

    @property
    def models(self) -> tuple[str, ...]:
        return (self.model_a, self.model_b)


def check_mlflow_server(tracking_uri: str) -> None:
    """Ping the MLflow server to ensure it is alive before running evaluations."""
    print(f"[SYSTEM] Checking MLflow server at {tracking_uri!r}...")
    try:
        requests.get(tracking_uri, timeout=2.0)
    except requests.exceptions.RequestException:
        print("\n" + "=" * 60)
        print("[FATAL ERROR] MLflow Tracking Server is offline!")
        print("=" * 60)
        print("DeepEval cannot log metrics because MLflow is not running.")
        print("\nPlease open a new terminal window and run this command:")
        print("uvx mlflow server --host 127.0.0.1 --port 5000")
        print("\nThen try running main.py again.")
        print("=" * 60 + "\n")
        sys.exit(1)


async def generate_batch_answers(
    pipeline: GuardrailsPipeline,
    dataset: list[dict[str, Any]],
) -> list[BatchItem]:
    """Generate answers sequentially through the guardrails pipeline within a single event loop."""
    results: list[BatchItem] = []

    for item in dataset:
        query = str(item["query"])
        expected_output = str(item["expected_output"])

        guardrails_res = await pipeline.ask(query)

        results.append({
            "query": query,
            "expected_output": expected_output,
            "answer": guardrails_res.get("answer", ""),
            "context": guardrails_res.get("context", []),
        })

    return results


def run_evaluation(config: EvalConfig | None = None) -> None:
    """Orchestrate Phase 1 generation and Phase 2 LLM-as-a-Judge evaluation with MLflow logging."""
    cfg = config or EvalConfig()
    check_mlflow_server(cfg.tracking_uri)

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    print(f"\nStarting Automated MLflow Evaluation with {len(EVALUATION_DATA)} queries...\n")

    LMStudioModel.unload_all()

    embedder = LMStudioModelEmbedder(cfg.embedding_model)
    embedder.load()

    for model in cfg.models:
        for temp in cfg.temperatures:
            run_name = f"{model.split('/')[-1]}_temp_{temp}"

            # ==========================================
            # PHASE 1: GENERATION
            # ==========================================
            print(f"\n{'=' * 40}\nPHASE 1: GENERATING ANSWERS FOR {run_name}\n{'=' * 40}")

            gen_model = LMStudioModel(model)
            gen_model.load()

            pipeline = GuardrailsPipeline(model)
            batch_results = asyncio.run(generate_batch_answers(pipeline, EVALUATION_DATA))

            gen_model.unload()  # Free VRAM before launching judge model

            # ==========================================
            # PHASE 2: EVALUATION (LLM-as-a-Judge)
            # ==========================================
            print(f"\n{'=' * 40}\nPHASE 2: EVALUATING {run_name} WITH JUDGE: {cfg.judge_model}\n{'=' * 40}")

            judge_model = LMStudioModel(cfg.judge_model)
            judge_model.load()

            local_judge = LMStudioJudge(model_name=cfg.judge_model)
            faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=local_judge)
            relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=local_judge)
            overlap_metric = NLPLexicalOverlapMetric(threshold=0.3)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("generator_model", model)
                mlflow.log_param("judge_model", cfg.judge_model)
                mlflow.log_param("temperature", temp)

                total_faithfulness = 0.0
                total_relevancy = 0.0
                total_overlap = 0.0
                successful_evals = 0
                detailed_results: list[dict[str, Any]] = []

                for data in batch_results:
                    test_case = LLMTestCase(
                        input=data["query"],
                        actual_output=data["answer"],
                        expected_output=data["expected_output"],
                        retrieval_context=cast(Any, data["context"]),
                    )

                    try:
                        print(f"Scoring {data['query']!r}...")
                        faithfulness_metric.measure(test_case)
                        relevancy_metric.measure(test_case)
                        overlap_metric.measure(test_case)

                        f_score = faithfulness_metric.score or 0.0
                        r_score = relevancy_metric.score or 0.0
                        o_score = overlap_metric.score or 0.0

                        total_faithfulness += f_score
                        total_relevancy += r_score
                        total_overlap += o_score
                        successful_evals += 1

                        detailed_results.append({
                            "Query": data["query"],
                            "Expected Output": data["expected_output"],
                            "Actual LLM Answer": data["answer"],
                            "Faithfulness Score": f_score,
                            "Faithfulness Reason": faithfulness_metric.reason,
                            "Relevancy Score": r_score,
                            "Relevancy Reason": relevancy_metric.reason,
                            "Overlap Score": o_score,
                            "Overlap Reason": overlap_metric.reason,
                        })

                    except Exception as err:
                        print(f"[WARNING] DeepEval scoring failed: {err}")

                # Aggregate and log averages
                avg_faithfulness = 0.0
                avg_relevancy = 0.0
                avg_overlap = 0.0

                if successful_evals > 0:
                    avg_faithfulness = total_faithfulness / successful_evals
                    avg_relevancy = total_relevancy / successful_evals
                    avg_overlap = total_overlap / successful_evals

                    mlflow.log_metric("avg_faithfulness", avg_faithfulness)
                    mlflow.log_metric("avg_relevancy", avg_relevancy)
                    mlflow.log_metric("avg_overlap", avg_overlap)

                    if detailed_results:
                        columns = detailed_results[0].keys()
                        table_data = {col: [row[col] for row in detailed_results] for col in columns}
                        mlflow.log_table(
                            data=table_data,
                            artifact_file="detailed_evaluation.json",
                        )

                print(
                    f"Result for {run_name}: "
                    f"Faithfulness={avg_faithfulness:.4f}, "
                    f"Relevancy={avg_relevancy:.4f}, "
                    f"Overlap={avg_overlap:.4f}"
                )

            judge_model.unload()  # Free VRAM for the next generator model iteration

    print("\nEvaluation Complete! Open http://127.0.0.1:5000 in your browser.")


if __name__ == "__main__":
    run_evaluation()
