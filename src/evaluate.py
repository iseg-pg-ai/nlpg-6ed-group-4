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


async def generate_batch_answers(pipeline: GuardrailsPipeline, dataset: list[dict[str, Any]]) -> list[BatchItem]:
    """Generate answers sequentially through the guardrails pipeline."""
    results: list[BatchItem] = []
    for item in dataset:
        query = str(item["query"])
        expected = str(item["expected_output"])
        guardrails_res = await pipeline.ask(query)

        results.append({
            "query": query,
            "expected_output": expected,
            "answer": guardrails_res.get("answer", ""),
            "context": guardrails_res.get("context", []),
        })
    return results


def _run_generation_phase(model: str, run_name: str) -> list[BatchItem]:
    """Handles loading the generator model and running the batch prompts."""
    print(f"\n{'=' * 40}\nPHASE 1: GENERATING ANSWERS FOR {run_name}\n{'=' * 40}")
    gen_model = LMStudioModel(model)
    gen_model.load()

    pipeline = GuardrailsPipeline(model)
    batch_results = asyncio.run(generate_batch_answers(pipeline, EVALUATION_DATA))

    gen_model.unload()
    return batch_results


def _score_single_case(
    data: BatchItem,
    f_metric: FaithfulnessMetric,
    r_metric: AnswerRelevancyMetric,
    o_metric: NLPLexicalOverlapMetric,
) -> dict[str, Any]:
    """Scores a single batch item and returns detailed metrics."""
    test_case = LLMTestCase(
        input=data["query"],
        actual_output=data["answer"],
        expected_output=data["expected_output"],
        retrieval_context=cast(Any, data["context"]),
    )

    f_score, r_score, o_score = 0.0, 0.0, 0.0

    try:
        print(f"Scoring {data['query']!r}...")
        f_metric.measure(test_case)
        r_metric.measure(test_case)
        o_metric.measure(test_case)

        f_score = f_metric.score or 0.0
        r_score = r_metric.score or 0.0
        o_score = o_metric.score or 0.0
    except Exception as err:
        print(f"[WARNING] DeepEval scoring failed: {err}")

    return {
        "Query": data["query"],
        "Expected Output": data["expected_output"],
        "Actual LLM Answer": data["answer"],
        "Faithfulness Score": f_score,
        "Faithfulness Reason": getattr(f_metric, "reason", ""),
        "Relevancy Score": r_score,
        "Relevancy Reason": getattr(r_metric, "reason", ""),
        "Overlap Score": o_score,
        "Overlap Reason": getattr(o_metric, "reason", ""),
    }


def _run_scoring_phase(
    batch_results: list[BatchItem], judge_model_name: str, run_name: str
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Handles loading the judge model and scoring the generated batch results."""
    print(f"\n{'=' * 40}\nPHASE 2: EVALUATING {run_name} WITH JUDGE: {judge_model_name}\n{'=' * 40}")

    judge_model = LMStudioModel(judge_model_name)
    judge_model.load()

    local_judge = LMStudioJudge(model_name=judge_model_name)
    f_metric = FaithfulnessMetric(threshold=0.5, model=local_judge)
    r_metric = AnswerRelevancyMetric(threshold=0.5, model=local_judge)
    o_metric = NLPLexicalOverlapMetric(threshold=0.3)

    detailed_results = []
    totals = {"faithfulness": 0.0, "relevancy": 0.0, "overlap": 0.0}

    # The loop is now incredibly clean!
    for data in batch_results:
        details = _score_single_case(data, f_metric, r_metric, o_metric)
        detailed_results.append(details)

        totals["faithfulness"] += details["Faithfulness Score"]
        totals["relevancy"] += details["Relevancy Score"]
        totals["overlap"] += details["Overlap Score"]

    judge_model.unload()

    success_count = len(batch_results)
    averages = {k: (v / success_count) if success_count > 0 else 0.0 for k, v in totals.items()}
    return detailed_results, averages


def _log_to_mlflow(
    run_name: str, params: dict[str, Any], detailed_results: list[dict[str, Any]], averages: dict[str, float]
) -> None:
    """Logs the aggregated metrics and detailed evaluation table to MLflow."""
    with mlflow.start_run(run_name=run_name):
        # We log the grouped params dictionary, saving argument space!
        mlflow.log_params(params)

        mlflow.log_metric("avg_faithfulness", averages["faithfulness"])
        mlflow.log_metric("avg_relevancy", averages["relevancy"])
        mlflow.log_metric("avg_overlap", averages["overlap"])

        if detailed_results:
            columns = detailed_results[0].keys()
            table_data = {col: [row[col] for row in detailed_results] for col in columns}
            mlflow.log_table(data=table_data, artifact_file="detailed_evaluation.json")

        print(
            f"Result for {run_name}: "
            f"Faithfulness={averages['faithfulness']:.4f}, "
            f"Relevancy={averages['relevancy']:.4f}, "
            f"Overlap={averages['overlap']:.4f}"
        )


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

            # Phase 1
            batch_results = _run_generation_phase(model, run_name)

            # Phase 2
            detailed_results, averages = _run_scoring_phase(batch_results, cfg.judge_model, run_name)

            # Phase 3
            params = {"generator_model": model, "judge_model": cfg.judge_model, "temperature": temp}
            _log_to_mlflow(run_name, params, detailed_results, averages)

    print("\nEvaluation Complete! Open http://127.0.0.1:5000 in your browser.")


if __name__ == "__main__":
    run_evaluation()
