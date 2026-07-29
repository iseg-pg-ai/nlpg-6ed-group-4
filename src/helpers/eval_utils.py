import asyncio
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ValidationError

from helpers.lm_studio_utils import LMStudioModel


# CUSTOM LOCAL LLM JUDGE FOR DEEPEVAL
class LMStudioJudge(DeepEvalBaseLLM):
    """Wraps our local LM Studio model so DeepEval uses it instead of OpenAI."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.llm = LMStudioModel(model_name)

    def load_model(self) -> DeepEvalBaseLLM:
        return self

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        response_format: dict[str, Any] | None = None

        if schema is not None:
            response_format = {  # Translate Pydantic schema into OpenAI's Structured Output format
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format=response_format,
        )

        content: str = response.choices[0].message.content or ""

        if schema is not None:
            try:  # Pydantic v2 native JSON deserialization directly into the schema class
                return schema.model_validate_json(content)
            except ValidationError as err:
                print(f"[ERROR] Failed to parse LLM output into schema: {err}")
                defaults = {}
                for name, field_info in schema.model_fields.items():
                    if field_info.is_required():
                        tp = field_info.annotation
                        defaults[name] = "" if tp is str else (0.0 if tp is float else (0 if tp is int else None))
                return schema.model_construct(**defaults)

        return content

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


# Based on Slide 13 of Presentation 4
class NLPLexicalOverlapMetric(BaseMetric):
    """A traditional NLP custom scorer based on the Bag of Words (Jaccard Similarity) concept.

    Measures how many words the actual output shares with the expected golden output.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        super().__init__()
        self.threshold = threshold
        self.score: float | None = 0.0
        self.reason: str | None = ""
        self.success: bool | None = False

    def measure(self, test_case: LLMTestCase) -> float:
        expected = test_case.expected_output or ""
        actual = test_case.actual_output or ""

        # Convert to Bag of Words (lowercase, split by spaces)
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words or not actual_words:
            self.score = 0.0
        else:
            # Jaccard Similarity: (Intersection) / (Union)
            intersection = expected_words & actual_words
            union = expected_words | actual_words
            self.score = len(intersection) / len(union)

        self.success = self.score >= self.threshold
        self.reason = f"Bag-of-Words Jaccard Similarity is {self.score:.2f}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "NLP Lexical Overlap"
