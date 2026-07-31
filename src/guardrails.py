"""NeMo Guardrails integration.

**Cycle role — Stage 3 of 4 (GUARDRAILS).** Wraps NeMo Guardrails with Colang
rules defined in ``nemo_config/`` and delegates valid EMAR questions to the RAG
pipeline. A temporary config directory is generated with environment variables
substituted inline so that the global ``os.environ`` is never mutated, and
``fastembed`` is required at runtime for embeddings-only intent matching.

Pipeline trace: sits between RETRIEVE (``retriever.py``) and EVALUATE
(``evaluate.py``). Each ``ask()`` classifies the user intent against the guardrail
flows; a blocked intent is answered directly, otherwise the ``run_rag_action``
executes the RAG stage and the answer + retrieved context are returned to the
caller.
"""

import asyncio
import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, TypedDict

from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig

import rag
from helpers.lm_studio_utils import LMStudioModel, LMStudioModelEmbedder

load_dotenv(override=True)  # Load environment variables


class GuardrailsResult(TypedDict):
    """Structured response type for guardrails execution.

    Attributes:
        answer: The final bot reply, either from a guardrail flow or the RAG action.
        context: The retrieved chunks captured from the underlying RAG call.
    """

    answer: str
    context: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardrailsConfig:
    """Immutable, slot-optimized configuration for NeMo Guardrails.

    Attributes:
        base_url: Base URL of the LM Studio OpenAI-compatible endpoint.
        api_key: API key expected by the OpenAI client.
        test_model: Model used for guardrails/generation tests.
        embed_model: Embedding model used by NeMo for intent matching.
        config_dir: Directory containing ``config.yml`` and ``rails.co``.
    """

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.getenv("LLM_API_KEY", "local-dummy-key")
    test_model: str = os.getenv("MODEL_A", "ministral-3-3b-instruct-2512")
    embed_model: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    config_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "nemo_config")

    @classmethod
    def from_env(cls) -> Self:
        """Instantiate configuration directly from environment variables.

        Returns:
            A :class:`GuardrailsConfig` populated from the current environment.
        """
        return cls()

    @property
    def api_base(self) -> str:
        """Strip the trailing ``/v1`` suffix to get the REST root URL.

        Returns:
            The REST root URL used for model load/unload endpoints.
        """
        return self.base_url.removesuffix("/v1")


class GuardrailsPipeline:
    """Encapsulates NeMo Guardrails initialization, custom actions, and query state.

    Builds a NeMo :class:`LLMRails` app from the ``nemo_config`` directory,
    registers a custom ``run_rag_action`` that delegates to the RAG pipeline,
    and processes queries through the configured guardrail flows.

    Args:
        model_name: LLM model used by NeMo for generation.
        config: Guardrails settings; uses environment-derived defaults if None.

    Attributes:
        last_context: The retrieved context from the most recent RAG action.
    """

    def __init__(
        self,
        model_name: str,
        config: GuardrailsConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.config = config or GuardrailsConfig.from_env()
        self.last_context: list[str] = []
        self._tmp_dir: str | None = None

        self.app = self._initialize_rails()
        weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        """Remove the temporary config directory when the instance is finalized."""
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def _initialize_rails(self) -> LLMRails:
        """Build the NeMo app from a temp config dir with env vars substituted inline.

        Copies the ``nemo_config`` files into a temporary directory, substituting
        the base URL, model name, and embedding model placeholders directly into
        the file contents (avoiding global ``os.environ`` mutation), then loads
        the rails config and registers the ``run_rag_action`` custom action.

        Returns:
            The configured NeMo :class:`LLMRails` application instance.

        Raises:
            FileNotFoundError: If the guardrails config directory does not exist.
        """
        print(f"[GUARDRAILS] Initializing NeMo Guardrails for {self.model_name!r}...")

        if not self.config.config_dir.exists():
            raise FileNotFoundError(f"Guardrails config directory not found: {self.config.config_dir!r}")

        # Copy config dir to a temp dir with env vars substituted inline,
        # avoiding global os.environ mutation.
        config_dir = self.config.config_dir
        tmp_dir = tempfile.mkdtemp(prefix="nemo_config_")
        self._tmp_dir = tmp_dir

        for fname in os.listdir(str(config_dir)):
            src = config_dir / fname
            if src.is_file():
                content = src.read_text(encoding="utf-8")
                content = content.replace("$LLM_BASE_URL", self.config.base_url)
                content = content.replace("$CURRENT_NEMO_MODEL", self.model_name)
                content = content.replace("$EMBEDDING_MODEL", self.config.embed_model)
                dst = Path(tmp_dir) / fname
                dst.write_text(content, encoding="utf-8")

        rails_config = RailsConfig.from_path(tmp_dir)
        app = LLMRails(rails_config)

        async def run_rag_action(query: str) -> str:
            # STAGE 3c: RAG EXECUTION — the Colang flow calls this custom action.
            # Delegates to rag.ask_rag (query rewrite → retriever → LLM) and
            # captures the retrieved chunks so evaluate can score faithfulness.
            result = rag.ask_rag(query, model_name=self.model_name)
            self.last_context = result.get("context", [])
            return result.get("answer", "")

        app.register_action(run_rag_action, name="run_rag_action")
        return app

    def clear_context(self) -> None:
        """Clear the stored context from previous queries."""
        self.last_context = []

    async def ask(self, user_query: str) -> GuardrailsResult:
        """Process a query through guardrails and return the answer alongside caught context.

        Resets the context state, generates a response through NeMo, and
        normalizes the response object into a plain string regardless of the
        shape NeMo returns.

        Args:
            user_query: The user message to process.

        Returns:
            A :class:`GuardrailsResult` with the bot reply and the retrieved
            context captured by the RAG action.
        """
        self.clear_context()  # Reset context state for each new query

        # STAGE 3a: INTENT MATCHING — NeMo compares the user message against
        # the Colang flows using embeddings (embeddings_only). If it matches a
        # refusal flow (politics/illegal/toxicity/classified/jailbreak), the
        # bot responds directly and the RAG action never runs.
        print(f"\n[USER]: {user_query}")
        response = await self.app.generate_async(messages=[{"role": "user", "content": user_query}])

        # STAGE 3b: RESPONSE — either the guardrail refusal text, or the answer
        # produced by run_rag_action (which delegates to rag.ask_rag, which in
        # turn calls retriever.retrieve_context). last_context holds whatever
        # chunks the RAG action retrieved.
        bot_message: str
        match response:
            case {"content": str() as msg}:
                bot_message = msg
            case [{"content": str() as msg}, *_]:
                bot_message = msg
            case _:
                bot_message = str(response)

        print(f"[BOT]: {bot_message}")
        return {"answer": bot_message, "context": self.last_context}


async def run_tests() -> None:
    """Run a manual guardrails test against a valid and an invalid query.

    Loads the embedder and test model, initializes the pipeline, and processes
    one politics-triggering query and one EMAR question to exercise both the
    refusal flows and the RAG action.
    """
    cfg = GuardrailsConfig.from_env()
    test_query = "According to EMAR 145, What is a CRS?"
    bad_query = "Who should I vote for in the next election?"

    # Clear VRAM and prepare the environment
    LMStudioModel.unload_all()
    LMStudioModelEmbedder(cfg.embed_model).load()
    LMStudioModel(cfg.test_model).load()

    # Init Guardrails Pipeline
    pipeline = GuardrailsPipeline(cfg.test_model, config=cfg)

    # Execute queries
    await pipeline.ask(bad_query)
    await pipeline.ask(test_query)


def run_tests_sync() -> None:
    """Synchronous wrapper around :func:`run_tests` for CLI execution."""
    asyncio.run(run_tests())


if __name__ == "__main__":
    run_tests_sync()
