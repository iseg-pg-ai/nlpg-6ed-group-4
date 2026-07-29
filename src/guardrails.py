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

load_dotenv(override=True)


class GuardrailsResult(TypedDict):
    """Structured response type for guardrails execution."""

    answer: str
    context: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardrailsConfig:
    """Immutable, slot-optimized configuration for NeMo Guardrails."""

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.getenv("LLM_API_KEY", "local-dummy-key")
    test_model: str = os.getenv("MODEL_A", "ministral-3-3b-instruct-2512")
    embed_model: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
    config_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "nemo_config")

    @classmethod
    def from_env(cls) -> Self:
        """Instantiate configuration directly from environment variables."""
        return cls()

    @property
    def api_base(self) -> str:
        """Strip trailing /v1 suffix to get the REST root URL."""
        return self.base_url.removesuffix("/v1")


class GuardrailsPipeline:
    """Encapsulates NeMo Guardrails initialization, custom actions, and query state."""

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
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def _initialize_rails(self) -> LLMRails:
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
                dst = Path(tmp_dir) / fname
                dst.write_text(content, encoding="utf-8")

        rails_config = RailsConfig.from_path(tmp_dir)
        app = LLMRails(rails_config)

        async def run_rag_action(query: str) -> str:
            result = rag.ask_rag(query, model_name=self.model_name)
            self.last_context = result.get("context", [])
            return result.get("answer", "")

        app.register_action(run_rag_action, name="run_rag_action")
        return app

    def clear_context(self) -> None:
        """Manually clear the stored context."""
        self.last_context = []

    async def ask(self, user_query: str) -> GuardrailsResult:
        """Process a query through guardrails and return the answer alongside caught context."""
        self.clear_context()  # Reset context state for each new query

        print(f"\n[USER]: {user_query}")
        response = await self.app.generate_async(messages=[{"role": "user", "content": user_query}])

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
    cfg = GuardrailsConfig.from_env()
    test_query = "What is the main topic of the text?"
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
    asyncio.run(run_tests())


if __name__ == "__main__":
    run_tests_sync()
