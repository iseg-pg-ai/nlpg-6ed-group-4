"""LM Studio API client utilities.

Manages model loading, unloading, text generation, and embeddings through the
LM Studio OpenAI-compatible REST API. A module-level :class:`requests.Session`
is reused for connection pooling and keep-alive across all HTTP calls.
"""

import os
from dataclasses import dataclass
from typing import Any, override

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)  # Load environment variables


@dataclass(frozen=True, slots=True, kw_only=True)
class LMStudioConfig:
    """Immutable, slot-optimized configuration for the LM Studio API client.

    Attributes:
        base_url: Base URL of the LM Studio OpenAI-compatible endpoint.
        api_key: API key expected by the OpenAI client.
    """

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.getenv("LLM_API_KEY", "local-dummy-key")

    @property
    def api_base(self) -> str:
        """Strip the trailing ``/v1`` suffix to get the REST root URL.

        Returns:
            The REST root URL used for model management endpoints.
        """
        return self.base_url.removesuffix("/v1")

    @property
    def download_url(self) -> str:
        """Return the model download endpoint URL."""
        return f"{self.api_base}/api/v1/models/download"

    @property
    def status_url(self) -> str:
        """Return the model download status endpoint URL."""
        return f"{self.api_base}/api/v1/models/download/status"

    @property
    def load_model_url(self) -> str:
        """Return the model load endpoint URL."""
        return f"{self.api_base}/api/v1/models/load"

    @property
    def unload_model_url(self) -> str:
        """Return the model unload endpoint URL."""
        return f"{self.api_base}/api/v1/models/unload"

    @property
    def get_all_models_url(self) -> str:
        """Return the endpoint that lists all loaded models."""
        return f"{self.api_base}/api/v1/models"


_session = requests.Session()


def json_post(url: str, payload: dict[str, Any], timeout: float = 10.0) -> requests.Response:
    """Issue a POST request with a JSON payload and a default timeout safeguard.

    Args:
        url: The endpoint to POST to.
        payload: The JSON payload to send.
        timeout: Request timeout in seconds.

    Returns:
        The :class:`requests.Response` from the server.
    """
    return _session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)


class LMStudioModel:
    """Base class for LM Studio models handling loading, unloading, and text generation.

    Wraps the OpenAI-compatible client pointed at the LM Studio server and
    exposes convenience methods for VRAM management and chat completion.

    Args:
        model_name: The model identifier as known to LM Studio.
        config: Optional client configuration; defaults to environment values.
    """

    def __init__(self, model_name: str, config: LMStudioConfig | None = None) -> None:
        self.model_name = model_name
        self.config = config or LMStudioConfig()

        self.client = OpenAI(  # Initialize the OpenAI-compatible client
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def load(self, context_length: int = 2048) -> None:
        """Load this specific model into LM Studio memory with context limits.

        Prints a confirmation or an error message; a 409 response is treated as
        "already loaded".

        Args:
            context_length: Maximum context length to allocate for the model.
        """
        print(f"[SYSTEM] Requesting LM Studio API to load {self.model_name!r}...")
        try:
            payload = {
                "model": self.model_name,
                "context_length": context_length,
            }

            response = json_post(self.config.load_model_url, payload)

            match response.status_code:
                case 200:
                    print(f"[SYSTEM] Successfully loaded {self.model_name!r}.")
                case 409:
                    print(f"[SYSTEM] {self.model_name!r} is already loaded.")
                case _:
                    print(f"[ERROR] Failed to load model. Status: {response.status_code}, Details: {response.text}")

        except requests.exceptions.RequestException as err:
            print(f"[ERROR] Could not connect to LM Studio: {err}")

    def unload(self) -> None:
        """Unload this specific model to free VRAM."""
        print(f"[SYSTEM] Requesting LM Studio API to unload {self.model_name!r}...")
        try:
            response = json_post(
                self.config.unload_model_url,
                {"instance_id": self.model_name},
            )
            if response.ok:
                print(f"[SYSTEM] Unloaded {self.model_name!r} to free up VRAM.")
            else:
                print(f"[ERROR] Failed to unload {self.model_name!r}: {response.text}")
        except requests.exceptions.RequestException as err:
            print(f"[ERROR] Failed to unload model: {err}")

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 500,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """Generate a chat completion, optionally with structured JSON mode.

        Args:
            messages: The chat messages, e.g. system/user roles.
            temperature: Sampling temperature; lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            response_format: Optional OpenAI JSON schema to constrain output.

        Returns:
            The chat completion response object.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def unload_all(config: LMStudioConfig | None = None) -> None:
        """Find and unload all currently loaded models to clear VRAM.

        Args:
            config: Optional client configuration; defaults to environment values.
        """
        cfg = config or LMStudioConfig()
        print("[SYSTEM] Checking for currently loaded models...")

        try:
            response = _session.get(cfg.get_all_models_url, timeout=10)
        except requests.exceptions.ConnectionError:
            print("[ERROR] Could not connect to LM Studio. Is the server running?")
            return

        if not response.ok:
            print(f"[ERROR] Failed to fetch models. Status code: {response.status_code}")
            return

        models_data = response.json().get("models", [])

        loaded_instance_ids = [
            instance["id"]
            for model in models_data
            for instance in model.get("loaded_instances", [])
            if "id" in instance
        ]

        if not loaded_instance_ids:
            print("[SYSTEM] No models are currently loaded in VRAM. Clean slate!")
            return

        for instance_id in loaded_instance_ids:
            print(f"[SYSTEM] Unloading {instance_id!r}...")
            json_post(cfg.unload_model_url, {"instance_id": instance_id})

        print("[SYSTEM] All loaded models successfully unloaded. VRAM is cleared.")


class LMStudioModelEmbedder(LMStudioModel):
    """LM Studio model specialized for generating text embeddings.

    Inherits loading/unloading from :class:`LMStudioModel` and adds single-text
    and batch embedding methods.
    """

    @override
    def load(self, context_length: int = 2048) -> None:
        """Load the embedding model into memory.

        Args:
            context_length: Maximum context length to allocate for the model.
        """
        super().load(context_length=context_length)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts in a single API call.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]
