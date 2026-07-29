import os
import time
from dataclasses import dataclass
from typing import Any, override

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass(frozen=True, slots=True, kw_only=True)
class LMStudioConfig:
    """Immutable, slot-optimized configuration for the LM Studio API client."""

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.getenv("LLM_API_KEY", "local-dummy-key")

    @property
    def api_base(self) -> str:
        """Strip trailing /v1 suffix to get the REST root URL."""
        return self.base_url.removesuffix("/v1")

    @property
    def download_url(self) -> str:
        return f"{self.api_base}/api/v1/models/download"

    @property
    def status_url(self) -> str:
        return f"{self.api_base}/api/v1/models/download/status"

    @property
    def load_model_url(self) -> str:
        return f"{self.api_base}/api/v1/models/load"

    @property
    def unload_model_url(self) -> str:
        return f"{self.api_base}/api/v1/models/unload"

    @property
    def get_all_models_url(self) -> str:
        return f"{self.api_base}/api/v1/models"


_session = requests.Session()


def json_post(url: str, payload: dict[str, Any], timeout: float = 10.0) -> requests.Response:
    """Helper to issue POST requests with JSON payload and default timeout safeguard."""
    return _session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)


class LMStudioModel:
    """Base class for LM Studio models handling loading, unloading, and text generation."""

    def __init__(self, model_name: str, config: LMStudioConfig | None = None) -> None:
        self.model_name = model_name
        self.config = config or LMStudioConfig()

        self.client = OpenAI(  # Initialize the OpenAI-compatible client
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def load(self, context_length: int = 2048) -> None:
        """Load this specific model into LM Studio memory with context limits."""
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
        """Generate a chat completion. Supports structured JSON mode via response_format."""
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
        """Find and unload all currently loaded models."""
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
    """Extended class specifically for models that generate embeddings."""

    @override
    def load(self, context_length: int = 2048) -> None:
        """Load the embedding model into memory."""
        super().load(context_length=context_length)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts in a single API call."""
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]


def ensure_models_downloaded(config: LMStudioConfig | None = None) -> None:
    """Ensure all required models are downloaded on disk via LM Studio REST API."""
    cfg = config or LMStudioConfig()
    models = (
        os.getenv("MODEL_A", "ministral-3-3b-instruct-2512"),
        os.getenv("MODEL_B", "qwen3.5-2b"),
        os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5"),
        os.getenv("JUDGE_MODEL", "llama-3.2-1b-instruct@q6_k"),
    )

    print("\n[SYSTEM] Verifying required models are downloaded via LM Studio API...")

    try:
        for model in models:
            print(f"[SYSTEM] Checking {model!r}...")
            response = json_post(cfg.download_url, {"model": model})

            if not response.ok:
                print(f"[ERROR] API rejected download request for {model!r}: {response.text}")
                continue

            match response.json():
                case {"status": "already_downloaded"}:
                    print("Already downloaded.")

                case {"status": "downloading", "job_id": job_id}:
                    print(f"Downloading (Job ID: {job_id}). This may take a few minutes...")

                    max_polls = 60
                    poll_interval = 3
                    for attempt in range(max_polls):
                        status_resp = _session.get(f"{cfg.status_url}/{job_id}", timeout=10)
                        if not status_resp.ok:
                            print(f"[ERROR] Polling failed for job {job_id!r}. Status code: {status_resp.status_code}")
                            break

                        match status_resp.json():
                            case {"status": "completed"}:
                                print("Download complete!")
                                break
                            case {"status": "failed" | "paused" as state}:
                                print(f"Download {state}. Please check the LM Studio UI.")
                                break
                            case status_data:
                                status_val = status_data.get("status", "unknown")
                                if attempt == 0:
                                    print(f"[DEBUG] Polling job {job_id!r}... Status: {status_val}")
                                if attempt == max_polls - 1:
                                    print(f"[ERROR] Download job {job_id!r} did not complete after {max_polls * poll_interval}s.")
                                    break

                        time.sleep(poll_interval)

                case unknown:
                    print(f"[WARNING] Unrecognized API response for {model!r}: {unknown}")

        print("[SYSTEM] All models verified!\n")

    except requests.exceptions.RequestException as err:
        print(f"[ERROR] Could not communicate with LM Studio server: {err}")
