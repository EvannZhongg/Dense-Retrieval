from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import requests


class EmbeddingProvider(ABC):
    """Transport-only interface for hosted embedding APIs."""

    provider_id: str

    @abstractmethod
    def embed(self, model_id: str, texts: Sequence[str]) -> np.ndarray:
        """Return raw embeddings in the same order as texts."""


class OpenAICompatibleProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 120,
        max_retries: int = 5,
        encoding_format: str = "float",
        provider_id: str = "openai-compatible",
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required environment variable: {api_key_env}")
        self.provider_id = provider_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.encoding_format = encoding_format
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def embed(self, model_id: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        payload = {
            "model": model_id,
            "input": list(texts),
            "encoding_format": self.encoding_format,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if not response.ok:
                    detail = response.text[:1000]
                    raise RuntimeError(
                        f"Embedding API returned HTTP {response.status_code}: {detail}"
                    )
                body = response.json()
                data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
                if len(data) != len(texts):
                    raise RuntimeError(
                        f"Embedding API returned {len(data)} vectors for {len(texts)} inputs"
                    )
                return np.asarray([item["embedding"] for item in data], dtype=np.float32)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Embedding API failed after {self.max_retries + 1} attempts"
                    ) from exc
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


class SiliconFlowProvider(OpenAICompatibleProvider):
    def __init__(self, **kwargs):
        super().__init__(
            base_url=kwargs.pop("base_url", "https://api.siliconflow.cn/v1"),
            api_key_env=kwargs.pop("api_key_env", "SILICONFLOW_API_KEY"),
            provider_id="siliconflow",
            **kwargs,
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, **kwargs):
        super().__init__(
            base_url=kwargs.pop("base_url", "https://openrouter.ai/api/v1"),
            api_key_env=kwargs.pop("api_key_env", "OPENROUTER_API_KEY"),
            provider_id="openrouter",
            **kwargs,
        )


class AibermProvider(EmbeddingProvider):
    """Aiberm's OpenAI SDK-compatible embeddings endpoint."""

    def __init__(
        self,
        base_url: str = "https://aiberm.com/v1",
        api_key_env: str = "AIBERM_API_KEY",
        timeout_seconds: float = 120,
        max_retries: int = 5,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required environment variable: {api_key_env}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The aiberm provider requires the 'openai' package"
            ) from exc
        self.provider_id = "aiberm"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def embed(self, model_id: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=model_id,
                    input=list(texts),
                )
                data = sorted(
                    response.data,
                    key=lambda item: getattr(item, "index", 0),
                )
                if len(data) != len(texts):
                    raise RuntimeError(
                        f"Embedding API returned {len(data)} vectors for {len(texts)} inputs"
                    )
                return np.asarray(
                    [item.embedding for item in data], dtype=np.float32
                )
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and status_code not in {408, 409, 429} and status_code < 500:
                    reason = (
                        "check AIBERM_API_KEY permissions and model access"
                        if status_code in {401, 403}
                        else "check the model name and request input"
                    )
                    raise RuntimeError(
                        f"Aiberm embedding API returned HTTP {status_code}; {reason}"
                    ) from exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Aiberm embedding API failed after {self.max_retries + 1} attempts"
                    ) from exc
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


def create_provider(config: dict) -> EmbeddingProvider:
    provider_type = config.get("type", "siliconflow").lower()
    common = {
        key: config[key]
        for key in (
            "base_url",
            "api_key_env",
            "timeout_seconds",
            "max_retries",
            "encoding_format",
        )
        if key in config
    }
    if provider_type == "siliconflow":
        return SiliconFlowProvider(**common)
    if provider_type == "openrouter":
        return OpenRouterProvider(**common)
    if provider_type == "aiberm":
        return AibermProvider(
            base_url=config.get("base_url", "https://aiberm.com/v1"),
            api_key_env=config.get("api_key_env", "AIBERM_API_KEY"),
            timeout_seconds=config.get("timeout_seconds", 120),
            max_retries=config.get("max_retries", 5),
        )
    if provider_type == "openai-compatible":
        if "base_url" not in common or "api_key_env" not in common:
            raise ValueError("openai-compatible provider requires base_url and api_key_env")
        return OpenAICompatibleProvider(**common)
    raise ValueError(f"Unsupported embedding provider: {provider_type}")
