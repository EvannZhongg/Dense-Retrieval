from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .providers import create_provider

class EmbeddingModel(ABC):
    model_id: str
    revision: str
    dimension: int
    encoding_config: dict

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

class HashEmbeddingModel(EmbeddingModel):
    """Small deterministic adapter for tests and pipeline smoke runs."""
    def __init__(self, dimension: int = 64, seed: int = 0):
        self.model_id, self.revision, self.dimension = "hash", str(seed), int(dimension)
        self.encoding_config = {"type": "sha256", "seed": seed}
    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        out = np.empty((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = hashlib.sha256(f"{self.revision}:{prefix}:{text}".encode()).digest()
            raw = (seed * ((self.dimension * 4 // len(seed)) + 1))[: self.dimension * 4]
            out[i] = np.frombuffer(raw, dtype=np.uint32).astype(np.float32) / 2**32 - 0.5
        return self.normalize(out)
    def encode_queries(self, texts): return self._encode(texts, "query")
    def encode_documents(self, texts): return self._encode(texts, "document")

class SentenceTransformerAdapter(EmbeddingModel):
    def __init__(self, model_id: str, revision: str | None = None, device: str | None = None, batch_size: int = 32, max_seq_length: int | None = None, query_instruction: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.revision = revision or "main"
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.encoding_config = {"adapter": "sentence-transformers", "batch_size": batch_size, "max_seq_length": max_seq_length, "query_instruction": query_instruction}
        self.model = SentenceTransformer(model_id, revision=revision, device=device) if revision else SentenceTransformer(model_id, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.dimension = int(self.model.get_sentence_embedding_dimension())
    def _encode(self, texts, prompt_name=None):
        kwargs = {"batch_size": self.batch_size, "convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": True}
        if prompt_name and hasattr(self.model, "encode_query"):
            return self.normalize(self.model.encode_query(list(texts), **kwargs))
        return self.normalize(self.model.encode(list(texts), **kwargs))
    def encode_queries(self, texts):
        texts = [f"{self.query_instruction}{t}" if self.query_instruction else t for t in texts]
        return self._encode(texts, "query")
    def encode_documents(self, texts): return self._encode(texts, "document")

class E5Adapter(SentenceTransformerAdapter):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def encode_queries(self, texts): return super().encode_queries([f"query: {t}" for t in texts])
    def encode_documents(self, texts): return super().encode_documents([f"passage: {t}" for t in texts])

class Qwen3Adapter(SentenceTransformerAdapter):
    def __init__(self, query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query", **kwargs): super().__init__(query_instruction=query_instruction, **kwargs)
    def encode_queries(self, texts):
        formatted = [f"Instruct: {self.query_instruction}\nQuery:{text}" for text in texts]
        return self._encode(formatted, "query")

class GTEAdapter(SentenceTransformerAdapter):
    pass


class HostedEmbeddingAdapter(EmbeddingModel):
    """Model-aware preprocessing backed by a transport-only provider."""

    def __init__(
        self,
        model_id: str,
        dimension: int,
        provider: dict,
        adapter: str,
        batch_size: int = 32,
        revision: str = "hosted",
        query_instruction: str | None = None,
        empty_text_placeholder: str = "[EMPTY]",
    ):
        self.model_id = model_id
        self.dimension = int(dimension)
        self.revision = revision
        self.adapter = adapter
        self.batch_size = int(batch_size)
        self.cache_batch_size = self.batch_size
        self.query_instruction = query_instruction
        self.empty_text_placeholder = empty_text_placeholder
        self.provider_config = dict(provider)
        self.provider = create_provider(provider)
        self.encoding_config = {
            "adapter": adapter,
            "provider": {
                k: v for k, v in provider.items() if k not in {"api_key", "token"}
            },
            "batch_size": self.batch_size,
            "query_instruction": query_instruction,
            "empty_text_placeholder": empty_text_placeholder,
            "normalization": "l2",
        }

    def _query_texts(self, texts: Sequence[str]) -> list[str]:
        if self.adapter == "e5":
            return [f"query: {text}" for text in texts]
        if self.adapter == "qwen3":
            instruction = self.query_instruction or (
                "Given a web search query, retrieve relevant passages that answer the query"
            )
            return [f"Instruct: {instruction}\nQuery:{text}" for text in texts]
        return list(texts)

    def _document_texts(self, texts: Sequence[str]) -> list[str]:
        if self.adapter == "e5":
            return [f"passage: {text}" for text in texts]
        return list(texts)

    def _sanitize_texts(self, texts: Sequence[str]) -> list[str]:
        return [
            text if text and text.strip() else self.empty_text_placeholder
            for text in texts
        ]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.provider.embed(self.model_id, texts)
        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Provider returned shape {vectors.shape}; expected {(len(texts), self.dimension)}"
            )
        return self.normalize(vectors)

    def encode_queries(self, texts):
        return self._encode(self._query_texts(self._sanitize_texts(texts)))

    def encode_documents(self, texts):
        return self._encode(self._document_texts(self._sanitize_texts(texts)))

def create_embedding_model(config: dict) -> EmbeddingModel:
    adapter = config.get("adapter", "auto")
    model_id = config.get("model_id", "hash")
    if adapter == "hash" or model_id == "hash": return HashEmbeddingModel(config.get("dimension", 64), config.get("seed", 0))
    common = {k: config[k] for k in ("model_id", "revision", "device", "batch_size", "max_seq_length") if k in config}
    if adapter == "auto":
        low = model_id.lower()
        adapter = "e5" if "e5" in low else "qwen3" if "qwen3" in low else "gte" if "gte" in low else "sentence-transformers"
    if config.get("provider"):
        if "dimension" not in config:
            raise ValueError("Hosted embedding models require an explicit dimension")
        return HostedEmbeddingAdapter(
            model_id=model_id,
            dimension=config["dimension"],
            provider=config["provider"],
            adapter=adapter,
            batch_size=config.get("batch_size", 32),
            revision=config.get("revision", "hosted"),
            query_instruction=config.get("query_instruction"),
            empty_text_placeholder=config.get("empty_text_placeholder", "[EMPTY]"),
        )
    cls = {"e5": E5Adapter, "qwen3": Qwen3Adapter, "gte": GTEAdapter, "sentence-transformers": SentenceTransformerAdapter}[adapter]
    if adapter == "qwen3":
        return cls(**common, **({"query_instruction": config["query_instruction"]} if config.get("query_instruction") is not None else {}))
    if adapter == "sentence-transformers" and config.get("query_instruction") is not None:
        return cls(**common, query_instruction=config["query_instruction"])
    return cls(**common)
