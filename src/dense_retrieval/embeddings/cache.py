"""Deterministic, resumable on-disk embedding caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def cache_fingerprint(dataset: str, split: str, model, encoding_config: dict | None = None) -> str:
    payload = {"dataset": dataset, "split": split, "model_id": model.model_id, "revision": model.revision, "dimension": int(model.dimension), "encoding_config": encoding_config if encoding_config is not None else model.encoding_config}
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _encode_parts(model, texts: Sequence[str], kind: str, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(getattr(model, "cache_batch_size", getattr(model, "batch_size", 32))))
    parts_dir = path.parent / f"{path.stem}.parts"
    parts_dir.mkdir(exist_ok=True)
    spans = [(start, min(start + batch_size, len(texts))) for start in range(0, len(texts), batch_size)]
    parts = []
    for start, end in spans:
        part_path = parts_dir / f"{start:09d}_{end:09d}.npy"
        expected = (end - start, int(model.dimension))
        try:
            part = np.load(part_path) if part_path.exists() else None
        except (OSError, ValueError):
            part = None
        if part is None or part.shape != expected:
            encoder = model.encode_queries if kind == "query" else model.encode_documents
            part = np.asarray(encoder(texts[start:end]), dtype=np.float32)
            if part.shape != expected:
                raise ValueError(f"{kind} batch shape {part.shape}; expected {expected}")
            np.save(part_path, part)
        parts.append(np.asarray(part, dtype=np.float32))
    result = np.concatenate(parts, axis=0) if parts else np.empty((0, model.dimension), dtype=np.float32)
    np.save(path, result)
    return result


def encode_with_resume(model, texts: Sequence[str], kind: str, path: str | Path) -> np.ndarray:
    """Encode in checkpointed batches, reusing complete arrays and batch parts."""
    if kind not in {"query", "document"}:
        raise ValueError("kind must be 'query' or 'document'")
    path = Path(path)
    expected = (len(texts), int(model.dimension))
    if path.exists():
        try:
            array = np.load(path)
            if array.shape == expected:
                return np.asarray(array, dtype=np.float32)
        except (OSError, ValueError):
            pass
    return _encode_parts(model, list(texts), kind, path)


def save_cache(cache_dir: str | Path, queries: np.ndarray, documents: np.ndarray, manifest: dict) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "queries.npy", np.asarray(queries, dtype=np.float32))
    np.save(cache_dir / "documents.npy", np.asarray(documents, dtype=np.float32))
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return cache_dir


def load_cache(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    queries = np.load(cache_dir / "queries.npy")
    documents = np.load(cache_dir / "documents.npy")
    if len(queries) != len(manifest.get("query_ids", [])) or len(documents) != len(manifest.get("document_ids", [])):
        raise ValueError("cache arrays and manifest ids have inconsistent lengths")
    if documents.ndim != 2 or queries.ndim != 2:
        raise ValueError("cache arrays must be rank-2")
    if len(documents) == 0 and manifest.get("document_ids"):
        raise ValueError("non-empty document_ids require document embeddings")
    return queries, documents, manifest
