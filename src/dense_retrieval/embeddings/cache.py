"""On-disk embedding cache helpers shared by the analysis scripts.

A cache entry is a directory under ``<cache_root>/<dataset>/<model_dir>/`` that
holds both ``queries.npy`` and ``documents.npy``.  Query encodings that are not
part of a full experiment run (FiQA train/dev queries, for example) are
additionally checkpointed per batch through :func:`cached_encode_queries`, so an
interrupted hosted encoding resumes instead of re-spending API calls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def find_cache_dir(cache_root, dataset: str, model_dir: str) -> Path:
    """Return the single complete cache directory for one dataset/model pair."""
    root = Path(cache_root) / dataset / model_dir
    candidates = (
        sorted(
            path
            for path in root.iterdir()
            if (path / "queries.npy").exists()
            and (path / "documents.npy").exists()
        )
        if root.exists()
        else []
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one complete cache under {root}, found {len(candidates)}"
        )
    return candidates[0]


def cached_encode_queries(model, texts, path: Path):
    """Encode query texts in checkpointed batches and cache the final array."""
    if path.exists():
        array = np.load(path)
        if array.shape == (len(texts), model.dimension):
            return array.astype(np.float32)
    batch_size = int(getattr(model, "cache_batch_size", model.batch_size))
    parts_dir = path.parent / f"{path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        part_path = parts_dir / f"{start:09d}_{end:09d}.npy"
        part = np.load(part_path) if part_path.exists() else None
        if part is None or part.shape != (end - start, model.dimension):
            part = model.encode_queries(texts[start:end]).astype(np.float32)
            np.save(part_path, part)
        parts.append(part)
        print(f"train query encoding: {end}/{len(texts)}", flush=True)
    array = np.concatenate(parts)
    np.save(path, array)
    return array
