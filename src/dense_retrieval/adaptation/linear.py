from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class LinearQueryCalibrator:
    """A small query-only linear map with a frozen document space.

    ``weights`` is stored as ``[query_dimension, document_dimension]`` so the
    online operation is a single matrix multiplication followed by L2
    normalization. The class has no document-side state by design.
    """

    def __init__(self, weights: np.ndarray, normalize: bool = True, method: str = "ridge"):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim != 2:
            raise ValueError("weights must be a rank-2 matrix")
        self.weights = weights
        self.normalize = bool(normalize)
        self.method = str(method)

    @property
    def query_dimension(self) -> int:
        return int(self.weights.shape[0])

    @property
    def document_dimension(self) -> int:
        return int(self.weights.shape[1])

    def transform(self, query_embeddings: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.query_dimension:
            raise ValueError(
                f"query shape {queries.shape}; expected (?, {self.query_dimension})"
            )
        transformed = queries @ self.weights
        return _normalize(transformed) if self.normalize else transformed.astype(np.float32)

    def save(self, path: str | Path, metadata: Mapping | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query_dimension": self.query_dimension,
            "document_dimension": self.document_dimension,
            "normalize": self.normalize,
            "method": self.method,
            **dict(metadata or {}),
        }
        np.savez(path, weights=self.weights)
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LinearQueryCalibrator":
        path = Path(path)
        with np.load(path) as payload:
            weights = np.asarray(payload["weights"], dtype=np.float32)
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return cls(weights, normalize=metadata.get("normalize", True), method=metadata.get("method", "ridge"))


def _ridge_map(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or len(source) != len(target):
        raise ValueError("source and target must be aligned rank-2 arrays")
    if len(source) == 0:
        raise ValueError("at least one training example is required")
    if float(alpha) < 0:
        raise ValueError("alpha must be non-negative")
    gram = source.T @ source
    regularizer = np.eye(source.shape[1], dtype=np.float64) * float(alpha)
    # Solve in the query space, which is substantially smaller than the
    # document collection and remains stable for underdetermined data.
    weights = np.linalg.solve(gram + regularizer, source.T @ target)
    return np.asarray(weights, dtype=np.float32)


def fit_corpus_alignment(
    query_side_document_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    alpha: float = 1e-2,
    normalize: bool = True,
) -> LinearQueryCalibrator:
    """Fit from query-protocol encodings of corpus text to frozen doc vectors."""
    source = np.asarray(query_side_document_embeddings, dtype=np.float32)
    target = np.asarray(document_embeddings, dtype=np.float32)
    if source.shape[0] != target.shape[0]:
        raise ValueError("corpus source and document arrays must have equal length")
    weights = _ridge_map(source, target, alpha)
    return LinearQueryCalibrator(weights, normalize=normalize, method="corpus_alignment_ridge")


def fit_gold_supervised(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    document_ids: Sequence[str],
    train_query_ids: Iterable[str] | None = None,
    alpha: float = 1e-2,
    normalize: bool = True,
) -> LinearQueryCalibrator:
    """Fit a query-only map to relevance-weighted positive document centroids.

    This is a diagnostic reference: qrels are used only to create training
    targets and must be restricted to a declared training query split by the
    caller. Queries without a positive document in the frozen cache are
    skipped rather than silently using a different document representation.
    """
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    document_embeddings = np.asarray(document_embeddings, dtype=np.float32)
    if query_embeddings.ndim != 2 or len(query_embeddings) != len(query_ids):
        raise ValueError("query embeddings and query_ids are not aligned")
    if document_embeddings.ndim != 2 or len(document_embeddings) != len(document_ids):
        raise ValueError("document embeddings and document_ids are not aligned")
    document_index = {str(doc_id): i for i, doc_id in enumerate(document_ids)}
    allowed = set(str(qid) for qid in train_query_ids) if train_query_ids is not None else set(map(str, query_ids))
    sources, targets = [], []
    for row, query_id in zip(query_embeddings, query_ids):
        query_id = str(query_id)
        if query_id not in allowed:
            continue
        positives = [
            (document_index[str(doc_id)], max(0, int(score)))
            for doc_id, score in qrels.get(query_id, {}).items()
            if int(score) > 0 and str(doc_id) in document_index
        ]
        if not positives:
            continue
        indices = np.asarray([idx for idx, _ in positives], dtype=np.int64)
        weights = np.asarray([score for _, score in positives], dtype=np.float32)
        target = (document_embeddings[indices] * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
        sources.append(row)
        targets.append(target)
    if not sources:
        raise ValueError("no usable positive training queries were found")
    calibrator = LinearQueryCalibrator(
        _ridge_map(np.asarray(sources), np.asarray(targets), alpha),
        normalize=normalize,
        method="gold_centroid_ridge",
    )
    return calibrator
