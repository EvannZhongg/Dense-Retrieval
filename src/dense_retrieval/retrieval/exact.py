import numpy as np

def exact_search(query_embeddings: np.ndarray, document_embeddings: np.ndarray, k: int = 100):
    q = np.asarray(query_embeddings, dtype=np.float32)
    d = np.asarray(document_embeddings, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]: raise ValueError(f"embedding shape mismatch: {q.shape}, {d.shape}")
    k = min(k, len(d))
    scores = q @ d.T
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(q))[:, None]
    order = np.argsort(-scores[row, idx], axis=1)
    idx = idx[row, order]
    return idx, scores[row, idx]

