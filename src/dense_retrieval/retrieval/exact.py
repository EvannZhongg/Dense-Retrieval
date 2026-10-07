import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the base retriever
    torch = None


_TORCH_DOCUMENT_CACHE = {}

def exact_search(query_embeddings: np.ndarray, document_embeddings: np.ndarray, k: int = 100):
    q = np.asarray(query_embeddings, dtype=np.float32)
    d = np.asarray(document_embeddings, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]: raise ValueError(f"embedding shape mismatch: {q.shape}, {d.shape}")
    k = min(k, len(d))
    if torch is not None and torch.cuda.is_available():
        # Keep the frozen document matrix resident across the many batched
        # evaluation calls made by lambda selection.
        key = (id(document_embeddings), d.shape, str(d.dtype))
        cached = _TORCH_DOCUMENT_CACHE.get(key)
        if cached is None or cached.device.type != "cuda":
            cached = torch.from_numpy(d).cuda(non_blocking=True)
            _TORCH_DOCUMENT_CACHE[key] = cached
        query = torch.from_numpy(q).cuda(non_blocking=True)
        with torch.inference_mode():
            scores, indices = torch.topk(query @ cached.T, k=k, dim=1)
        return indices.cpu().numpy(), scores.cpu().numpy()
    scores = q @ d.T
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(q))[:, None]
    order = np.argsort(-scores[row, idx], axis=1)
    idx = idx[row, order]
    return idx, scores[row, idx]
