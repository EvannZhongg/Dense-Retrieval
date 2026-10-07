from collections import OrderedDict
import weakref

import numpy as np

_TORCH = None
_CUDA_AVAILABLE = None


def _cuda_available():
    """Load torch and query CUDA availability once, on first GPU retrieval."""
    global _TORCH, _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            import torch as torch_module
        except ImportError:  # pragma: no cover - torch is optional for base retrieval
            _CUDA_AVAILABLE = False
        else:
            _TORCH = torch_module
            _CUDA_AVAILABLE = bool(torch_module.cuda.is_available())
    return _CUDA_AVAILABLE



class _CudaDocumentCache:
    """Bounded LRU cache for CUDA document matrices.

    Entries retain a weak reference to the source ndarray and verify object
    identity on lookup.  This prevents a recycled ``id(ndarray)`` from
    returning a tensor for a different corpus.  The byte budget bounds device
    memory even when callers keep all source arrays alive.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self.max_bytes = int(max_bytes)
        self._entries = OrderedDict()
        self._bytes = 0

    def get(self, source, values):
        key = id(source)
        entry = self._entries.get(key)
        if entry is not None and entry["source"]() is source:
            if entry["shape"] == values.shape and entry["dtype"] == values.dtype:
                self._entries.move_to_end(key)
                return entry["tensor"]
            self._remove(key, entry)
        elif entry is not None:
            self._remove(key, entry)

        upload_values = values
        if not values.flags.c_contiguous or not values.flags.writeable:
            upload_values = np.ascontiguousarray(values).copy()
        tensor = _TORCH.from_numpy(upload_values).cuda(non_blocking=True)
        try:
            source_ref = weakref.ref(source, lambda ref, key=key: self._remove_ref(key, ref))
        except TypeError:
            return tensor
        entry = {
            "source": source_ref,
            "tensor": tensor,
            "shape": values.shape,
            "dtype": values.dtype,
            "bytes": int(tensor.numel() * tensor.element_size()),
        }
        self._entries[key] = entry
        self._bytes += entry["bytes"]
        self._evict()
        return tensor

    def _remove_ref(self, key, source_ref):
        entry = self._entries.get(key)
        if entry is not None and entry["source"] is source_ref:
            self._remove(key, entry)

    def _remove(self, key, entry):
        self._entries.pop(key, None)
        self._bytes -= entry["bytes"]

    def _evict(self):
        evicted = False
        while self._bytes > self.max_bytes and self._entries:
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry["bytes"]
            evicted = True
        if evicted and _TORCH is not None and _CUDA_AVAILABLE:
            _TORCH.cuda.empty_cache()

    def clear(self):
        self._entries.clear()
        self._bytes = 0
        if _TORCH is not None and _CUDA_AVAILABLE:
            _TORCH.cuda.empty_cache()


_TORCH_DOCUMENT_CACHE = _CudaDocumentCache()


def clear_cuda_document_cache():
    """Release all cached CUDA document matrices."""
    _TORCH_DOCUMENT_CACHE.clear()


def exact_search(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    k: int = 100,
):
    q = np.asarray(query_embeddings, dtype=np.float32)
    d = np.asarray(document_embeddings, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]:
        raise ValueError(f"embedding shape mismatch: {q.shape}, {d.shape}")
    k = min(k, len(d))
    if _cuda_available():
        # Keep the frozen document matrix resident across the many batched
        # evaluation calls made by lambda selection.
        source = document_embeddings if isinstance(document_embeddings, np.ndarray) else d
        cached = _TORCH_DOCUMENT_CACHE.get(source, d)
        upload_queries = q
        if not q.flags.c_contiguous or not q.flags.writeable:
            upload_queries = np.ascontiguousarray(q).copy()
        query = _TORCH.from_numpy(upload_queries).cuda(non_blocking=True)
        with _TORCH.inference_mode():
            scores, indices = _TORCH.topk(query @ cached.T, k=k, dim=1)
        return indices.cpu().numpy(), scores.cpu().numpy()
    scores = q @ d.T
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(q))[:, None]
    order = np.argsort(-scores[row, idx], axis=1)
    idx = idx[row, order]
    return idx, scores[row, idx]
