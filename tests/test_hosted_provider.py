import numpy as np

from dense_retrieval.embeddings.base import HostedEmbeddingAdapter


class FakeProvider:
    def __init__(self):
        self.calls = []

    def embed(self, model_id, texts):
        self.calls.append((model_id, list(texts)))
        return np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (len(texts), 1))


def test_hosted_e5_preprocessing_and_normalization():
    adapter = HostedEmbeddingAdapter.__new__(HostedEmbeddingAdapter)
    adapter.model_id = "e5"
    adapter.dimension = 2
    adapter.adapter = "e5"
    adapter.provider = FakeProvider()
    adapter.empty_text_placeholder = "[EMPTY]"
    query = adapter.encode_queries([""])
    adapter.encode_documents(["text"])
    assert adapter.provider.calls[0][1] == ["query: [EMPTY]"]
    assert adapter.provider.calls[1][1] == ["passage: text"]
    assert np.allclose(np.linalg.norm(query, axis=1), 1.0)
