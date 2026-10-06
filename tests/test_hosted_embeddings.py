import sys
sys.path.insert(0, "src")

import numpy as np

from dense_retrieval.embeddings.base import HostedEmbeddingAdapter


class FakeProvider:
    provider_id = "fake"

    def __init__(self):
        self.inputs = []

    def embed(self, model_id, texts):
        self.inputs.append((model_id, list(texts)))
        return np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (len(texts), 1))


def make_adapter():
    adapter = HostedEmbeddingAdapter.__new__(HostedEmbeddingAdapter)
    adapter.model_id = "Qwen/Qwen3-Embedding-0.6B"
    adapter.dimension = 2
    adapter.adapter = "qwen3"
    adapter.query_instruction = "Retrieve relevant passages"
    adapter.provider = FakeProvider()
    return adapter


def test_qwen_query_and_document_are_asymmetric():
    adapter = make_adapter()
    q = adapter.encode_queries(["capital of China?"])
    adapter.encode_documents(["Beijing is the capital of China."])
    assert adapter.provider.inputs[0][1] == [
        "Instruct: Retrieve relevant passages\nQuery:capital of China?"
    ]
    assert adapter.provider.inputs[1][1] == ["Beijing is the capital of China."]
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
