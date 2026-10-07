import sys
sys.path.insert(0, "src")
import os

import numpy as np

from dense_retrieval.embeddings.base import HostedEmbeddingAdapter
from dense_retrieval.embeddings.providers.base import AibermProvider, create_provider


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


def test_e5_query_and_document_use_official_prefixes():
    adapter = make_adapter()
    adapter.model_id = "intfloat/e5-base-v2"
    adapter.adapter = "e5"
    adapter.encode_queries(["business expense"])
    adapter.encode_documents(["A deductible business expense"])
    assert adapter.provider.inputs[0][1] == ["query: business expense"]
    assert adapter.provider.inputs[1][1] == ["passage: A deductible business expense"]


def test_hosted_adapter_replaces_blank_text_with_configured_placeholder():
    adapter = make_adapter()
    adapter.adapter = "text-embedding-3-small"
    adapter.empty_text_placeholder = "[EMPTY]"
    adapter.encode_documents(["", "   "])
    assert adapter.provider.inputs[0][1] == ["[EMPTY]", "[EMPTY]"]


def test_e5_applies_prefix_after_blank_text_replacement():
    adapter = make_adapter()
    adapter.adapter = "e5"
    adapter.empty_text_placeholder = "[EMPTY]"
    adapter.encode_queries([""])
    adapter.encode_documents(["   "])
    assert adapter.provider.inputs[0][1] == ["query: [EMPTY]"]
    assert adapter.provider.inputs[1][1] == ["passage: [EMPTY]"]


def test_aiberm_provider_uses_openai_sdk_configuration(monkeypatch):
    class FakeEmbeddings:
        def create(self, model, input):
            class Item:
                def __init__(self, index, embedding):
                    self.index = index
                    self.embedding = embedding

            class Response:
                data = [Item(1, [0.0, 1.0]), Item(0, [1.0, 0.0])]

            assert model == "text-embedding-3-small"
            assert input == ["first", "second"]
            return Response()

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.embeddings = FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "openai", type("OpenAIModule", (), {"OpenAI": FakeClient}))
    monkeypatch.setenv("AIBERM_API_KEY", "test-key")
    provider = AibermProvider()
    vectors = provider.embed("text-embedding-3-small", ["first", "second"])
    assert provider.provider_id == "aiberm"
    assert provider.base_url == "https://aiberm.com/v1"
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_aiberm_config_provider_type(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "test-key")
    provider = create_provider(
        {"type": "aiberm", "api_key_env": "AIBERM_API_KEY"}
    )
    assert provider.provider_id == "aiberm"
