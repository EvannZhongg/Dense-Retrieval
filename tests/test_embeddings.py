import numpy as np

from dense_retrieval.embeddings import HashEmbeddingModel


def test_hash_embeddings_are_normalized_and_asymmetric():
    model = HashEmbeddingModel(8)
    queries = model.encode_queries(["same"])
    documents = model.encode_documents(["same"])
    assert queries.shape == (1, 8)
    assert documents.shape == (1, 8)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1)
    assert not np.array_equal(queries, documents)
