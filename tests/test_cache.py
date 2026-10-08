import numpy as np

from dense_retrieval.embeddings import HashEmbeddingModel, encode_with_resume, load_cache, save_cache


def test_cache_resume_reuses_batch_parts(tmp_path):
    model = HashEmbeddingModel(4)
    model.cache_batch_size = 2
    path = tmp_path / "queries.npy"
    first = encode_with_resume(model, ["a", "b", "c"], "query", path)
    path.unlink()
    second = encode_with_resume(model, ["a", "b", "c"], "query", path)
    assert np.array_equal(first, second)


def test_manifest_order_matches_arrays(tmp_path):
    manifest = {"query_ids": ["q1"], "document_ids": ["d1"], "dataset": "toy"}
    save_cache(tmp_path, np.ones((1, 2)), np.zeros((1, 2)), manifest)
    queries, documents, loaded = load_cache(tmp_path)
    assert loaded == manifest
    assert queries.shape == documents.shape == (1, 2)
