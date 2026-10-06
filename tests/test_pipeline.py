import sys
sys.path.insert(0, "src")
import numpy as np
from dense_retrieval.embeddings import HashEmbeddingModel
from dense_retrieval.evaluation import evaluate_embeddings, run_oracle_correction

def test_normalization_and_multi_positive():
    model = HashEmbeddingModel(16)
    q = model.encode_queries(["one"]); d = model.encode_documents(["a", "b", "c"])
    assert np.allclose(np.linalg.norm(q, axis=1), 1); assert np.allclose(np.linalg.norm(d, axis=1), 1)
    metrics, frame = evaluate_embeddings(["q"], {"q": {"a": 1, "b": 1}}, q, d, ["a", "b", "c"], search_k=3)
    assert frame.loc[0, "best_relevant_rank"] == min(frame.loc[0, "best_relevant_rank"], 3)

def test_oracle_lambda_zero_matches_baseline():
    model = HashEmbeddingModel(8); q = model.encode_queries(["q"]); d = model.encode_documents(["a", "b"])
    _, frame = evaluate_embeddings(["q"], {"q": {"a": 1}}, q, d, ["a", "b"], search_k=2)
    oracle = run_oracle_correction(["q"], {"q": {"a": 1}}, q, d, ["a", "b"], lambdas=[0], search_k=2)
    assert int(oracle.iloc[0].original_rank) == int(frame.iloc[0].best_relevant_rank)

def test_positive_score_is_available_when_positive_is_outside_search_k():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [0.9, 0.4358899]], dtype=np.float32)
    _, frame = evaluate_embeddings(["q"], {"q": {"d2": 1}}, q, d, ["d1", "d2"], search_k=1)
    assert frame.loc[0, "best_relevant_rank"] == 2
    assert abs(frame.loc[0, "best_relevant_score"] - 0.9) < 1e-6
