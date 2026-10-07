import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dense_retrieval.analysis.linear_ranking import (
    LinearRankCorrection,
    predict_coordinates,
    train_linear_rank_model,
)


def test_linear_rank_correction_starts_at_exact_zero():
    model = LinearRankCorrection(input_dimension=3, rank=2)
    values = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.0, 2.0]])

    coordinates = model(values)

    assert torch.equal(coordinates, torch.zeros((2, 2)))


def test_predict_coordinates_returns_numpy_rows():
    model = LinearRankCorrection(input_dimension=2, rank=1)
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor([[2.0, -1.0]])
        model.linear.bias[:] = torch.tensor([0.5])

    prediction = predict_coordinates(model, np.asarray([[1.0, 3.0]]))

    assert prediction.shape == (1, 1)
    assert np.allclose(prediction, [[-0.5]])


def test_linear_rank_training_improves_positive_negative_margin():
    class Sample:
        query_id = "q"

    class Dataset:
        name = "toy"
        queries = [Sample()]
        corpus = {"positive": object(), "negative": object()}
        qrels = {"q": {"positive": 1}}

    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    documents = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    corpora = {
        "toy": {
            "train_dataset": Dataset(),
            "train_queries": query,
            "train_features": np.asarray([[1.0]], dtype=np.float32),
            "documents": documents,
        }
    }
    model, history = train_linear_rank_model(
        corpora,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        epochs=20,
        batch_size=1,
        train_lambdas=[1.0],
        temperature=0.1,
        learning_rate=0.1,
        weight_decay=0.0,
        coordinate_penalty=0.0,
        hard_negatives=1,
        random_state=0,
        device="cpu",
    )
    coordinate = predict_coordinates(model, np.asarray([[1.0]], dtype=np.float32))
    corrected = query + coordinate @ np.asarray([[0.0, 1.0]], dtype=np.float32)
    corrected /= np.linalg.norm(corrected, axis=1, keepdims=True)
    baseline_margin = float((query @ (documents[0] - documents[1])).item())
    corrected_margin = float((corrected @ (documents[0] - documents[1])).item())

    assert history[-1] < history[0]
    assert corrected_margin > baseline_margin
