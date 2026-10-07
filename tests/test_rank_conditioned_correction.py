import importlib.util
from pathlib import Path

import numpy as np
import torch

from dense_retrieval.analysis.shared_anchors import SharedAnchorCodebook


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_rank_conditioned_correction.py"
SPEC = importlib.util.spec_from_file_location("rank_conditioned_correction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ranking_loss_is_lower_when_positive_beats_hard_negative():
    good = MODULE.multi_positive_hard_negative_loss(
        torch.tensor([[2.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
    )
    bad = MODULE.multi_positive_hard_negative_loss(
        torch.tensor([[0.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
    )
    assert float(good) < float(bad)


def test_rank_records_exclude_positives_from_current_corpus_negatives():
    class Sample:
        def __init__(self, query_id):
            self.query_id = query_id

    class Dataset:
        name = "toy"
        queries = [Sample("q1")]
        corpus = {"p": object(), "n1": object(), "n2": object()}
        qrels = {"q1": {"p": 1}}

    documents = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32
    )
    records = MODULE.build_rank_records(
        Dataset(), np.asarray([[1.0, 0.0]], dtype=np.float32), documents, hard_k=2
    )
    assert records["positive_indices"].tolist() == [[0]]
    assert 0 not in records["negative_indices"][0]
    assert records["negative_indices"].shape == (1, 2)


def test_corpus_sketch_preserves_per_anchor_occupancy_features():
    rng = np.random.default_rng(4)
    documents = {"a": rng.normal(size=(20, 4)), "b": rng.normal(size=(24, 4))}
    codebook = SharedAnchorCodebook.fit(
        documents, n_anchors=5, top_m=3, projection=np.eye(4)[:2], random_state=3
    )
    first = codebook.occupancy(documents["a"])
    second = codebook.occupancy(documents["b"])
    queries = rng.normal(size=(3, 4))
    first_sketch = MODULE.corpus_sketch(queries, codebook, first)
    second_sketch = MODULE.corpus_sketch(queries, codebook, second)
    assert first_sketch.shape == (3, 3 * 4)
    assert not np.allclose(first_sketch, second_sketch)
