import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_corpus_rank_oracle.py"
SPEC = importlib.util.spec_from_file_location("corpus_rank_oracle", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_query_balanced_external_selection_covers_each_query_neighborhood():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    external = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )

    selected = MODULE.query_balanced_external_indices(external, queries, per_query=1)

    assert set(selected.tolist()) == {0, 2}


def test_query_local_variant_uses_per_query_external_union():
    documents = np.asarray([[0.7, 0.7]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    external = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )

    variants = MODULE.build_variants(
        documents,
        external,
        queries,
        random_external=1,
        hard_external_per_query=1,
        high_external_per_query=2,
        random_state=3,
    )

    assert len(variants["D2_query_local_external"]) == 3
    assert len(variants["D3_high_query_local_external"]) == 5


def test_external_intrusion_reports_local_effect():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = np.asarray(
        [[0.8, 0.2], [0.2, 0.8], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    diagnostics = MODULE.external_intrusion_diagnostics(
        queries, documents, native_document_count=2, cutoffs=(1, 2)
    )

    assert diagnostics["queries_with_external_top1"] == 1.0
    assert diagnostics["mean_external_count_top1"] == 1.0
    assert diagnostics["queries_with_external_top2"] == 1.0


def test_coordinate_optimizer_reports_loss_at_returned_coordinates():
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    record = {
        "positive_docs": np.asarray([[[0.0, 1.0]]], dtype=np.float32),
        "negative_docs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "positive_mask": np.asarray([[True]]),
        "negative_mask": np.asarray([[True]]),
    }
    mean = np.zeros(2, dtype=np.float32)
    components = np.eye(2, dtype=np.float32)

    coordinates, reported = MODULE.optimize_coordinates(
        queries,
        [record],
        mean,
        components,
        beta=1e-3,
        steps=2,
        learning_rate=0.1,
        shared=False,
    )
    recomputed = MODULE.rank_loss_at_coordinates(
        queries, record, mean, components, coordinates
    )

    assert np.isclose(reported, recomputed)
