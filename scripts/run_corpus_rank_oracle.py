"""Evaluate corpus-specific and same-query rank oracles in a shared B128.

The oracle uses a fixed query and four corpus variants.  For each variant it
optimizes a separate low-rank coordinate ``a`` in
``Normalize(q + mu + B @ a)``.  A second optimization forces one ``a`` to be
shared by all four variants for that query.  The difference between those
metrics is the corpus-specific rank signal that cannot be explained by a
query-only correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
    fit_pca_subspace,
)
from dense_retrieval.analysis.study_data import (  # noqa: E402
    other_documents,
    prepare_three_way,
)
from dense_retrieval.datasets import Document, RetrievalDataset  # noqa: E402
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402


METRIC_COLUMNS = [
    "HitRate@5",
    "HitRate@10",
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "NDCG@10",
]


def fixed_query_indices(query_ids, count: int) -> np.ndarray:
    order = sorted(
        range(len(query_ids)),
        key=lambda index: hashlib.sha256(
            f"corpus-rank-oracle-v1:{query_ids[index]}".encode()
        ).digest(),
    )
    return np.asarray(order[: min(count, len(order))], dtype=np.int64)


def query_balanced_external_indices(
    external: np.ndarray,
    queries: np.ndarray,
    per_query: int,
) -> np.ndarray:
    """Select each query's nearest external documents, then take their union.

    A single global ordering by maximum query similarity can devote nearly the
    entire external-document budget to a few queries. Equal per-query budgets
    make the counterfactual perturbation local to every query without using
    relevance labels.
    """
    if per_query <= 0 or len(external) == 0 or len(queries) == 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(per_query), len(external))
    scores = np.asarray(queries, dtype=np.float32) @ np.asarray(
        external, dtype=np.float32
    ).T
    if count == len(external):
        selected = np.broadcast_to(np.arange(len(external)), (len(queries), count))
    else:
        selected = np.argpartition(-scores, kth=count - 1, axis=1)[:, :count]
    return np.unique(selected.reshape(-1))


def build_variants(
    documents: np.ndarray,
    external: np.ndarray,
    queries: np.ndarray,
    *,
    random_external: int,
    hard_external_per_query: int,
    high_external_per_query: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    random_indices = rng.choice(
        len(external), size=min(random_external, len(external)), replace=False
    )
    hard_indices = query_balanced_external_indices(
        external, queries, hard_external_per_query
    )
    high_indices = query_balanced_external_indices(
        external, queries, high_external_per_query
    )
    return {
        "D0_original": np.asarray(documents),
        "D1_random_external": np.concatenate(
            [np.asarray(documents), external[random_indices]], axis=0
        ),
        "D2_query_local_external": np.concatenate(
            [np.asarray(documents), external[hard_indices]],
            axis=0,
        ),
        "D3_high_query_local_external": np.concatenate(
            [np.asarray(documents), external[high_indices]],
            axis=0,
        ),
    }


def external_intrusion_diagnostics(
    queries: np.ndarray,
    documents: np.ndarray,
    native_document_count: int,
    cutoffs: tuple[int, ...] = (10, 256),
) -> dict[str, float]:
    """Measure whether added documents reach the query's retrieval neighborhood."""
    external_count = len(documents) - int(native_document_count)
    result = {"external_documents": int(max(external_count, 0))}
    if external_count <= 0:
        for cutoff in cutoffs:
            result[f"queries_with_external_top{cutoff}"] = 0.0
            result[f"mean_external_count_top{cutoff}"] = 0.0
        result["mean_external_margin_over_native_top10"] = float("nan")
        return result

    scores = np.asarray(queries, dtype=np.float32) @ np.asarray(
        documents, dtype=np.float32
    ).T
    for cutoff in cutoffs:
        count = min(int(cutoff), len(documents))
        top = np.argpartition(-scores, kth=count - 1, axis=1)[:, :count]
        external_per_query = np.sum(top >= native_document_count, axis=1)
        result[f"queries_with_external_top{cutoff}"] = float(
            np.mean(external_per_query > 0)
        )
        result[f"mean_external_count_top{cutoff}"] = float(
            np.mean(external_per_query)
        )

    native_scores = scores[:, :native_document_count]
    external_scores = scores[:, native_document_count:]
    native_cutoff = min(10, native_document_count)
    native_top = np.partition(
        native_scores, kth=native_document_count - native_cutoff, axis=1
    )[:, -native_cutoff:]
    margin = np.max(external_scores, axis=1) - np.min(native_top, axis=1)
    result["mean_external_margin_over_native_top10"] = float(np.mean(margin))
    return result


def positive_indices(dataset, query_rows: np.ndarray) -> list[list[int]]:
    document_index = {str(doc_id): index for index, doc_id in enumerate(dataset.corpus)}
    result = []
    for row in query_rows:
        sample = dataset.queries[int(row)]
        values = [
            document_index[str(doc_id)]
            for doc_id, relevance in dataset.qrels[str(sample.query_id)].items()
            if int(relevance) > 0 and str(doc_id) in document_index
        ]
        result.append(values)
    return result


def build_rank_records(
    variant_documents: np.ndarray,
    queries: np.ndarray,
    positives: list[list[int]],
    hard_negatives: int,
) -> dict[str, np.ndarray]:
    scores = queries @ variant_documents.T
    max_positive = max((len(value) for value in positives), default=0)
    if max_positive == 0:
        raise ValueError("fixed queries contain no available positives")
    positive_array = np.full((len(queries), max_positive), -1, dtype=np.int64)
    negative_array = np.full((len(queries), hard_negatives), -1, dtype=np.int64)
    for row, values in enumerate(positives):
        positive_array[row, : len(values)] = values
        positive_set = set(values)
        order = np.argsort(-scores[row], kind="mergesort")
        negatives = [int(index) for index in order if int(index) not in positive_set]
        negative_array[row, : min(hard_negatives, len(negatives))] = negatives[:hard_negatives]
    return {
        "positive_indices": positive_array,
        "negative_indices": negative_array,
    }


def masked_rank_loss(corrected, positive_docs, negative_docs, positive_mask, negative_mask):
    positive_scores = torch.einsum("bd,bpd->bp", corrected, positive_docs)
    negative_scores = torch.einsum("bd,bnd->bn", corrected, negative_docs)
    positive_scores = positive_scores.masked_fill(~positive_mask, -torch.inf)
    negative_scores = negative_scores.masked_fill(~negative_mask, -torch.inf)
    numerator = torch.logsumexp(positive_scores, dim=1)
    denominator = torch.logsumexp(
        torch.cat([positive_scores, negative_scores], dim=1), dim=1
    )
    return -(numerator - denominator).mean()


def optimize_coordinates(
    queries: np.ndarray,
    records: list[dict[str, np.ndarray]],
    mean: np.ndarray,
    components: np.ndarray,
    *,
    beta: float,
    steps: int,
    learning_rate: float,
    shared: bool,
) -> tuple[np.ndarray, float]:
    """Optimize one coordinate row per query, or one shared row per query."""
    if not records:
        raise ValueError("at least one rank record is required")
    torch.set_num_threads(1)
    q = torch.from_numpy(np.asarray(queries, dtype=np.float32))
    mu = torch.from_numpy(np.asarray(mean, dtype=np.float32))
    basis = torch.from_numpy(np.asarray(components, dtype=np.float32).T)
    coordinates = torch.zeros(
        (len(queries), components.shape[0]), dtype=torch.float32, requires_grad=True
    )
    tensors = []
    for record in records:
        positives = torch.from_numpy(record["positive_docs"].astype(np.float32))
        negatives = torch.from_numpy(record["negative_docs"].astype(np.float32))
        positive_mask = torch.from_numpy(record["positive_mask"])
        negative_mask = torch.from_numpy(record["negative_mask"])
        tensors.append((positives, negatives, positive_mask, negative_mask))
    optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
    for _ in range(int(steps)):
        corrected = F.normalize(q + mu + coordinates @ basis.T, dim=1)
        losses = [
            masked_rank_loss(corrected, positives, negatives, positive_mask, negative_mask)
            for positives, negatives, positive_mask, negative_mask in tensors
        ]
        rank_loss = torch.stack(losses).mean() if shared else losses[0]
        loss = rank_loss + float(beta) * torch.mean(coordinates * coordinates)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        corrected = F.normalize(q + mu + coordinates @ basis.T, dim=1)
        losses = [
            masked_rank_loss(
                corrected, positives, negatives, positive_mask, negative_mask
            )
            for positives, negatives, positive_mask, negative_mask in tensors
        ]
        final_rank_loss = torch.stack(losses).mean() if shared else losses[0]
    return coordinates.detach().cpu().numpy(), float(final_rank_loss)


def rank_loss_at_coordinates(
    queries: np.ndarray,
    record: dict[str, np.ndarray],
    mean: np.ndarray,
    components: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """Evaluate one variant's ranking loss at fixed low-rank coordinates."""
    with torch.no_grad():
        corrected = F.normalize(
            torch.from_numpy(np.asarray(queries, dtype=np.float32))
            + torch.from_numpy(np.asarray(mean, dtype=np.float32))
            + torch.from_numpy(np.asarray(coordinates, dtype=np.float32))
            @ torch.from_numpy(np.asarray(components, dtype=np.float32)),
            dim=1,
        )
        return float(
            masked_rank_loss(
                corrected,
                torch.from_numpy(record["positive_docs"].astype(np.float32)),
                torch.from_numpy(record["negative_docs"].astype(np.float32)),
                torch.from_numpy(record["positive_mask"]),
                torch.from_numpy(record["negative_mask"]),
            )
        )


def correction_dataset(base_dataset, variant_name: str, document_count: int) -> RetrievalDataset:
    corpus = {
        str(doc_id): Document(str(doc_id), "", "")
        for doc_id in base_dataset.corpus
    }
    corpus.update(
        {
            f"external:{index}": Document(f"external:{index}", "", "")
            for index in range(document_count - len(corpus))
        }
    )
    return RetrievalDataset(
        f"{base_dataset.name}_{variant_name}",
        base_dataset.queries,
        corpus,
        base_dataset.qrels,
    )


def evaluate_variant(dataset, queries, documents, batch_size):
    return evaluate_retrieval(dataset, queries, documents, batch_size)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    coordinate_rows = []
    geometry_rows = []
    metadata = []
    for model_key in args.models:
        pooled_deltas = []
        for dataset_name in args.datasets:
            train_dataset, _dev_dataset, _test_dataset, train_queries, _dev_queries, _test_queries, documents, _cache, _split = prepare_three_way(
                dataset_name, model_key, args.cache_root
            )
            _rows, deltas, _missing = build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            pooled_deltas.append(deltas)
        mean, components = fit_pca_subspace(
            np.concatenate(pooled_deltas, axis=0), args.rank
        )

        (
            _train_dataset,
            _dev_dataset,
            test_dataset,
            _train_queries,
            _dev_queries,
            test_queries,
            documents,
            _cache_dir,
            _split,
        ) = prepare_three_way("fiqa", model_key, args.cache_root)
        query_ids = [sample.query_id for sample in test_dataset.queries]
        query_rows = fixed_query_indices(query_ids, args.fixed_queries)
        queries = np.asarray(test_queries)[query_rows].astype(np.float32)
        fixed_dataset = RetrievalDataset(
            "fiqa_fixed_oracle",
            [test_dataset.queries[int(row)] for row in query_rows],
            test_dataset.corpus,
            {test_dataset.queries[int(row)].query_id: test_dataset.qrels[test_dataset.queries[int(row)].query_id] for row in query_rows},
        )
        positives = positive_indices(test_dataset, query_rows)
        external = other_documents("fiqa", model_key, args.cache_root)
        variants = build_variants(
            documents,
            external,
            queries,
            random_external=args.random_external,
            hard_external_per_query=args.hard_external_per_query,
            high_external_per_query=args.high_external_per_query,
            random_state=args.random_state,
        )
        records_by_variant = {}
        datasets_by_variant = {}
        baselines = {}
        for variant_name, variant_documents in variants.items():
            geometry_rows.append(
                {
                    "model": model_key,
                    "variant": variant_name,
                    **external_intrusion_diagnostics(
                        queries, variant_documents, len(documents)
                    ),
                }
            )
            records = build_rank_records(
                variant_documents, queries, positives, args.hard_negatives
            )
            positive_docs = np.zeros(
                (len(queries), records["positive_indices"].shape[1], variant_documents.shape[1]),
                dtype=np.float32,
            )
            negative_docs = np.zeros(
                (len(queries), args.hard_negatives, variant_documents.shape[1]),
                dtype=np.float32,
            )
            positive_mask = records["positive_indices"] >= 0
            negative_mask = records["negative_indices"] >= 0
            for row in range(len(queries)):
                positive_docs[row] = variant_documents[np.maximum(records["positive_indices"][row], 0)]
                negative_docs[row] = variant_documents[np.maximum(records["negative_indices"][row], 0)]
            rank_record = {
                "positive_docs": positive_docs,
                "negative_docs": negative_docs,
                "positive_mask": positive_mask,
                "negative_mask": negative_mask,
            }
            records_by_variant[variant_name] = rank_record
            datasets_by_variant[variant_name] = correction_dataset(
                fixed_dataset, variant_name, len(variant_documents)
            )
            baselines[variant_name] = evaluate_variant(
                datasets_by_variant[variant_name], queries, variant_documents, args.batch_size
            )

        corpus_coordinates = {}
        corpus_losses = {}
        for variant_name, record in records_by_variant.items():
            coordinates, loss = optimize_coordinates(
                queries,
                [record],
                mean,
                components,
                beta=args.beta,
                steps=args.steps,
                learning_rate=args.learning_rate,
                shared=False,
            )
            corpus_coordinates[variant_name] = coordinates
            corpus_losses[variant_name] = loss
        shared_coordinates, shared_loss = optimize_coordinates(
            queries,
            [records_by_variant[name] for name in variants],
            mean,
            components,
            beta=args.beta,
            steps=args.steps,
            learning_rate=args.learning_rate,
            shared=True,
        )

        for variant_name, variant_documents in variants.items():
            shared_variant_loss = rank_loss_at_coordinates(
                queries,
                records_by_variant[variant_name],
                mean,
                components,
                shared_coordinates,
            )
            corpus_objective = corpus_losses[variant_name] + args.beta * float(
                np.mean(corpus_coordinates[variant_name] ** 2)
            )
            shared_variant_objective = shared_variant_loss + args.beta * float(
                np.mean(shared_coordinates**2)
            )
            corpus_query = F.normalize(
                torch.from_numpy(queries)
                + torch.from_numpy(mean.astype(np.float32))
                + torch.from_numpy(corpus_coordinates[variant_name].astype(np.float32))
                @ torch.from_numpy(components.astype(np.float32)),
                dim=1,
            ).numpy()
            shared_query = F.normalize(
                torch.from_numpy(queries)
                + torch.from_numpy(mean.astype(np.float32))
                + torch.from_numpy(shared_coordinates.astype(np.float32))
                @ torch.from_numpy(components.astype(np.float32)),
                dim=1,
            ).numpy()
            corpus_metrics = evaluate_variant(
                datasets_by_variant[variant_name], corpus_query, variant_documents, args.batch_size
            )
            shared_metrics = evaluate_variant(
                datasets_by_variant[variant_name], shared_query, variant_documents, args.batch_size
            )
            for metric in METRIC_COLUMNS:
                rows.extend(
                    [
                        {
                            "model": model_key,
                            "variant": variant_name,
                            "oracle": "baseline",
                            "metric": metric,
                            "value": baselines[variant_name][metric],
                        },
                        {
                            "model": model_key,
                            "variant": variant_name,
                            "oracle": "corpus_oracle",
                            "metric": metric,
                            "value": corpus_metrics[metric],
                        },
                        {
                            "model": model_key,
                            "variant": variant_name,
                            "oracle": "q_only_oracle",
                            "metric": metric,
                            "value": shared_metrics[metric],
                        },
                        {
                            "model": model_key,
                            "variant": variant_name,
                            "oracle": "H_corpus",
                            "metric": metric,
                            "value": corpus_metrics[metric] - shared_metrics[metric],
                        },
                    ]
                )
            coordinate_rows.extend(
                {
                    "model": model_key,
                    "variant": variant_name,
                    "oracle": "corpus_oracle",
                    "coordinate_norm_mean": float(np.mean(np.linalg.norm(corpus_coordinates[variant_name], axis=1))),
                    "corpus_rank_loss": corpus_losses[variant_name],
                    "shared_rank_loss": shared_variant_loss,
                    "H_rank_loss": shared_variant_loss - corpus_losses[variant_name],
                    "corpus_regularized_objective": corpus_objective,
                    "shared_regularized_objective": shared_variant_objective,
                    "H_regularized_objective": (
                        shared_variant_objective - corpus_objective
                    ),
                    "coordinate_distance_from_shared": float(
                        np.mean(
                            np.linalg.norm(
                                corpus_coordinates[variant_name] - shared_coordinates,
                                axis=1,
                            )
                        )
                    ),
                }
                for _ in [0]
            )
        metadata.append(
            {
                "model": model_key,
                "datasets_for_global_mu_B": list(args.datasets),
                "rank": args.rank,
                "fixed_queries": len(queries),
                "n_anchors": 256,
                "hard_negatives": args.hard_negatives,
                "random_external": args.random_external,
                "hard_external_per_query": args.hard_external_per_query,
                "high_external_per_query": args.high_external_per_query,
                "beta": args.beta,
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "same_query_shared_across_variants": True,
                "shared_rank_loss": shared_loss,
                "external_selection_uses_qrels": False,
                "test_qrels_used_for_mu_B": False,
            }
        )
        print(f"{model_key}: complete", flush=True)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.output_dir / "corpus_rank_oracle_metrics_long.csv", index=False)
    summary = metrics_frame.pivot_table(
        index=["model", "variant", "metric"],
        columns="oracle",
        values="value",
        aggfunc="first",
    ).reset_index()
    summary.to_csv(args.output_dir / "corpus_rank_oracle_summary.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(args.output_dir / "corpus_rank_oracle_coordinates.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(
        args.output_dir / "corpus_variant_geometry.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "corpus_rank_oracle")
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"])
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--fixed-queries", type=int, default=64)
    parser.add_argument("--random-external", type=int, default=1000)
    parser.add_argument("--hard-external-per-query", type=int, default=16)
    parser.add_argument("--high-external-per-query", type=int, default=64)
    parser.add_argument("--hard-negatives", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
