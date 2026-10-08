from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.adaptation import fit_corpus_alignment, fit_gold_supervised
from dense_retrieval.datasets import load_beir_dataset
from dense_retrieval.embeddings import (
    create_embedding_model,
    encode_with_resume,
    load_cache,
    load_model_config,
)
from dense_retrieval.evaluation import evaluate_baseline, evaluate_calibrator


def _dataset_for_cache(cache_manifest, root: str | Path, split: str):
    return load_beir_dataset(
        cache_manifest["dataset"], root, split,
        missing_relevant_policy=cache_manifest.get("missing_relevant_policy", "error"),
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Train/evaluate an index-preserving query-only linear calibrator."
    )
    parser.add_argument("--cache", required=True, help="Existing frozen cache directory")
    parser.add_argument("--mode", choices=("gold", "corpus"), required=True)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--model", default=None, help="Model registry name; required for corpus mode")
    parser.add_argument("--train-query-cache", default=None, help="Optional cache containing embeddings for the gold train split")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--adaptation-root", default="adaptation_results")
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--max-corpus-documents", type=int, default=None)
    args = parser.parse_args()

    query_embeddings, document_embeddings, manifest = load_cache(args.cache)
    dataset_name = manifest["dataset"]
    eval_split = args.eval_split or manifest["split"]
    eval_dataset = _dataset_for_cache(manifest, args.dataset_root, eval_split)
    query_ids = list(manifest["query_ids"])
    document_ids = list(manifest["document_ids"])
    if [sample.query_id for sample in eval_dataset.queries] != query_ids:
        raise ValueError("evaluation query ID order does not match cache manifest")
    if list(eval_dataset.corpus) != document_ids:
        raise ValueError("dataset document ID order does not match cache manifest")

    if args.mode == "gold":
        train_dataset = _dataset_for_cache(manifest, args.dataset_root, args.train_split)
        train_query_embeddings = query_embeddings
        train_query_ids = query_ids
        train_cache_manifest = None
        if args.train_query_cache:
            train_query_embeddings, _, train_cache_manifest = load_cache(args.train_query_cache)
            train_query_ids = list(train_cache_manifest["query_ids"])
            if train_cache_manifest.get("dimension") != manifest.get("dimension"):
                raise ValueError("train query cache dimension does not match evaluation cache")
        train_ids = [sample.query_id for sample in train_dataset.queries]
        if not set(train_ids).intersection(train_query_ids):
            raise ValueError(
                "gold training queries are absent from the evaluation cache; provide --train-query-cache "
                "encoded from the same frozen model and train split"
            )
        calibrator = fit_gold_supervised(
            train_query_embeddings,
            document_embeddings,
            train_query_ids,
            train_dataset.qrels,
            document_ids,
            train_query_ids=train_ids,
            alpha=args.alpha,
        )
        provenance = {
            "train_split": args.train_split,
            "label_source": "qrels_positive_centroid",
            "train_query_cache": args.train_query_cache,
        }
    else:
        if not args.model:
            raise ValueError("--model is required for corpus mode")
        model = create_embedding_model(load_model_config(args.model, args.models_config))
        if int(model.dimension) != int(query_embeddings.shape[1]):
            raise ValueError(
                "corpus query encoder dimension does not match the frozen query cache; "
                "use the same embedding model/protocol as the cache"
            )
        limit = args.max_corpus_documents or len(document_ids)
        if limit < 1 or limit > len(document_ids):
            raise ValueError("--max-corpus-documents must be between 1 and corpus size")
        texts = [f"{eval_dataset.corpus[doc_id].title}\n{eval_dataset.corpus[doc_id].text}".strip() for doc_id in document_ids[:limit]]
        source_path = (
            Path(args.adaptation_root)
            / dataset_name
            / args.model
            / str(manifest.get("cache_fingerprint", "unknown"))
            / "corpus_query_embeddings.npy"
        )
        source = encode_with_resume(model, texts, "query", source_path)
        calibrator = fit_corpus_alignment(source, document_embeddings[:limit], alpha=args.alpha)
        provenance = {
            "label_source": "same_document_query_protocol_alignment",
            "corpus_documents_used": limit,
            "query_corpus_embedding_path": str(source_path),
            "query_model": model.model_id,
            "query_model_revision": model.revision,
        }

    baseline = evaluate_baseline(eval_dataset, query_embeddings, document_embeddings, document_ids, args.search_k)
    calibrated = evaluate_calibrator(
        eval_dataset, query_embeddings, document_embeddings, document_ids, calibrator, args.search_k
    )
    output = {
        "dataset": dataset_name,
        "eval_split": eval_split,
        "cache": str(Path(args.cache)),
        "cache_fingerprint": manifest.get("cache_fingerprint"),
        "calibrator": calibrator.method,
        "alpha": args.alpha,
        "search_k": args.search_k,
        "baseline": baseline,
        "calibrated": calibrated,
        "delta": {key: calibrated[key] - baseline[key] for key in calibrated if key != "num_queries"},
        **provenance,
    }
    out_dir = (
        Path(args.adaptation_root)
        / dataset_name
        / (manifest.get("model_name") or "cache")
        / str(manifest.get("cache_fingerprint", "unknown"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    calibrator.save(out_dir / f"{args.mode}_calibrator.npz", metadata=output)
    (out_dir / f"{args.mode}_metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
