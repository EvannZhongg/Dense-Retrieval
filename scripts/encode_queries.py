from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.datasets import load_beir_dataset
from dense_retrieval.embeddings import (
    cache_fingerprint,
    create_embedding_model,
    encode_with_resume,
    load_model_config,
    save_cache,
)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Encode only BEIR queries for adaptation training; no document vectors are written."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--missing-relevant-policy", choices=("error", "keep"), default="error")
    args = parser.parse_args()

    dataset = load_beir_dataset(
        args.dataset,
        args.dataset_root,
        args.split,
        args.download,
        missing_relevant_policy=args.missing_relevant_policy,
    )
    model = create_embedding_model(load_model_config(args.model, args.models_config))
    fingerprint = cache_fingerprint(dataset.name, dataset.split, model)
    # Keep query-only training artifacts separate from a full frozen index cache
    # with the same dataset/model/split fingerprint.
    cache_dir = Path(args.cache_root) / dataset.name / args.model / f"{fingerprint}-query-only"
    query_ids = [sample.query_id for sample in dataset.queries]
    queries = encode_with_resume(model, dataset.query_texts, "query", cache_dir / "queries.npy")
    manifest = {
        "dataset": dataset.name,
        "split": dataset.split,
        "model_id": model.model_id,
        "model_name": args.model,
        "revision": model.revision,
        "dimension": int(model.dimension),
        "encoding_config": model.encoding_config,
        "query_ids": query_ids,
        "document_ids": [],
        "cache_fingerprint": fingerprint,
        "missing_relevant_policy": args.missing_relevant_policy,
        "cache_kind": "query_only",
    }
    documents = np.empty((0, int(model.dimension)), dtype=np.float32)
    save_cache(cache_dir, queries, documents, manifest)
    print(json.dumps({"cache": str(cache_dir), "queries": len(queries), "documents": 0, "fingerprint": fingerprint}, indent=2))


if __name__ == "__main__":
    main()
