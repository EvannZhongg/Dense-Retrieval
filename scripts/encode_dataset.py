from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Encode a BEIR dataset into a frozen embedding cache.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="Name from configs/models.yaml")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--split", default="test")
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
    cache_dir = Path(args.cache_root) / dataset.name / args.model / fingerprint
    query_ids = [sample.query_id for sample in dataset.queries]
    query_texts = [sample.query_text for sample in dataset.queries]
    document_ids = list(dataset.corpus)
    document_texts = [f"{dataset.corpus[doc_id].title}\n{dataset.corpus[doc_id].text}".strip() for doc_id in document_ids]
    queries = encode_with_resume(model, query_texts, "query", cache_dir / "queries.npy")
    documents = encode_with_resume(model, document_texts, "document", cache_dir / "documents.npy")
    manifest = {
        "dataset": dataset.name,
        "split": dataset.split,
        "model_id": model.model_id,
        "model_name": args.model,
        "revision": model.revision,
        "dimension": int(model.dimension),
        "encoding_config": model.encoding_config,
        "query_ids": query_ids,
        "document_ids": document_ids,
        "cache_fingerprint": fingerprint,
        "missing_relevant_policy": args.missing_relevant_policy,
    }
    save_cache(cache_dir, queries, documents, manifest)
    print(json.dumps({"cache": str(cache_dir), "queries": len(queries), "documents": len(documents), "fingerprint": fingerprint}, indent=2))


if __name__ == "__main__":
    main()
