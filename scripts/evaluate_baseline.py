from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.datasets import load_beir_dataset
from dense_retrieval.embeddings import load_cache
from dense_retrieval.evaluation import evaluate_baseline


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Evaluate frozen embeddings with exact baseline retrieval.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--missing-relevant-policy", choices=("error", "keep"), default=None)
    args = parser.parse_args()

    queries, documents, manifest = load_cache(args.cache)
    missing_policy = args.missing_relevant_policy or manifest.get("missing_relevant_policy", "error")
    dataset = load_beir_dataset(
        manifest["dataset"],
        args.dataset_root,
        manifest["split"],
        missing_relevant_policy=missing_policy,
    )
    query_ids = [sample.query_id for sample in dataset.queries]
    document_ids = list(dataset.corpus)
    if query_ids != manifest["query_ids"] or document_ids != manifest["document_ids"]:
        raise ValueError("dataset ID order does not match the embedding cache manifest")
    metrics = evaluate_baseline(dataset, queries, documents, document_ids, args.search_k)
    output = {
        "dataset": manifest["dataset"], "split": manifest["split"],
        "model_id": manifest["model_id"], "model_name": manifest.get("model_name"),
        "revision": manifest["revision"], "dimension": manifest["dimension"],
        "cache_fingerprint": manifest["cache_fingerprint"], "search_k": args.search_k,
        **metrics,
    }
    model_name = manifest.get("model_name") or manifest["model_id"].replace("/", "__")
    out_dir = Path(args.results_root) / manifest["dataset"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
