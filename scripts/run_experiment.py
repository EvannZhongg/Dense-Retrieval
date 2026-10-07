from __future__ import annotations
import argparse, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dense_retrieval.datasets import load_beir_dataset
from dense_retrieval.embeddings import create_embedding_model
from dense_retrieval.evaluation import evaluate_embeddings, run_oracle_correction

def cache_key(dataset, model):
    payload = {"dataset": dataset, "model_id": model.model_id, "revision": model.revision, "dimension": model.dimension, "encoding_config": model.encoding_config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

def cached_encode(model, texts, path, kind, workers=1):
    path = Path(path)
    if path.exists():
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape == (len(texts), model.dimension): return arr.astype(np.float32)
    batch_size = getattr(model, "cache_batch_size", None)
    if batch_size:
        parts_dir = path.parent / f"{path.stem}_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        spans = [
            (start, min(start + batch_size, len(texts)))
            for start in range(0, len(texts), batch_size)
        ]
        missing = []
        for start, end in spans:
            end = min(start + batch_size, len(texts))
            part_path = parts_dir / f"{start:09d}_{end:09d}.npy"
            if part_path.exists():
                part = np.load(part_path)
                if part.shape != (end - start, model.dimension):
                    part_path.unlink()
                    missing.append((start, end, part_path))
            else:
                missing.append((start, end, part_path))

        def encode_part(item):
            start, end, part_path = item
            batch = texts[start:end]
            part = model.encode_queries(batch) if kind == "query" else model.encode_documents(batch)
            if part.shape != (end - start, model.dimension):
                raise ValueError(
                    f"{kind} batch shape {part.shape}, expected {(end - start, model.dimension)}"
                )
            np.save(part_path, part.astype(np.float32))
            return end - start

        if missing:
            completed = len(spans) - len(missing)
            workers = max(1, int(workers))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(encode_part, item) for item in missing]
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    print(
                        f"{kind}: completed batches {completed}/{len(spans)}",
                        flush=True,
                    )
        parts = [
            np.load(parts_dir / f"{start:09d}_{end:09d}.npy").astype(np.float32)
            for start, end in spans
        ]
        arr = np.concatenate(parts, axis=0) if parts else np.empty((0, model.dimension), dtype=np.float32)
    else:
        arr = model.encode_queries(texts) if kind == "query" else model.encode_documents(texts)
    if arr.shape != (len(texts), model.dimension): raise ValueError(f"{kind} embedding shape {arr.shape}, expected {(len(texts), model.dimension)}")
    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, arr); return arr

def run(config):
    ds_cfg, model_cfg = config["dataset"], config["model"]
    dataset = load_beir_dataset(
        ds_cfg["name"],
        ds_cfg.get("root", "datasets"),
        ds_cfg.get("split", "test"),
        ds_cfg.get("download", False),
        ds_cfg.get("missing_relevant_policy", "error"),
    )
    model = create_embedding_model(model_cfg); key = cache_key(dataset.name, model); cache = Path(config.get("cache_dir", "cache")) / dataset.name / model.model_id.replace("/", "__") / key
    query_ids = [q.query_id for q in dataset.queries]; query_texts = [q.query_text for q in dataset.queries]; doc_ids = list(dataset.corpus); doc_texts = [f"{dataset.corpus[d].title}\n{dataset.corpus[d].text}".strip() for d in doc_ids]
    workers = int(config.get("encoding_workers", 1))
    qemb = cached_encode(model, query_texts, cache / "queries.npy", "query", workers)
    demb = cached_encode(model, doc_texts, cache / "documents.npy", "document", workers)
    metrics, frame = evaluate_embeddings(query_ids, dataset.qrels, qemb, demb, doc_ids, float(config.get("temperature", 1.0)), int(config.get("search_k", 100)))
    oracle = run_oracle_correction(query_ids, dataset.qrels, qemb, demb, doc_ids, config.get("oracle_lambdas", [0, .01, .02, .05, .1, .2]), int(config.get("search_k", 100)))
    out = Path(config.get("results_dir", "results")) / dataset.name / model.model_id.split("/")[-1].lower(); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); frame.to_parquet(out / "query_metrics.parquet", index=False); oracle.to_parquet(out / "oracle_correction.parquet", index=False); (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(out), **metrics}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/example.yaml"); args = ap.parse_args()
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
