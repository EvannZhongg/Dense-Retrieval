# Dense Retrieval Embedding Baseline

This repository is the small data and embedding layer for later Dense Retrieval research. It loads BEIR data, creates query and document embeddings, stores a reproducible frozen cache, and evaluates that cache with exact inner-product retrieval. New research code should consume the cache rather than modify these modules.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Local Hugging Face models use sentence-transformers. Hosted models use provider settings in `configs/models.yaml` and an API key such as `SILICONFLOW_API_KEY`, `OPENROUTER_API_KEY`, or `AIBERM_API_KEY` in the environment.

## Dataset layout

Place a BEIR dataset under `datasets/<name>/` or `datasets/<name>/<name>/` with `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. Use `--download` for registered BEIR downloads.

## Model configuration

`configs/models.yaml` is the only model registry. `hash` is deterministic and useful for tests and smoke runs.

## Generate a cache

```bash
python scripts/encode_dataset.py --dataset fiqa --model qwen3-embedding-0.6b
```

Encoding is checkpointed in batches and resumes after interrupted hosted requests. A cache under `cache/<dataset>/<model>/<fingerprint>/` contains `queries.npy`, `documents.npy`, and `manifest.json`. The manifest records dataset/split, model ID/revision/dimension, encoding configuration, IDs in exact array order, and the cache fingerprint.

## Evaluate the frozen baseline

```bash
python scripts/evaluate_baseline.py --cache cache/fiqa/qwen3-embedding-0.6b/<fingerprint> --search-k 10
```

This reads existing embeddings only, performs exact inner-product retrieval, and writes `results/<dataset>/<model>/baseline.json` with HitRate, Recall, MRR, NDCG, and provenance. It never re-encodes data or runs query correction.

Run infrastructure tests with `pytest`.

## Query-side adaptation experiments

The independent `dense_retrieval.adaptation` package provides an
index-preserving ridge calibrator. It transforms only query vectors and leaves
the cached document vectors untouched. Use
`scripts/run_adaptation.py --mode gold` for a qrels-backed diagnostic reference
or `--mode corpus` for the no-qrels corpus-alignment baseline. Both commands
write metrics and a provenance sidecar next to the calibrator; see
`docs/query_adaptation_experiment.md` for split and leakage requirements.
