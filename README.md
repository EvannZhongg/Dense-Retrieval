# Dense Retrieval Baseline

This repository provides one config-driven baseline for BEIR FiQA, SciFact, and NFCorpus and for E5, Qwen3-Embedding, and GTE adapters. Dataset parsing, asymmetric model encoding, exact normalized-dot retrieval, per-query geometry, and oracle correction are separate modules.

Run from the project root (activate the project virtual environment first):

```powershell
python scripts/run_experiment.py --config configs/example.yaml
python scripts/analyze_results.py --results results --output results/summary.json
python scripts/plot_results.py --results results
```

Run the complete FiQA test split with Qwen3-Embedding through SiliconFlow:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_experiment.py --config configs/fiqa_qwen3_siliconflow.yaml
python scripts/analyze_results.py --results results --output results/summary.json
python scripts/plot_results.py --results results
```

The API key is read from `SILICONFLOW_API_KEY` in the project-root `.env`. Hosted encoding is checkpointed per batch under `cache/`, so rerunning the same configuration resumes completed API batches.

Replace the `model` block in the YAML with a block from `configs/models.yaml`; set `dataset.name` to `fiqa`, `scifact`, or `nfcorpus` and `download: true` when the dataset is not present locally. Hugging Face authentication or mirrors can be supplied through the normal `.env`/environment configuration used by the model hub.

Outputs are written under `results/<dataset>/<model>/`: `metrics.json`, one-row-per-query `query_metrics.parquet`, `oracle_correction.parquet`, the exact run `config.json`, and a per-experiment `plots/` directory. Embedding caches are keyed by dataset, model id/revision, dimension, and encoding configuration.

`HitRate@K` is the fraction of queries with at least one relevant document in the top K. `Recall@K` is macro-averaged across queries as `retrieved relevant / all relevant`. The per-query parquet stores both values and their underlying counts.

The first version intentionally uses exact in-memory retrieval. PostgreSQL/pgvector can be added behind the retrieval interface later without changing dataset/model adapters.
