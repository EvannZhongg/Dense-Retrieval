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

Run the complete FiQA test split with BGE-M3 through SiliconFlow:

```powershell
python scripts/run_experiment.py --config configs/fiqa_bge_m3_siliconflow.yaml
python scripts/analyze_results.py --results results --output results/summary.json
python scripts/plot_results.py --results results
```

Run the complete FiQA test split with E5 Base v2 through OpenRouter:

```powershell
python scripts/run_experiment.py --config configs/fiqa_e5_base_v2_openrouter.yaml
python scripts/analyze_results.py --results results --output results/summary.json
python scripts/plot_results.py --results results
```

Set `OPENROUTER_API_KEY` in the project-root `.env` before running. E5 queries use the `query:` prefix and corpus documents use the `passage:` prefix.

Replace the `model` block in the YAML with a block from `configs/models.yaml`; set `dataset.name` to `fiqa`, `scifact`, or `nfcorpus` and `download: true` when the dataset is not present locally. Hugging Face authentication or mirrors can be supplied through the normal `.env`/environment configuration used by the model hub.

Outputs are written under `results/<dataset>/<model>/`: `metrics.json`, one-row-per-query `query_metrics.parquet`, `oracle_correction.parquet`, the exact run `config.json`, and a per-experiment `plots/` directory. Embedding caches are keyed by dataset, model id/revision, dimension, and encoding configuration.

Reference runs for FiQA and ArguAna across the Qwen3-Embedding, BGE-M3, and E5-Base-v2 adapters are committed under `results/`, together with the aggregated `results/summary.json` and the cross-model PCA analysis in `results/delta_pca/`. Only the embedding cache under `cache/` is excluded, so those runs can be inspected without re-encoding any corpus. The `cache_dir` field recorded in `results/delta_pca/metadata.json` refers to the local cache layout of the machine that produced it and is informational only.

`HitRate@K` is the fraction of queries with at least one relevant document in the top K. `Recall@K` is macro-averaged across queries as `retrieved relevant / all relevant`. The per-query parquet stores both values and their underlying counts.

The first version intentionally uses exact in-memory retrieval. PostgreSQL/pgvector can be added behind the retrieval interface later without changing dataset/model adapters.

Prototype Correction Field
--------------------------

The frozen corpus can also provide a compact query-side correction prior. Run
`python scripts/run_prototype_correction.py --datasets fiqa --models e5` to fit
64 spherical K-Means document prototypes, retain the Top-8 prototype weights
for each query, and learn one correction coordinate vector per prototype from
the training oracle coordinates. The script saves the field under
`results/prototype_correction/models/` and evaluates the one-pass corrected
query against the unchanged document index. Prototype fitting uses document
embeddings only; test qrels are not used to construct the field.

The reusable implementation is in
`dense_retrieval.analysis.prototype_correction`: `fit_spherical_kmeans`,
`prototype_weights`, `fit_prototype_values`, and
`PrototypeCorrectionField`. When correction values are low-rank coordinates,
pass the PCA `mean` and `components` to the field so it can reconstruct a
full-dimensional delta before applying it to a normalized query.

ArguAna is supported as a local BEIR dataset under `datasets/arguana`. The official release contains five qrels whose relevant document is absent from the official corpus. ArguAna configs explicitly use `missing_relevant_policy: keep`: those queries remain in the 1,406-query evaluation denominator, count as retrieval misses, expose missing-positive counts in per-query output, and have unavailable positive geometry/oracle fields.

NFCorpus and SciFact are supported as local BEIR datasets under `datasets/nfcorpus` and `datasets/scifact`. Ready-to-run test-split configs are provided for Qwen3, BGE-M3, and E5 Base v2.
