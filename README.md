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

Shared Anchor Occupancy
-----------------------

For each embedding model, the shared-corpus experiments fit one fixed anchor
codebook from the union of all selected corpus document embeddings. The default
is `K=256`; each corpus stores only its anchor occupancy `p[D,k]`, plus optional
per-cell distance and low-dimensional shape statistics. A query uses the same
Top-M anchor indices and query-side anchor features for every corpus, while the
`log(p[D,k])` feature changes with the corpus. This keeps the semantic regions
comparable when the knowledge base changes. The reusable implementation is in
`dense_retrieval.analysis.shared_anchors`.

Run the shared regression experiment with:
`python scripts/run_shared_corpus_correction.py --models e5`

The rank-conditioned experiment uses the same global codebook:
`python scripts/run_rank_conditioned_correction.py --models e5`

Prototype Correction Field
--------------------------

The legacy single-corpus command can also provide a compact query-side
correction prior. Shared multi-corpus runs should use the anchor-occupancy
commands above. Run
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

Ranking-supervised corpus conditioning
---------------------------------------

The ranking trainer and exact retriever use CUDA automatically when the
installed PyTorch build exposes it.  Override with `--device cpu|cuda|auto`;
`auto` is the default.  On a CUDA-capable Windows machine, install a CUDA
wheel in the project environment before running the study, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

The experiment metadata records the resolved device.  If `torch.cuda.is_available()`
is false, training and retrieval fall back to CPU without changing the index.

The regression target `d_positive - q` is corpus-independent, so it cannot
learn that a query may need a different movement when a corpus contains extra
hard negatives. `scripts/run_rank_conditioned_correction.py` addresses this by
fitting one global `mu,B` from the pooled training corpora and optimizing the
query correction directly against frozen-corpus positive and baseline hard
negative scores. The script compares `rank_q_only` with
`rank_corpus_sketch`; the latter receives the current corpus's Top-M shared-anchor
sketch with occupancy features, and its negatives are generated from that same corpus. Both models use
the same train/dev/test splits, architecture, ranking loss, and macro-dev
lambda selection. The ranking path now preserves flattened per-anchor
`[B.T(anchor-query), query-anchor-similarity, log_occupancy]` features instead
of pooling the anchors into a query-only vector. Its ranking loss uses
temperature `0.05`, and training samples correction magnitudes from the same
lambda set used during dev selection.

The corrected trainer refreshes hard negatives after every epoch, balances
optimizer updates across corpora, scales sketch columns from training data only,
and fits a zero-origin correction basis so a query can emit an exact zero delta.
To measure transfer rather than in-corpus sharing, run
`python scripts/run_leave_one_corpus_out.py --models text-embedding-3-small-aiberm`.
Each held-out corpus is excluded from anchor fitting, predictor training, and
lambda selection.

Corpus-specific rank Oracle
---------------------------

`scripts/run_corpus_rank_oracle.py` measures the upper bound available to a
corpus-aware correction in the fixed global `B128` coordinate system. It
constructs the same-query variants `D0_original`, `D1_random_external`,
`D2_hard_external`, and `D3_high_hard_external`, then optimizes
`Normalize(q + mu + B @ a)` against each variant's hard-negative ranking loss.
It also optimizes one shared `a_q` across all four variants. The output records
`corpus_oracle`, `q_only_oracle`, and `H_corpus = corpus_oracle - q_only_oracle`
for every retrieval metric.

Run the full study with:
`python scripts/run_corpus_rank_oracle.py --models qwen3 bge-m3 e5`

The long-form metrics and pivoted summary are written to
`results/corpus_rank_oracle/`.

ArguAna is supported as a local BEIR dataset under `datasets/arguana`. The official release contains five qrels whose relevant document is absent from the official corpus. ArguAna configs explicitly use `missing_relevant_policy: keep`: those queries remain in the 1,406-query evaluation denominator, count as retrieval misses, expose missing-positive counts in per-query output, and have unavailable positive geometry/oracle fields.

NFCorpus and SciFact are supported as local BEIR datasets under `datasets/nfcorpus` and `datasets/scifact`. Ready-to-run test-split configs are provided for Qwen3, BGE-M3, and E5 Base v2.

The embedding stack is split into model adapters and transport providers. The adapter owns model-specific query/document text rules and normalization; the provider owns authentication, batching transport, retries, and response parsing. Aiberm uses the official OpenAI-compatible SDK client with `https://aiberm.com/v1`, while the existing SiliconFlow and OpenRouter providers remain independent.

Run `text-embedding-3-small` through Aiberm on the available test datasets:

```powershell
cd "F:\Dense Retrieval"
.\.venv\Scripts\Activate.ps1

python scripts/run_experiment.py --config configs/fiqa_text_embedding_3_small_aiberm.yaml
python scripts/run_experiment.py --config configs/arguana_text_embedding_3_small_aiberm.yaml
python scripts/run_experiment.py --config configs/nfcorpus_text_embedding_3_small_aiberm.yaml
python scripts/run_experiment.py --config configs/scifact_text_embedding_3_small_aiberm.yaml

python scripts/analyze_results.py --results results --output results/summary.json
python scripts/plot_results.py --results results
```

Set the project-root `.env` value before running:

```dotenv
AIBERM_API_KEY=your_aiberm_api_key_here
```
