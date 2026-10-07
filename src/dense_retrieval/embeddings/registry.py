"""Embedding models benchmarked by the query-correction studies.

``cache_dir`` is the directory name used under the embedding cache,
``result_dir`` the directory name used under ``results/<dataset>/``, and
``config`` the config file used to rebuild the hosted model when a cached query
encoding is missing.  Scripts that need to iterate over the studied models use
these keys and labels instead of repeating the mapping.
"""
from __future__ import annotations

MODEL_SPECS = {
    "text-embedding-3-small-aiberm": {
        "label": "text-embedding-3-small (Aiberm)",
        "cache_dir": "openai__text-embedding-3-small",
        "result_dir": "text-embedding-3-small",
        "config": "fiqa_text_embedding_3_small_aiberm.yaml",
    },
    "qwen3": {
        "label": "Qwen3",
        "cache_dir": "Qwen__Qwen3-Embedding-0.6B",
        "result_dir": "qwen3-embedding-0.6b",
        "config": "fiqa_qwen3_siliconflow.yaml",
    },
    "bge-m3": {
        "label": "BGE-M3",
        "cache_dir": "BAAI__bge-m3",
        "result_dir": "bge-m3",
        "config": "fiqa_bge_m3_siliconflow.yaml",
    },
    "e5": {
        "label": "E5",
        "cache_dir": "intfloat__e5-base-v2",
        "result_dir": "e5-base-v2",
        "config": "fiqa_e5_base_v2_openrouter.yaml",
    },
}
