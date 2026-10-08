from __future__ import annotations

from pathlib import Path

import yaml


def load_model_config(name: str, path: str | Path = "configs/models.yaml") -> dict:
    """Load one model definition from the repository's single model registry."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = payload.get("models", payload)
    if name not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown model {name!r}; available models: {available}")
    config = dict(models[name])
    config.setdefault("model_id", name)
    return config
