"""BEIR-format adapters with local path and BEIR download support."""
import json
from pathlib import Path
from typing import Dict

from .base import DatasetSample, Document, RetrievalDataset

BEIR_URLS = {
    name: f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    for name in ("arguana", "fiqa", "scifact", "nfcorpus")
}

def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _resolve_root(name: str, root: str | Path, download: bool) -> Path:
    root = Path(root)
    candidates = [root / name / name, root / name]
    for candidate in candidates:
        if (candidate / "corpus.jsonl").exists():
            return candidate
    if not download:
        raise FileNotFoundError(f"No BEIR dataset at {candidates[0]} or {candidates[1]}")
    if name not in BEIR_URLS:
        raise ValueError(
            f"No download URL registered for {name}; provide a local BEIR-format dataset"
        )
    try:
        from beir import util
        out = Path(util.download_and_unzip(BEIR_URLS[name], str(root)))
        return out / name if (out / name / "corpus.jsonl").exists() else out
    except ImportError as e:
        raise RuntimeError("Install beir or provide a local BEIR-format dataset") from e

def load_beir_dataset(
    name: str,
    root: str | Path = "datasets",
    split: str = "test",
    download: bool = False,
    missing_relevant_policy: str = "error",
) -> RetrievalDataset:
    name = name.lower()
    base = _resolve_root(name, root, download)
    corpus = {}
    for row in _read_jsonl(base / "corpus.jsonl"):
        doc_id = str(row.get("_id", row.get("id")))
        corpus[doc_id] = Document(doc_id, str(row.get("title", "") or ""), str(row.get("text", "") or ""))
    queries = {str(row.get("_id", row.get("id"))): str(row.get("text", row.get("query", ""))) for row in _read_jsonl(base / "queries.jsonl")}
    qrel_path = base / "qrels" / f"{split}.tsv"
    if not qrel_path.exists():
        raise FileNotFoundError(f"Missing qrels split: {qrel_path}")
    qrels: Dict[str, Dict[str, int]] = {}
    with qrel_path.open(encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[:3]
                qrels.setdefault(str(qid), {})[str(did)] = int(float(score))
    samples = [DatasetSample(qid, queries[qid], [did for did, score in rels.items() if score > 0]) for qid, rels in qrels.items() if qid in queries]
    dataset = RetrievalDataset(name=name, queries=samples, corpus=corpus, qrels=qrels, split=split)
    if missing_relevant_policy not in {"error", "keep"}:
        raise ValueError("missing_relevant_policy must be 'error' or 'keep'")
    dataset.validate(allow_missing_relevant_documents=missing_relevant_policy == "keep")
    return dataset
