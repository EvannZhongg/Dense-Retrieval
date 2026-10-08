import json
from pathlib import Path

from dense_retrieval.datasets import load_beir_dataset


def test_beir_loader_preserves_ids_and_text(tmp_path: Path):
    base = tmp_path / "toy"
    (base / "qrels").mkdir(parents=True)
    (base / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "title": "T", "text": "body"}) + "\n", encoding="utf-8")
    (base / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "question"}) + "\n", encoding="utf-8")
    (base / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    dataset = load_beir_dataset("toy", tmp_path)
    assert dataset.split == "test"
    assert dataset.query_texts == ["question"]
    assert dataset.corpus_documents[0].text == "body"
    assert dataset.qrels == {"q1": {"d1": 1}}
