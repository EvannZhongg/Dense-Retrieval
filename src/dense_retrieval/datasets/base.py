from dataclasses import dataclass
from typing import Dict, Iterable, List

@dataclass(frozen=True)
class DatasetSample:
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]

@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

@dataclass
class RetrievalDataset:
    name: str
    queries: List[DatasetSample]
    corpus: Dict[str, Document]
    qrels: Dict[str, Dict[str, int]]

    @property
    def query_texts(self) -> List[str]:
        return [q.query_text for q in self.queries]

    @property
    def corpus_documents(self) -> List[Document]:
        return list(self.corpus.values())

    def validate(self, allow_missing_relevant_documents: bool = False) -> None:
        query_ids = {q.query_id for q in self.queries}
        missing_q = set(self.qrels) - query_ids
        if missing_q:
            raise ValueError(f"qrels contain unknown query ids: {sorted(missing_q)[:5]}")
        missing_d = {d for rels in self.qrels.values() for d, score in rels.items() if score > 0 and d not in self.corpus}
        if missing_d and not allow_missing_relevant_documents:
            raise ValueError(f"qrels contain documents absent from corpus: {sorted(missing_d)[:5]}")
        if len(query_ids) != len(self.queries):
            raise ValueError("duplicate query ids")
