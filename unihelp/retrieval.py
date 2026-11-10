"""Hybrid handbook search: BM25 plus optional embeddings, fused by rank."""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import log1p
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import ConfigDict


HANDBOOK = Path(__file__).with_name("handbook")
# Chosen with evals/retrieval_sweep.py: the highest cut-off that loses no recall@3.
MIN_SCORE = 1.0
# Provisional cosine floor for the embedding half. It has not been swept against a
# live embedding model yet; set it with `python -m evals.retrieval_sweep --hybrid`.
MIN_SIMILARITY = 0.3
RANK_CONSTANT = 60
STOP = {"a", "an", "and", "can", "has", "have", "i", "in", "is", "it", "keeps", "my", "of",
        "on", "please", "the", "this", "to", "was", "will", "with"}
SYNONYMS = {"faucet": "tap", "dripping": "drip", "drips": "drip", "bulb": "light",
            "broken": "failed", "stopped": "failed", "broke": "failed", "fix": "repair",
            "bedroom": "room", "radiators": "radiator", "leaking": "leak"}


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    text: str


def tokenize(text):
    terms = []
    for term in re.findall(r"[^\W_]+", text.casefold()):
        if term in STOP:
            continue
        singular = term[:-1] if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")) else term
        terms.append(SYNONYMS.get(term, SYNONYMS.get(singular, singular)))
    return terms


class BM25:
    def __init__(self, documents):
        self.documents = {key: Counter(tokenize(text)) for key, text in documents.items()}
        self.lengths = {key: terms.total() for key, terms in self.documents.items()}
        self.average = max(sum(self.lengths.values()) / max(len(self.lengths), 1), 1)
        self.frequencies = Counter(term for doc in self.documents.values() for term in doc)

    def search(self, query, limit=3, *, min_score=0):
        if limit < 1:
            return []
        terms = set(tokenize(query))
        scores = []
        for key, document in self.documents.items():
            norm = 1.2 * (0.25 + 0.75 * self.lengths[key] / self.average)
            score = sum(
                log1p((len(self.documents) - self.frequencies[term] + 0.5)
                      / (self.frequencies[term] + 0.5))
                * 2.2 * document[term] / (document[term] + norm)
                for term in sorted(terms & document.keys())
            )
            if score > min_score:
                scores.append((key, score))
        return sorted(scores, key=lambda row: (-row[1], row[0]))[:limit]


@lru_cache(maxsize=1)
def handbook_index():
    sections = {}
    for path in sorted(HANDBOOK.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        pieces = re.split(r"^## ([a-z0-9-]+): (.+)$", content, flags=re.MULTILINE)[1:]
        for section_id, title, body in zip(pieces[::3], pieces[1::3], pieces[2::3]):
            if section_id in sections:
                raise ValueError(f"Duplicate handbook section: {section_id}")
            sections[section_id] = Section(section_id, title, body.strip())
    return sections, BM25({key: f"{row.title} {row.text}" for key, row in sections.items()})


class HandbookRetriever(BaseRetriever):
    """Rank handbook sections by reciprocal rank fusion of BM25 and embedding matches.

    Without an embedding model the retriever is lexical only. An empty result means
    the handbook has nothing relevant; callers must not treat it as an answer.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sections: dict[str, Section]
    lexical: BM25
    dense: Any = None
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        rankings = {"lexical": [key for key, _ in self.lexical.search(query, self.k, min_score=MIN_SCORE)]}
        if self.dense is not None:
            try:
                rankings["dense"] = [document.id for document, score
                                     in self.dense.similarity_search_with_score(query, k=self.k)
                                     if score >= MIN_SIMILARITY]
            except Exception as exc:  # A search outage must not block triage.
                logging.warning("Handbook embedding search unavailable (%s)", type(exc).__name__)
        fused, matched = Counter(), {}
        for name, ranking in rankings.items():
            for rank, key in enumerate(ranking):
                fused[key] += 1 / (RANK_CONSTANT + rank + 1)
                matched.setdefault(key, []).append(name)
        ordered = sorted(fused, key=lambda key: (-fused[key], key))[:self.k]
        return [Document(id=key, page_content=self.sections[key].text,
                         metadata={"id": key, "title": self.sections[key].title, "matched": matched[key]})
                for key in ordered]


_dense_stores = {}


def dense_store(embeddings, model_name):
    """Embed the handbook once per embedding model. A failure leaves search lexical."""
    if embeddings is None:
        return None
    if model_name not in _dense_stores:
        sections, _ = handbook_index()
        try:
            store = InMemoryVectorStore(embeddings)
            store.add_documents([Document(id=key, page_content=f"{row.title}\n{row.text}")
                                 for key, row in sections.items()], ids=list(sections))
        except Exception as exc:
            logging.warning("Handbook embeddings unavailable (%s)", type(exc).__name__)
            return None
        _dense_stores[model_name] = store
    return _dense_stores[model_name]


def handbook_retriever(limit=3, *, embeddings=None, model_name=None):
    if embeddings is None:
        from unihelp.models import embedding_model

        embeddings, model_name = embedding_model()
    sections, lexical = handbook_index()
    dense = dense_store(embeddings, model_name or type(embeddings).__name__)
    return HandbookRetriever(name="search_handbook", sections=sections, lexical=lexical, dense=dense, k=limit,
                             metadata={"retrieval_mode": "hybrid" if dense is not None else "lexical"})


def search_handbook(query, limit=3):
    """Return up to `limit` relevant sections, or an empty list if none is relevant."""
    if limit < 1 or not query.strip():
        return []
    return handbook_retriever(limit).invoke(query)
