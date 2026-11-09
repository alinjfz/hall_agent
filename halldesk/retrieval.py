"""Search small sets of prose documents using BM25."""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import log1p
from pathlib import Path

HANDBOOK = Path(__file__).with_name("handbook")


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    text: str


def tokenize(text):
    return re.findall(r"[^\W_]+", text.casefold())


class BM25:
    def __init__(self, documents):
        self.documents = {key: Counter(tokenize(text)) for key, text in documents.items()}
        self.lengths = {key: terms.total() for key, terms in self.documents.items()}
        self.average = max(sum(self.lengths.values()) / max(len(self.lengths), 1), 1)
        self.frequencies = Counter(term for doc in self.documents.values() for term in doc)

    def search(self, query, limit=3):
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
            if score > 0:
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


def search_handbook(query, limit=3):
    sections, index = handbook_index()
    return [sections[key] for key, _ in index.search(query, limit)]
