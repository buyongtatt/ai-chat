"""
MemoryStore — in-memory document chunk store loaded at API startup.
All retrieval happens against this RAM store for maximum speed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    content: str            # AI-generated dense summary
    raw_preview: str        # Original text snippet (first 400 chars)
    chunk_index: int
    source_type: str        # "text" | "image"
    image_path: str | None = None       # absolute path on disk
    image_url: str | None = None        # served URL for frontend
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocMeta:
    doc_id: str
    name: str
    file_type: str
    total_chunks: int
    image_chunks: int


class MemoryStore:
    _instance: MemoryStore | None = None

    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._doc_chunks: dict[str, list[str]] = {}   # doc_id -> chunk_ids
        self._doc_meta: dict[str, DocMeta] = {}

    @classmethod
    def get_instance(cls) -> MemoryStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_chunk(self, chunk: Chunk):
        self._chunks[chunk.id] = chunk
        self._doc_chunks.setdefault(chunk.doc_id, []).append(chunk.id)

    def set_doc_meta(self, meta: DocMeta):
        self._doc_meta[meta.doc_id] = meta

    def total_chunks(self) -> int:
        return len(self._chunks)

    def total_documents(self) -> int:
        return len(self._doc_chunks)

    def list_documents(self) -> list[dict]:
        result = []
        for doc_id, chunk_ids in self._doc_chunks.items():
            meta = self._doc_meta.get(doc_id)
            image_count = sum(
                1 for cid in chunk_ids
                if self._chunks.get(cid) and self._chunks[cid].source_type == "image"
            )
            result.append({
                "id": doc_id,
                "name": meta.name if meta else "Unknown",
                "type": meta.file_type if meta else "unknown",
                "chunks": len(chunk_ids),
                "image_chunks": image_count,
            })
        return sorted(result, key=lambda x: x["name"].lower())

    def search(self, query: str, top_k: int = 8) -> list[Chunk]:
        """TF-IDF keyword search over all chunks in memory."""
        query_terms = _tokenize(query)
        if not query_terms:
            return list(self._chunks.values())[:top_k]

        total = max(self.total_chunks(), 1)
        all_chunks = list(self._chunks.values())

        # Pre-compute IDF
        idf: dict[str, float] = {}
        for term in set(query_terms):
            df = sum(1 for c in all_chunks if term in _tokenize(c.content + " " + c.raw_preview))
            idf[term] = math.log((total + 1) / (df + 1)) + 1.0

        scored: list[tuple[float, Chunk]] = []
        for chunk in all_chunks:
            doc_terms = _tokenize(chunk.content + " " + chunk.raw_preview)
            doc_len = max(len(doc_terms), 1)
            score = sum(
                (doc_terms.count(t) / doc_len) * idf.get(t, 1.0)
                for t in query_terms
            )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def get_image_chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        """Get all image chunks belonging to a specific document."""
        chunk_ids = self._doc_chunks.get(doc_id, [])
        return [
            self._chunks[cid]
            for cid in chunk_ids
            if cid in self._chunks and self._chunks[cid].source_type == "image"
        ]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())
