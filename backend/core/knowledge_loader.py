"""
KnowledgeLoader — reads processed_cache/*.json into MemoryStore at startup.
Run the indexer first: cd backend && python -m services.indexer
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.memory_store import Chunk, DocMeta, MemoryStore

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "processed_cache"
BASE_IMAGE_URL = "http://localhost:8000/images"


class KnowledgeLoader:
    def __init__(self):
        CACHE_DIR.mkdir(exist_ok=True)

    async def load_all_into_memory(self, store: MemoryStore):
        cache_files = sorted(CACHE_DIR.glob("*.json"))

        if not cache_files:
            logger.warning(
                "⚠️  No cache files found. "
                "Run: cd backend && python -m services.indexer"
            )
            return

        for cache_file in cache_files:
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                doc_id = data["doc_id"]
                doc_name = data["doc_name"]
                chunks_data = data.get("chunks", [])

                image_chunks = sum(1 for c in chunks_data if c.get("source_type") == "image")
                store.set_doc_meta(DocMeta(
                    doc_id=doc_id,
                    name=doc_name,
                    file_type=data.get("doc_type", "unknown"),
                    total_chunks=len(chunks_data),
                    image_chunks=image_chunks,
                ))

                for raw in chunks_data:
                    # Build a served URL for image chunks
                    image_url = None
                    if raw.get("image_path"):
                        img_filename = Path(raw["image_path"]).name
                        image_url = f"{BASE_IMAGE_URL}/{img_filename}"

                    chunk = Chunk(
                        id=raw["id"],
                        doc_id=doc_id,
                        doc_name=doc_name,
                        content=raw["content"],
                        raw_preview=raw.get("raw_preview", ""),
                        chunk_index=raw["chunk_index"],
                        source_type=raw.get("source_type", "text"),
                        image_path=raw.get("image_path"),
                        image_url=image_url,
                        metadata=raw.get("metadata", {}),
                    )
                    store.add_chunk(chunk)

                logger.info(f"  📄 {doc_name} — {len(chunks_data)} chunks ({image_chunks} images)")
            except Exception as e:
                logger.error(f"  ❌ Failed: {cache_file.name} — {e}")
