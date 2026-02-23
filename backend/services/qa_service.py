"""
QAService — retrieval + streaming answer generation.
Key feature: automatically attaches relevant document images to the Ollama vision call.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from core.memory_store import Chunk, MemoryStore
from services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are DocMind, an intelligent assistant with access to a curated document library.

RULES:
- Base answers ONLY on the provided knowledge base context
- Cite sources using [Source: filename, page X] notation inline
- If attached images are provided, describe and analyze them as part of your answer
- Use markdown formatting: **bold** for key terms, bullet lists, ```code blocks```
- If the context is insufficient, say so honestly — never invent information
- Be precise, thorough, and structured
"""

# Max number of document images to attach per answer call
MAX_VISION_IMAGES = 3


class QAService:
    def __init__(self):
        self.store = MemoryStore.get_instance()
        self.client = OllamaClient.local()

    async def stream_answer(
        self,
        question: str,
        user_image: bytes | None = None,
        user_image_ext: str = "png",
        user_text_attachment: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:

        # 1. Retrieve relevant chunks (text + image)
        relevant_chunks = self.store.search(question, top_k=10)

        # 2. Build context string
        context = _build_context(relevant_chunks)

        # 3. Collect image paths from relevant image-type chunks to send to vision model
        doc_image_paths = [
            c.image_path for c in relevant_chunks
            if c.source_type == "image" and c.image_path and Path(c.image_path).exists()
        ][:MAX_VISION_IMAGES]

        # 4. Optionally inject attached text file content
        attachment_note = ""
        if user_text_attachment:
            attachment_note = (
                f"\n\n[User attached a file with the following content]\n"
                f"{user_text_attachment[:3000]}"
            )

        # 5. Build full prompt
        prompt = (
            f"=== KNOWLEDGE BASE CONTEXT ===\n{context}"
            f"{attachment_note}\n\n"
            f"=== USER QUESTION ===\n{question}"
        )

        # 6. Save user's attached image to temp file
        all_image_paths = list(doc_image_paths)
        temp_path: str | None = None
        if user_image:
            tmp = tempfile.NamedTemporaryFile(
                suffix=f".{user_image_ext}", delete=False
            )
            tmp.write(user_image)
            tmp.close()
            temp_path = tmp.name
            # User's image goes first so the model sees it prominently
            all_image_paths.insert(0, temp_path)

        try:
            async for token in self.client.chat_stream(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                image_paths=all_image_paths if all_image_paths else None,
                cancel_event=cancel_event,
            ):
                yield token
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def get_sources_and_images(self, question: str) -> dict:
        """Return relevant sources + image URLs for the frontend to preview."""
        chunks = self.store.search(question, top_k=8)

        seen_docs: set[str] = set()
        sources = []
        image_urls = []

        for c in chunks:
            if c.doc_name not in seen_docs:
                seen_docs.add(c.doc_name)
                sources.append({
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "source_type": c.source_type,
                })
            if c.source_type == "image" and c.image_url:
                image_urls.append({
                    "url": c.image_url,
                    "doc_name": c.doc_name,
                    "page": c.metadata.get("page", 0),
                    "summary": c.content[:120] + "..." if len(c.content) > 120 else c.content,
                })

        return {"sources": sources, "images": image_urls[:6]}


def _build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "No relevant content found in knowledge base."

    parts = []
    for i, c in enumerate(chunks, 1):
        label = f"[Source: {c.doc_name}"
        if c.source_type == "image":
            label += " | image"
        if "page" in c.metadata:
            label += f" | page {c.metadata['page'] + 1}"
        label += "]"
        parts.append(f"--- Context {i} {label} ---\n{c.content}")

    return "\n\n".join(parts)