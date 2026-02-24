"""
QAService — retrieval + streaming answer generation.

Image strategy (multi-image model e.g. qwen3-vl, gemma3, llava):
  - User attached image + relevant doc images → all sent together (max MAX_VISION_IMAGES)
  - User attached image only                  → sent alone
  - No user image                             → top relevant doc images sent
  - All image summaries always included as text context regardless
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

# Max doc images to pass to the vision model alongside user's question
# Increase if your model supports more (qwen3-vl supports many)
MAX_DOC_IMAGES = 3

SYSTEM_PROMPT = """\
You are DocMind, a helpful assistant that answers questions from a document library.
Answer directly. Do not narrate your steps, explain your process, or describe what you are about to do.
Use only the provided context. Cite sources inline as [Source: filename].
If the context does not contain the answer, say so briefly.
Use markdown only when it genuinely helps readability.
"""


class QAService:
    def __init__(self):
        self.store  = MemoryStore.get_instance()
        self.client = OllamaClient.local()

    async def stream_answer(
        self,
        question: str,
        user_image: bytes | None = None,
        user_image_ext: str = "png",
        user_text_attachment: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:

        # 1. Retrieve relevant chunks
        relevant_chunks = self.store.search(question, top_k=10)

        # 2. Build text context (image chunk summaries included as text too)
        context = _build_context(relevant_chunks)

        # 3. Collect all relevant doc image paths (multi-image model supports many)
        doc_image_paths = [
            c.image_path for c in relevant_chunks
            if c.source_type == "image"
            and c.image_path
            and Path(c.image_path).exists()
        ][:MAX_DOC_IMAGES]

        # 4. Handle user-attached image — save to temp file
        all_image_paths: list[str] = []
        temp_path: str | None = None
        image_note = ""

        if user_image:
            tmp = tempfile.NamedTemporaryFile(
                suffix=f".{user_image_ext}", delete=False
            )
            tmp.write(user_image)
            tmp.close()
            temp_path = tmp.name
            # User image goes first, then relevant doc images
            all_image_paths = [temp_path] + doc_image_paths
            image_note = "\n\n[User has attached an image for analysis]"
            logger.info(
                f"Vision call: 1 user image + {len(doc_image_paths)} doc image(s) "
                f"= {len(all_image_paths)} total"
            )
        else:
            all_image_paths = doc_image_paths
            if doc_image_paths:
                image_note = f"\n\n[Referencing {len(doc_image_paths)} relevant document image(s)]"
                logger.info(f"Vision call: {len(doc_image_paths)} doc image(s)")

        # 5. Inject attached text file content
        attachment_note = ""
        if user_text_attachment:
            attachment_note = (
                f"\n\n[User attached a file]\n{user_text_attachment[:3000]}"
            )

        # 6. Build full prompt
        prompt = (
            f"=== KNOWLEDGE BASE CONTEXT ===\n{context}"
            f"{image_note}"
            f"{attachment_note}\n\n"
            f"=== USER QUESTION ===\n{question}"
        )

        # 7. Stream answer with all images
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
        sources    = []
        image_urls = []

        for c in chunks:
            if c.doc_name not in seen_docs:
                seen_docs.add(c.doc_name)
                sources.append({
                    "doc_id":      c.doc_id,
                    "doc_name":    c.doc_name,
                    "source_type": c.source_type,
                })
            if c.source_type == "image" and c.image_url:
                image_urls.append({
                    "url":      c.image_url,
                    "doc_name": c.doc_name,
                    "page":     c.metadata.get("page", 0),
                    "summary":  c.content[:120] + "..." if len(c.content) > 120 else c.content,
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