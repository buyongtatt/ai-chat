"""
QAService — retrieval + streaming answer generation with per-step timing logs.

After streaming the answer, emits a special metadata footer:
  @@METADATA@@{...json...}@@END@@
containing only the images the model actually cited ([Image N]) in its answer.
The frontend strips this line and uses it to display the referenced images.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import urlparse, urlunparse

from core.memory_store import Chunk, MemoryStore
from core.timer import Timer
from services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

MAX_DOC_IMAGES    = 5   # max images sent to vision model
METADATA_PREFIX   = "@@METADATA@@"
METADATA_SUFFIX   = "@@END@@"

SYSTEM_PROMPT = """\
You are DocMind, a helpful assistant that answers questions from a document library.
Answer directly. Do not narrate your steps, explain your process, or describe what you are about to do.
Use only the provided context. Cite document sources inline as [Source: filename].
When your answer refers to a provided image, cite it inline as [Image 1], [Image 2], etc.
If the context does not contain the answer, say so briefly.
Use markdown only when it genuinely helps readability.
"""


@dataclass
class ImageRef:
    index: int
    path: str
    label: str
    doc_name: str
    page: int
    image_url: str
    is_user: bool = False


# ── IP / URL helpers ────────────────────────────────────────────────────────

def _get_server_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _is_local_request(client_host: str | None) -> bool:
    if not client_host:
        return True
    return client_host in ("127.0.0.1", "::1", "localhost")


def _rewrite_image_url(url: str, client_host: str | None) -> str:
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return url
    if _is_local_request(client_host):
        return url
    u = urlparse(url)
    if (u.hostname or "").lower() in ("localhost", "127.0.0.1"):
        lan_ip = _get_server_lan_ip()
        netloc = f"{lan_ip}:{u.port}" if u.port else lan_ip
        return urlunparse((u.scheme, netloc, u.path, u.params, u.query, u.fragment))
    return url


def _parse_cited_indices(answer: str) -> set[int]:
    """Extract all [Image N] references from the model's answer."""
    return {int(m) for m in re.findall(r'\[Image\s+(\d+)\]', answer, re.IGNORECASE)}


# ── QA Service ──────────────────────────────────────────────────────────────

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
        client_host: str | None = None,
    ) -> AsyncGenerator[str, None]:

        request_start = time.perf_counter()
        logger.info("─── New Q&A request ───────────────────────────────")
        logger.info(f"  Question: {question[:120]}{'...' if len(question) > 120 else ''}")

        # 1. Retrieve relevant chunks — separate searches so images aren't crowded out
        with Timer("chunk retrieval"):
            text_chunks  = [c for c in self.store.search(question, top_k=20)
                            if c.source_type == "text"][:15]
            image_chunks = self.store.search_images(question, top_k=MAX_DOC_IMAGES)
            relevant_chunks = text_chunks + image_chunks
            logger.info(
                f"       → {len(text_chunks)} text + {len(image_chunks)} image chunks "
                f"from {len({c.doc_name for c in relevant_chunks})} doc(s)"
            )

        # 2. Build text context
        with Timer("context assembly"):
            context = _build_context(relevant_chunks)
            logger.info(f"       → context length: {len(context)} chars")

        # 3. Build image reference list
        with Timer("image path resolution"):
            image_refs: list[ImageRef] = []
            temp_path: str | None = None

            if user_image:
                tmp = tempfile.NamedTemporaryFile(suffix=f".{user_image_ext}", delete=False)
                tmp.write(user_image)
                tmp.close()
                temp_path = tmp.name
                image_refs.append(ImageRef(
                    index=1, path=temp_path,
                    label="User attached image",
                    doc_name="user", page=0,
                    image_url="",     # user image has no served URL
                    is_user=True,
                ))

            for c in image_chunks:
                if not c.image_path or not Path(c.image_path).exists():
                    continue
                page = c.metadata.get("page", 0)
                url  = _rewrite_image_url(c.image_url or "", client_host)
                image_refs.append(ImageRef(
                    index=len(image_refs) + 1,
                    path=c.image_path,
                    label=f"{c.doc_name} — page {page + 1}",
                    doc_name=c.doc_name,
                    page=page,
                    image_url=url,
                ))

            logger.info(
                f"       → {len(image_refs)} image(s) prepared: "
                + ", ".join(f"[Image {r.index}] {r.label}" for r in image_refs)
            )

        # 4. Image manifest for prompt
        image_note = ""
        if image_refs:
            lines = ["", "=== ATTACHED IMAGES ==="]
            for r in image_refs:
                lines.append(f"[Image {r.index}] {r.label}")
            lines.append(
                "When referring to any image in your answer, "
                "cite it inline as [Image 1], [Image 2], etc."
            )
            image_note = "\n".join(lines)

        # 5. Text attachment
        attachment_note = ""
        if user_text_attachment:
            attachment_note = f"\n\n[User attached a file]\n{user_text_attachment[:3000]}"

        # 6. Build prompt
        with Timer("prompt assembly"):
            prompt = (
                f"=== KNOWLEDGE BASE CONTEXT ===\n{context}"
                f"{image_note}"
                f"{attachment_note}\n\n"
                f"=== USER QUESTION ===\n{question}"
            )
            logger.info(f"       → total prompt length: {len(prompt)} chars")

        # 7. Stream — accumulate full answer to parse [Image N] citations
        logger.info("  ⏱  [ollama stream] sending to model...")
        stream_start     = time.perf_counter()
        first_token_time = None
        token_count      = 0
        full_answer      = ""

        try:
            async for token in self.client.chat_stream(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                image_paths=[r.path for r in image_refs] if image_refs else None,
                cancel_event=cancel_event,
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    logger.info(
                        f"  ✅ [ollama stream] first token in "
                        f"{first_token_time - stream_start:.3f}s"
                    )
                token_count  += 1
                full_answer  += token
                yield token

        finally:
            stream_elapsed = time.perf_counter() - stream_start
            total_elapsed  = time.perf_counter() - request_start
            tps = token_count / stream_elapsed if stream_elapsed > 0 else 0
            logger.info(
                f"  ✅ [ollama stream] finished — "
                f"{token_count} tokens in {stream_elapsed:.2f}s ({tps:.1f} tok/s)"
            )

            # 8. Parse which [Image N] the model cited and build metadata footer
            cited_indices = _parse_cited_indices(full_answer)
            logger.info(f"  ⏱  [image citation] model cited image indices: {cited_indices or 'none'}")

            cited_images = []
            for r in image_refs:
                if r.index in cited_indices and not r.is_user:
                    cited_images.append({
                        "image_label": f"Image {r.index}",
                        "url":         r.image_url,
                        "doc_name":    r.doc_name,
                        "page":        r.page,
                        "label":       r.label,
                    })

            # Build source list from text chunks
            seen: set[str] = set()
            sources = []
            for c in text_chunks:
                if c.doc_name not in seen:
                    seen.add(c.doc_name)
                    sources.append({"doc_id": c.doc_id, "doc_name": c.doc_name})

            metadata = {
                "sources":       sources,
                "cited_images":  cited_images,   # only images actually cited
                "total_images_sent": len(image_refs),
            }

            logger.info(
                f"  ✅ [metadata] {len(cited_images)} cited image(s) of "
                f"{len(image_refs)} sent, {len(sources)} source doc(s)"
            )
            logger.info(f"  🏁 total request time: {total_elapsed:.3f}s")
            logger.info("────────────────────────────────────────────────────")

            # Emit metadata footer — frontend strips and parses this
            yield f"\n{METADATA_PREFIX}{json.dumps(metadata)}{METADATA_SUFFIX}"

            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def get_sources_and_images(self, question: str, client_host: str | None = None) -> dict:
        """Fallback endpoint — returns relevant sources without citation filtering."""
        with Timer("sources & images lookup"):
            text_chunks  = self.store.search(question, top_k=10)
            image_chunks = self.store.search_images(question, top_k=6)

        seen: set[str] = set()
        sources = []
        for c in text_chunks:
            if c.doc_name not in seen:
                seen.add(c.doc_name)
                sources.append({"doc_id": c.doc_id, "doc_name": c.doc_name, "source_type": c.source_type})

        images = []
        for idx, c in enumerate(image_chunks, 1):
            if c.image_url:
                images.append({
                    "url":         _rewrite_image_url(c.image_url, client_host),
                    "doc_name":    c.doc_name,
                    "page":        c.metadata.get("page", 0),
                    "summary":     c.content[:120] + "..." if len(c.content) > 120 else c.content,
                    "image_label": f"Image {idx}",
                })

        return {"sources": sources, "images": images}


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