"""
API Routes — Q&A streaming, cancellation, document listing, image retrieval.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from core.memory_store import MemoryStore
from services.qa_service import QAService

logger = logging.getLogger(__name__)
router = APIRouter()

# session_id -> asyncio.Event; set to cancel the stream
_sessions: dict[str, asyncio.Event] = {}

ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}


# ── Knowledge base ──────────────────────────────────────────────────────────

@router.get("/documents")
async def list_documents():
    return {"documents": MemoryStore.get_instance().list_documents()}


@router.get("/stats")
async def stats():
    s = MemoryStore.get_instance()
    return {
        "total_documents": s.total_documents(),
        "total_chunks": s.total_chunks(),
        "model": "qwen3-vl",
    }


# ── Q&A ─────────────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """
    Stream an AI answer. Automatically includes relevant document images.
    Supports cancellation via POST /cancel/{session_id}.
    """
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")

    cancel_event = asyncio.Event()
    _sessions[session_id] = cancel_event

    # Parse attached file
    user_image: bytes | None = None
    user_image_ext = "png"
    user_text: str | None = None

    if file and file.filename:
        content_type = (file.content_type or "").lower()
        raw = await file.read()
        if content_type in ALLOWED_IMAGE_MIME:
            user_image = raw
            user_image_ext = content_type.split("/")[-1].replace("jpeg", "jpg")
        else:
            try:
                user_text = raw.decode("utf-8", errors="ignore")[:4000]
            except Exception:
                pass

    svc = QAService()

    async def generate():
        try:
            async for token in svc.stream_answer(
                question=question,
                user_image=user_image,
                user_image_ext=user_image_ext,
                user_text_attachment=user_text,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    yield "\n\n_[Response cancelled by user]_"
                    return
                yield token
        except Exception as e:
            logger.error(f"Stream error [{session_id}]: {e}")
            yield f"\n\n⚠️ Error: {e}"
        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/cancel/{session_id}")
async def cancel(session_id: str):
    """Cancel an active streaming response."""
    ev = _sessions.get(session_id)
    if ev and not ev.is_set():
        ev.set()
        return {"cancelled": True, "session_id": session_id}
    return {"cancelled": False, "reason": "Session not found or already complete"}


# ── Source preview ──────────────────────────────────────────────────────────

@router.get("/sources")
async def get_sources(q: str):
    """Return source docs + relevant image URLs for a given query."""
    if not q.strip():
        return {"sources": [], "images": []}
    svc = QAService()
    return svc.get_sources_and_images(q)


@router.get("/document/{doc_id}/images")
async def document_images(doc_id: str):
    """Get all image URLs for a specific document."""
    store = MemoryStore.get_instance()
    img_chunks = store.get_image_chunks_for_doc(doc_id)
    return {
        "doc_id": doc_id,
        "images": [
            {
                "chunk_id": c.id,
                "url": c.image_url,
                "page": c.metadata.get("page", 0),
                "summary": c.content[:150],
            }
            for c in img_chunks
            if c.image_url
        ],
    }
