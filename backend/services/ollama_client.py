"""
OllamaClient — unified async client for both:
  • Cloud Ollama  (https://api.ollama.com)  → used by indexer
  • Local Ollama  (http://localhost:11434)  → used for answering questions

Both endpoints speak the same /api/chat protocol, so we reuse the same
streaming logic. The only difference is the base URL and optional API key.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

TIMEOUT = 300  # seconds — generous for large vision payloads


class OllamaClient:
    """
    Single client class that works for both cloud and local Ollama.

    Cloud usage (indexing):
        client = OllamaClient.cloud()

    Local usage (answering):
        client = OllamaClient.local()
    """

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.api_key  = api_key

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def cloud(cls) -> "OllamaClient":
        """Cloud Ollama instance — for indexing."""
        from core.config import OLLAMA_CLOUD_URL, OLLAMA_CLOUD_MODEL, OLLAMA_CLOUD_API_KEY
        return cls(
            base_url=OLLAMA_CLOUD_URL,
            model=OLLAMA_CLOUD_MODEL,
            api_key=OLLAMA_CLOUD_API_KEY,
        )

    @classmethod
    def local(cls) -> "OllamaClient":
        """Local Ollama instance — for answering questions."""
        from core.config import OLLAMA_LOCAL_URL, OLLAMA_LOCAL_MODEL
        return cls(
            base_url=OLLAMA_LOCAL_URL,
            model=OLLAMA_LOCAL_MODEL,
        )

    # ── Health check ───────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """
        Cloud Ollama: hits /api/tags with Authorization header.
        Local Ollama: hits /api/tags with no auth.
        """
        headers = self._auth_headers()
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(f"{self.base_url}/api/tags", headers=headers)
                if r.status_code == 200:
                    data   = r.json()
                    names  = [m["name"] for m in data.get("models", [])]
                    if names and not any(self.model.split(":")[0] in n for n in names):
                        logger.warning(
                            f"Model '{self.model}' not found at {self.base_url}. "
                            f"Available: {names}"
                        )
                    return True
                logger.error(f"Ollama health check failed: HTTP {r.status_code} at {self.base_url}")
                return False
        except Exception as e:
            logger.error(f"Cannot reach Ollama at {self.base_url}: {e}")
            return False

    # ── Streaming chat ─────────────────────────────────────────────────────

    async def chat_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        image_paths: list[str] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens.
        Accepts optional image_paths for vision tasks.
        Respects cancel_event for mid-stream cancellation.
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_msg: dict = {"role": "user", "content": prompt}
        if image_paths:
            encoded = [b64 for p in image_paths if (b64 := _encode_image(p))]
            if encoded:
                user_msg["images"] = encoded

        messages.append(user_msg)

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  True,
            "options": {
                "temperature": 0.2,
                "num_predict": 2048,
                "top_p":       0.9,
            },
        }

        headers = self._auth_headers()

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if cancel_event and cancel_event.is_set():
                            logger.info("Stream cancelled by user")
                            return
                        if not line.strip():
                            continue
                        try:
                            data  = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done"):
                                return
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            msg = f"Ollama HTTP {e.response.status_code} at {self.base_url}"
            if e.response.status_code == 401:
                msg += " — check your OLLAMA_CLOUD_API_KEY in .env"
            logger.error(msg)
            yield f"\n\n⚠️ {msg}"
        except httpx.ConnectError:
            msg = f"Cannot connect to Ollama at {self.base_url}"
            if "localhost" in self.base_url:
                msg += " — is Ollama running? Try: ollama serve"
            logger.error(msg)
            yield f"\n\n⚠️ {msg}"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\n\n⚠️ Unexpected error: {e}"

    # ── Non-streaming summarisation (used by indexer) ──────────────────────

    async def generate_summary(
        self,
        text: str,
        doc_name: str,
        chunk_index: int = 0,
        image_paths: list[str] | None = None,
    ) -> str:
        """
        Called by the indexer for each chunk.
        Uses cloud Ollama for speed + quality.
        Returns a dense factual summary.
        """
        prompt = (
            f"Document: '{doc_name}' | Chunk #{chunk_index}\n\n"
            f"Content:\n{text}\n\n"
            "Write a dense factual summary capturing ALL key information: "
            "names, numbers, dates, concepts, relationships, and technical details. "
            "This summary will be used for semantic search retrieval — be thorough."
        )
        result = ""
        async for token in self.chat_stream(
            prompt=prompt,
            image_paths=image_paths,
            system_prompt=(
                "You are a precise document analyst. "
                "Summarise content faithfully, densely, and completely."
            ),
        ):
            result += token
        return result.strip() or text[:500]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        """
        Cloud Ollama requires:  Authorization: Bearer <api_key>
        Local Ollama needs no auth header at all.
        """
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def __repr__(self) -> str:
        kind = "cloud" if self.api_key else "local"
        return f"OllamaClient({kind}, model={self.model}, url={self.base_url})"


# ── Image encoding helper ───────────────────────────────────────────────────

def _encode_image(path: str) -> str | None:
    try:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Image not found: {path}")
            return None
        return base64.b64encode(p.read_bytes()).decode()
    except Exception as e:
        logger.error(f"Image encode error ({path}): {e}")
        return None