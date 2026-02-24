"""
OllamaClient — unified async client for both:
  • Cloud Ollama  (https://api.ollama.com)  → used by indexer
  • Local Ollama  (http://localhost:11434)  → used for answering questions

Auto-detects the correct installed model name so partial names like
"qwen3-vl" will match "qwen3-vl:latest" automatically.
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

TIMEOUT = 300  # seconds


class OllamaClient:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url  = base_url.rstrip("/")
        self.model     = model          # may be partial e.g. "llama3.2-vision"
        self.api_key   = api_key
        self._resolved: str | None = None   # exact matched model name from /api/tags

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def cloud(cls) -> "OllamaClient":
        """Cloud Ollama — used by indexer."""
        from core.config import OLLAMA_CLOUD_URL, OLLAMA_CLOUD_MODEL, OLLAMA_CLOUD_API_KEY
        return cls(base_url=OLLAMA_CLOUD_URL, model=OLLAMA_CLOUD_MODEL, api_key=OLLAMA_CLOUD_API_KEY)

    @classmethod
    def local(cls) -> "OllamaClient":
        """Local Ollama — used for answering questions."""
        from core.config import OLLAMA_LOCAL_URL, OLLAMA_LOCAL_MODEL
        return cls(base_url=OLLAMA_LOCAL_URL, model=OLLAMA_LOCAL_MODEL)

    # ── Model resolution ───────────────────────────────────────────────────

    async def resolve_model(self) -> str:
        """
        Fetch /api/tags and find the best matching model name.
        e.g. "llama3.2-vision" → "llama3.2-vision:latest"
        Falls back to self.model if nothing matches.
        """
        if self._resolved:
            return self._resolved

        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(
                    f"{self.base_url}/api/tags",
                    headers=self._auth_headers(),
                )
                r.raise_for_status()
                installed = [m["name"] for m in r.json().get("models", [])]

            if not installed:
                logger.warning(f"No models found at {self.base_url}")
                self._resolved = self.model
                return self._resolved

            # Exact match first
            if self.model in installed:
                self._resolved = self.model
                logger.info(f"Model resolved (exact): {self._resolved}")
                return self._resolved

            # Partial match — e.g. "llama3.2-vision" matches "llama3.2-vision:latest"
            base = self.model.split(":")[0].lower()
            matches = [n for n in installed if n.lower().startswith(base)]
            if matches:
                self._resolved = matches[0]
                logger.info(
                    f"Model resolved (partial match): '{self.model}' → '{self._resolved}'"
                )
                return self._resolved

            # Nothing matched — log all available and fallback
            logger.error(
                f"Model '{self.model}' not found at {self.base_url}.\n"
                f"  Installed models: {installed}\n"
                f"  Fix: update OLLAMA_LOCAL_MODEL in .env to one of the above,\n"
                f"       or run: ollama pull {self.model}"
            )
            self._resolved = self.model   # use as-is, will likely 500
            return self._resolved

        except Exception as e:
            logger.warning(f"Could not resolve model name: {e}")
            self._resolved = self.model
            return self._resolved

    async def list_models(self) -> list[str]:
        """Return list of installed model names."""
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(
                    f"{self.base_url}/api/tags",
                    headers=self._auth_headers(),
                )
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ── Health check ───────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        models = await self.list_models()
        if not models and not self.api_key:
            logger.error(
                f"Cannot reach Ollama at {self.base_url}\n"
                "  Make sure Ollama is running: ollama serve"
            )
            return False
        # Pre-resolve model name so first query is fast
        await self.resolve_model()
        return True

    # ── Streaming chat ─────────────────────────────────────────────────────

    async def chat_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        image_paths: list[str] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens with auto model resolution."""

        model_name = await self.resolve_model()

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_msg: dict = {"role": "user", "content": prompt}
        if image_paths:
            encoded = [b64 for p in image_paths if (b64 := _encode_image(p))]
            if encoded:
                user_msg["images"] = encoded
                logger.debug(f"Sending {len(encoded)} image(s) to model")

        messages.append(user_msg)

        # Cloud Ollama uses max_tokens (must be positive integer)
        # Local Ollama uses num_predict (-1 = unlimited)
        is_cloud = bool(self.api_key)
        if is_cloud:
            gen_options = {
                "temperature": 0.2,
                "top_p":       0.9,
                "max_tokens":  4096,  # cloud API requires positive integer
            }
        else:
            gen_options = {
                "temperature": 0.2,
                "top_p":       0.9,
                "num_predict": -1,    # local: -1 = no limit, stop naturally
                "num_ctx":     8192,  # local context window
            }

        payload = {
            "model":    model_name,
            "messages": messages,
            "stream":   True,
            "options":  gen_options,
        }

        logger.debug(f"Sending to {self.base_url}/api/chat — model={model_name}")

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=self._auth_headers(),
                ) as resp:

                    # ── Handle error status before streaming ──
                    if resp.status_code != 200:
                        body = await resp.aread()
                        await self._handle_error(resp.status_code, body, model_name)
                        return

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

        except httpx.ConnectError:
            msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                + ("Run: ollama serve" if "localhost" in self.base_url else "Check your network.")
            )
            logger.error(msg)
            yield f"\n\n⚠️ {msg}"
        except httpx.ReadTimeout:
            yield "\n\n⚠️ Ollama timed out. The model may be loading — try again in a moment."
        except Exception as e:
            logger.error(f"Unexpected stream error: {e}", exc_info=True)
            yield f"\n\n⚠️ Unexpected error: {e}"

    async def _handle_error(self, status: int, body: bytes, model_name: str):
        """Decode and log Ollama error responses clearly."""
        try:
            detail = json.loads(body).get("error", body.decode())
        except Exception:
            detail = body.decode(errors="replace")

        if status == 500:
            installed = await self.list_models()
            logger.error(
                f"Ollama 500 Internal Server Error\n"
                f"  Model used: {model_name}\n"
                f"  Error: {detail}\n"
                f"  Installed models: {installed}\n"
                f"  Fix: set OLLAMA_LOCAL_MODEL in .env to one of: {installed}\n"
                f"       or run: ollama pull {model_name}"
            )
            raise httpx.HTTPStatusError(
                f"500 — model '{model_name}' error: {detail}. "
                f"Installed: {installed}",
                request=None, response=None,  # type: ignore
            )
        elif status == 401:
            logger.error(f"Ollama 401 Unauthorized — check OLLAMA_CLOUD_API_KEY in .env")
            raise httpx.HTTPStatusError("401 Unauthorized", request=None, response=None)  # type: ignore
        elif status == 404:
            logger.error(f"Ollama 404 — model '{model_name}' not found. Run: ollama pull {model_name}")
            raise httpx.HTTPStatusError(f"404 — model not found: {model_name}", request=None, response=None)  # type: ignore
        else:
            raise httpx.HTTPStatusError(f"HTTP {status}: {detail}", request=None, response=None)  # type: ignore

    # ── Non-streaming summary (used by indexer) ────────────────────────────

    async def generate_summary(
        self,
        text: str,
        doc_name: str,
        chunk_index: int = 0,
        image_paths: list[str] | None = None,
    ) -> str:
        prompt = (
            f"Document: '{doc_name}' | Chunk #{chunk_index}\n\n"
            f"Content:\n{text}\n\n"
            "Write a dense factual summary capturing ALL key information: "
            "names, numbers, dates, concepts, relationships, and technical details. "
            "This summary is used for semantic search — be thorough."
        )
        result = ""
        async for token in self.chat_stream(
            prompt=prompt,
            image_paths=image_paths,
            system_prompt="You are a precise document analyst. Summarise content faithfully and densely.",
        ):
            result += token
        return result.strip() or text[:500]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def __repr__(self) -> str:
        kind = "cloud" if self.api_key else "local"
        return f"OllamaClient({kind}, model={self.model}, url={self.base_url})"


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