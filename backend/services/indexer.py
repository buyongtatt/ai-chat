"""
Indexer — offline document processor using pure-Python libraries only.
No compilation required. Works on Windows, Mac, Linux out of the box.

Libraries used:
  - pypdf        : PDF text + image extraction (pure Python)
  - Pillow       : image saving/conversion
  - python-docx  : Word document text extraction

Usage:
    cd backend
    python -m services.indexer              # Process new/changed files only
    python -m services.indexer --force      # Reprocess everything
    python -m services.indexer --file report.pdf
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import re
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge_base"
CACHE_DIR     = Path(__file__).parent.parent / "processed_cache"
IMAGES_DIR    = CACHE_DIR / "extracted_images"

SUPPORTED    = {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
CHUNK_SIZE   = 1200
CHUNK_OVERLAP = 150


class DocumentIndexer:
    def __init__(self, force: bool = False):
        from services.ollama_client import OllamaClient
        self.client = OllamaClient.local()
        self.force  = force
        CACHE_DIR.mkdir(exist_ok=True)
        IMAGES_DIR.mkdir(exist_ok=True)

    def _doc_id(self, path: Path) -> str:
        return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:14]

    def _cache_path(self, doc_id: str) -> Path:
        return CACHE_DIR / f"{doc_id}.json"

    def _is_stale(self, path: Path, doc_id: str) -> bool:
        if self.force:
            return True
        cache = self._cache_path(doc_id)
        return not cache.exists() or path.stat().st_mtime > cache.stat().st_mtime

    async def run(self, single_file: str | None = None):
        if not await self.client.is_available():
            logger.error("Ollama not reachable at localhost:11434 — start Ollama first.")
            return

        if single_file:
            files = [KNOWLEDGE_DIR / single_file]
        else:
            files = [f for f in KNOWLEDGE_DIR.rglob("*") if f.suffix.lower() in SUPPORTED]

        logger.info(f"Found {len(files)} file(s) in knowledge_base/")

        for fp in files:
            if not fp.exists():
                logger.warning(f"  Not found: {fp}")
                continue
            doc_id = self._doc_id(fp)
            if not self._is_stale(fp, doc_id):
                logger.info(f"  Skipping (cached): {fp.name}")
                continue
            logger.info(f"  Processing: {fp.name}")
            try:
                t0 = time.perf_counter()
                chunks = await self._dispatch(fp, doc_id)
                self._save(doc_id, fp, chunks)
                elapsed = time.perf_counter() - t0
                logger.info(f"    -> {len(chunks)} chunks saved in {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"    ERROR {fp.name}: {e}", exc_info=True)

        logger.info("Indexing complete.")

    async def _dispatch(self, path: Path, doc_id: str) -> list[dict]:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return await self._process_pdf(path, doc_id)
        elif ext in {".txt", ".md"}:
            return await self._process_text_file(path, doc_id)
        elif ext == ".docx":
            return await self._process_docx(path, doc_id)
        elif ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return await self._process_image_file(path, doc_id)
        return []

    async def _process_pdf(self, path: Path, doc_id: str) -> list[dict]:
        from pypdf import PdfReader
        chunks: list[dict] = []
        reader = PdfReader(str(path))

        for page_num, page in enumerate(reader.pages):
            # Text
            text = (page.extract_text() or "").strip()
            if text:
                for i, chunk_text in enumerate(split_text(text)):
                    cid = f"{doc_id}_p{page_num}_t{i}"
                    summary = await self.client.generate_summary(
                        chunk_text, path.name, chunk_index=len(chunks)
                    )
                    chunks.append(make_text_chunk(
                        cid, doc_id, path.name, summary, chunk_text, len(chunks), {"page": page_num}
                    ))

            # Images
            try:
                page_images = page.images
            except Exception:
                page_images = []

            for img_idx, img_obj in enumerate(page_images):
                try:
                    img_bytes    = img_obj.data
                    raw_name     = getattr(img_obj, "name", f"img{img_idx}")
                    ext          = Path(raw_name).suffix.lower().lstrip(".") or "png"
                    ext          = ext if ext in {"png","jpg","jpeg","webp","gif"} else "png"
                    img_filename = f"{doc_id}_p{page_num}_i{img_idx}.{ext}"
                    img_path     = IMAGES_DIR / img_filename

                    _save_image_bytes(img_bytes, img_path)

                    cid = f"{doc_id}_p{page_num}_img{img_idx}"
                    summary = await self.client.generate_summary(
                        f"Image on page {page_num + 1} of '{path.name}'",
                        path.name,
                        chunk_index=len(chunks),
                        image_paths=[str(img_path)],
                    )
                    chunks.append(make_image_chunk(
                        cid, doc_id, path.name, summary, len(chunks), str(img_path),
                        {"page": page_num, "img_index": img_idx}
                    ))
                except Exception as e:
                    logger.warning(f"    Image error p{page_num} i{img_idx}: {e}")

        return chunks

    async def _process_text_file(self, path: Path, doc_id: str) -> list[dict]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return await self._chunk_and_summarize(text, doc_id, path.name)

    async def _process_docx(self, path: Path, doc_id: str) -> list[dict]:
        try:
            from docx import Document
            doc  = Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            logger.warning("python-docx not installed — reading as plain text")
            text = path.read_text(errors="ignore")
        return await self._chunk_and_summarize(text, doc_id, path.name)

    async def _process_image_file(self, path: Path, doc_id: str) -> list[dict]:
        dest = IMAGES_DIR / path.name
        if not dest.exists():
            _save_image_bytes(path.read_bytes(), dest)
        cid     = f"{doc_id}_img0"
        summary = await self.client.generate_summary(
            f"Standalone image: {path.name}", path.name,
            chunk_index=0, image_paths=[str(dest)],
        )
        return [make_image_chunk(cid, doc_id, path.name, summary, 0, str(dest), {})]

    async def _chunk_and_summarize(self, text: str, doc_id: str, name: str) -> list[dict]:
        chunks: list[dict] = []
        for i, chunk_text in enumerate(split_text(text)):
            cid     = f"{doc_id}_t{i}"
            summary = await self.client.generate_summary(chunk_text, name, chunk_index=i)
            chunks.append(make_text_chunk(cid, doc_id, name, summary, chunk_text, i, {}))
        return chunks

    def _save(self, doc_id: str, path: Path, chunks: list[dict]):
        data = {
            "doc_id":      doc_id,
            "doc_name":    path.name,
            "doc_type":    path.suffix.lower().lstrip("."),
            "source_path": str(path),
            "chunks":      chunks,
        }
        self._cache_path(doc_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def make_text_chunk(cid, doc_id, name, summary, raw, idx, meta) -> dict:
    return {
        "id": cid, "doc_id": doc_id, "doc_name": name,
        "content": summary, "raw_preview": raw[:400],
        "chunk_index": idx, "source_type": "text",
        "image_path": None, "metadata": meta,
    }


def make_image_chunk(cid, doc_id, name, summary, idx, img_path, meta) -> dict:
    return {
        "id": cid, "doc_id": doc_id, "doc_name": name,
        "content": summary, "raw_preview": "",
        "chunk_index": idx, "source_type": "image",
        "image_path": img_path, "metadata": meta,
    }


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) > size and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap:] + " " + sent if overlap else sent
        else:
            buf = (buf + " " + sent).strip()
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) > 20]


def _save_image_bytes(data: bytes, dest: Path):
    """Save image bytes via Pillow for format validation and normalisation."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        ext = dest.suffix.lower().lstrip(".")
        fmt = {"jpg": "JPEG", "jpeg": "JPEG"}.get(ext, ext.upper()) or "PNG"
        img.save(str(dest), format=fmt)
    except Exception:
        dest.write_bytes(data)


async def main():
    parser = argparse.ArgumentParser(description="DocMind offline indexer")
    parser.add_argument("--force", action="store_true", help="Reprocess all files")
    parser.add_argument("--file",  type=str,            help="Process single file by name")
    args    = parser.parse_args()
    indexer = DocumentIndexer(force=args.force)
    await indexer.run(single_file=args.file)


if __name__ == "__main__":
    asyncio.run(main())