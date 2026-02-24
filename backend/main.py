"""
DocMind — Local AI Document Intelligence
FastAPI + Ollama qwen3-vl backend
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import router
from core.knowledge_loader import KnowledgeLoader
from core.memory_store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).parent / "processed_cache" / "extracted_images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: process knowledge_base/ if needed, then load cache into RAM."""
    logger.info("🚀 DocMind API starting...")
    store = MemoryStore.get_instance()
    loader = KnowledgeLoader()
    await loader.load_all_into_memory(store)
    logger.info(
        f"✅ Ready — {store.total_documents()} docs, "
        f"{store.total_chunks()} chunks in memory"
    )
    yield
    logger.info("👋 DocMind shutting down")


app = FastAPI(
    title="DocMind API",
    description="Offline AI Q&A with vision powered by Ollama qwen3-vl",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted images statically so frontend can display them
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    store = MemoryStore.get_instance()
    return {
        "status": "ok",
        "model": "qwen3-vl",
        "documents": store.total_documents(),
        "chunks": store.total_chunks(),
    }
