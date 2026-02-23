"""
Config — reads from .env file.
Copy .env.example to .env and fill in your Ollama cloud API key.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Cloud Ollama — used by indexer (fast, powerful, runs once) ─
OLLAMA_CLOUD_URL:     str = os.getenv("OLLAMA_CLOUD_URL",     "https://api.ollama.com")
OLLAMA_CLOUD_API_KEY: str = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_MODEL:   str = os.getenv("OLLAMA_CLOUD_MODEL",   "llama3.2-vision")

# ── Local Ollama — used at query time (private, offline) ───────
OLLAMA_LOCAL_URL:   str = os.getenv("OLLAMA_LOCAL_URL",   "http://localhost:11434")
OLLAMA_LOCAL_MODEL: str = os.getenv("OLLAMA_LOCAL_MODEL", "llama3.2-vision")