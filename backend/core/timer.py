"""
Timer utility — simple context manager and decorator for logging process times.
Usage:
    with Timer("search chunks"):
        results = store.search(query)

    async with AsyncTimer("ollama stream"):
        async for token in client.chat_stream(...):
            ...
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger("docmind.timer")


@contextmanager
def Timer(label: str):
    start = time.perf_counter()
    logger.info(f"  ⏱  [{label}] starting...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"  ✅ [{label}] done in {elapsed:.3f}s")


@asynccontextmanager
async def AsyncTimer(label: str):
    start = time.perf_counter()
    logger.info(f"  ⏱  [{label}] starting...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"  ✅ [{label}] done in {elapsed:.3f}s")