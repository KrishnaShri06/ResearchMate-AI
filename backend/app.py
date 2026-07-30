"""
Research Assistant Backend - Main Application Entry Point.

An AI-powered research paper recommendation engine that:
1. Accepts scientific figure uploads
2. Analyzes them using Grok Vision API
3. Recommends relevant papers via semantic search
4. Explains recommendations using Grok LLM
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.config import PROCESSED_CSV_PATH
from models.paper_loader import load_papers
from services.embedding_service import init_embeddings
from routes.chatbot import router as chatbot_router

# ── Logging Configuration ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: Startup & Shutdown Events ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load dataset and generate embeddings.
    Shutdown: Cleanup (no-op for now).
    """
    logger.info("=" * 60)
    logger.info("  Research Assistant Backend - Starting Up")
    logger.info("=" * 60)

    # Load research papers
    logger.info(f"Loading research papers (may download from HF if not cached)...")
    papers_df = load_papers(PROCESSED_CSV_PATH)
    logger.info(f"Loaded {len(papers_df)} papers successfully.")

    # Generate and cache embeddings
    logger.info("Generating paper embeddings (one-time operation)...")
    init_embeddings(papers_df)
    logger.info("Embeddings cached. Server is ready to accept requests.")

    logger.info("=" * 60)
    logger.info("  Startup Complete - Server Ready")
    logger.info("=" * 60)

    yield  # Server runs here

    logger.info("Shutting down Research Assistant Backend.")


# ── FastAPI Application ─────────────────────────────────────────────
app = FastAPI(
    title="Research Assistant API",
    description=(
        "AI-powered research paper recommendation engine. "
        "Upload a scientific figure and receive relevant paper recommendations "
        "with LLM-generated explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware (for future frontend integration) ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ─────────────────────────────────────────────────
app.include_router(chatbot_router)


# ── Health Check ─────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Research Assistant API",
        "version": "1.0.0",
    }


# ── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
