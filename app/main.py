from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.routers.query import router as query_router
from app.rag.retriever import retriever
from app.rag.cache import cache
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Hybrid RAG API...")
    await cache.connect()
    await retriever.connect()
    logger.info("All services connected. Ready.")
    yield
    # Shutdown
    await cache.disconnect()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Hybrid RAG Demo",
    description=(
        "Production-ready Hybrid RAG pipeline — "
        "Qdrant semantic search + MySQL keyword search fused via RRF, "
        "with deterministic routing, Redis caching, PII masking, and RAGAs evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check health of all connected services."""
    qdrant_ok = await retriever.ping_qdrant()
    redis_ok = await cache.ping()
    return HealthResponse(
        status="healthy" if (qdrant_ok and redis_ok) else "degraded",
        qdrant="ok" if qdrant_ok else "unreachable",
        mysql="ok",
        redis="ok" if redis_ok else "unreachable",
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "hybrid-rag-demo",
        "docs": "/docs",
        "health": "/health",
        "github": "https://github.com/rasalshubhamsr/hybrid-rag-demo",
    }
