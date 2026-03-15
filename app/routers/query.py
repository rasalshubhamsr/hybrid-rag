from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.hybrid import run_pipeline

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse, summary="Run hybrid RAG query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a query to the Hybrid RAG pipeline.

    - Checks Redis cache first (LRU, 8hr TTL)
    - Routes via LLM intent classifier (η=0.6 threshold)
    - High confidence → Hybrid retrieval (Qdrant + MySQL → RRF)
    - Low confidence → External API fallback (Tavily / Wikipedia)
    - Returns answer + retrieved chunks + latency + route used
    """
    try:
        return await run_pipeline(request)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
