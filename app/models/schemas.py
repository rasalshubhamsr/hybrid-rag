from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RouteType(str, Enum):
    HYBRID = "hybrid"
    EXTERNAL = "external"
    CACHE = "cache"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    use_cache: bool = Field(default=True, description="Whether to use Redis cache")


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    retrieval_type: str  # "semantic" | "keyword"


class QueryResponse(BaseModel):
    query: str
    answer: str
    route_used: RouteType
    confidence_score: float
    retrieved_chunks: List[RetrievedChunk]
    from_cache: bool
    latency_ms: float


class IngestRequest(BaseModel):
    source: str = Field(default="arxiv", description="Data source: arxiv | wikipedia")
    limit: int = Field(default=100, ge=10, le=1000)
    topic: Optional[str] = Field(default="retrieval augmented generation")


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    mysql: str
    redis: str
