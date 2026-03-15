import time
from openai import AsyncOpenAI
from loguru import logger
from app.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, RouteType, RetrievedChunk
from app.rag.retriever import retriever
from app.rag.router import router
from app.rag.cache import cache
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import List

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

# PII masking engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

RAG_SYSTEM_PROMPT = """You are a precise, factual assistant. 
Answer the user's question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
Never hallucinate or add information not present in the context.
Be concise and cite the source when possible."""


def mask_pii(text: str) -> str:
    """Strip PII from text before sending to LLM."""
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text


async def generate_answer(query: str, context_chunks: List[RetrievedChunk]) -> str:
    """Generate LLM response from retrieved context."""
    context = "\n\n---\n\n".join([
        f"[Source: {c.source}]\n{c.content}" for c in context_chunks
    ])

    masked_query = mask_pii(query)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {masked_query}"},
        ],
        temperature=0.1,
        max_tokens=settings.max_tokens,
    )
    return response.choices[0].message.content


async def run_pipeline(request: QueryRequest) -> QueryResponse:
    """Full hybrid RAG pipeline: cache → route → retrieve → generate."""
    start = time.perf_counter()

    # 1. Cache check
    if request.use_cache:
        cached = await cache.get(request.query, request.top_k)
        if cached:
            cached["from_cache"] = True
            cached["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            return QueryResponse(**cached)

    # 2. Route query
    route_type, confidence = await router.route(request.query)

    # 3. Retrieve context
    if route_type == RouteType.HYBRID:
        chunks = await retriever.retrieve(request.query, request.top_k)
    else:
        external_context = await router.external_search(request.query)
        chunks = [RetrievedChunk(
            content=external_context,
            source="external",
            score=confidence,
            retrieval_type="external",
        )]

    # 4. Generate answer
    answer = await generate_answer(request.query, chunks)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"Pipeline complete: route={route_type}, chunks={len(chunks)}, latency={latency_ms}ms")

    response_data = {
        "query": request.query,
        "answer": answer,
        "route_used": route_type,
        "confidence_score": round(confidence, 4),
        "retrieved_chunks": [c.model_dump() for c in chunks],
        "from_cache": False,
        "latency_ms": latency_ms,
    }

    # 5. Cache result
    if request.use_cache:
        await cache.set(request.query, request.top_k, response_data)

    return QueryResponse(**response_data)
