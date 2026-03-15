from openai import AsyncOpenAI
from typing import Tuple
from loguru import logger
from app.config import get_settings
from app.models.schemas import RouteType
import httpx

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a RAG system.
Given a user query, determine:
1. confidence: float 0.0-1.0 (how confident you are the query is answerable from the knowledge base)
2. intent: "factual" | "exploratory" | "ambiguous"

Respond ONLY in JSON:
{"confidence": 0.85, "intent": "factual"}
"""


class QueryRouter:
    async def classify(self, query: str) -> Tuple[float, str]:
        """Classify query intent and return (confidence, intent_type)."""
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            import json
            parsed = json.loads(result)
            confidence = float(parsed.get("confidence", 0.5))
            intent = parsed.get("intent", "ambiguous")
            logger.info(f"Query classified: confidence={confidence:.2f}, intent={intent}")
            return confidence, intent
        except Exception as e:
            logger.warning(f"Router classification failed: {e}. Defaulting to hybrid.")
            return 0.7, "factual"

    async def route(self, query: str) -> Tuple[RouteType, float]:
        """
        Route query based on confidence threshold (η=0.6):
        - confidence >= 0.6 → HYBRID retrieval
        - confidence < 0.6  → EXTERNAL API fallback
        """
        confidence, intent = await self.classify(query)

        if confidence >= settings.confidence_threshold:
            logger.info(f"Route: HYBRID (conf={confidence:.2f} >= η={settings.confidence_threshold})")
            return RouteType.HYBRID, confidence
        else:
            logger.info(f"Route: EXTERNAL (conf={confidence:.2f} < η={settings.confidence_threshold})")
            return RouteType.EXTERNAL, confidence

    async def external_search(self, query: str) -> str:
        """Fallback to Tavily or Wikipedia for low-confidence queries."""
        # Try Tavily first
        if settings.tavily_api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as http:
                    resp = await http.post(
                        "https://api.tavily.com/search",
                        json={"api_key": settings.tavily_api_key, "query": query, "max_results": 3},
                    )
                    data = resp.json()
                    results = data.get("results", [])
                    return "\n\n".join([r.get("content", "") for r in results[:3]])
            except Exception as e:
                logger.warning(f"Tavily failed: {e}. Falling back to Wikipedia.")

        # Wikipedia fallback
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia("hybrid-rag-demo/1.0", "en")
            search_term = " ".join(query.split()[:4])
            page = wiki.page(search_term)
            if page.exists():
                return page.summary[:2000]
        except Exception as e:
            logger.error(f"Wikipedia fallback failed: {e}")

        return "No external context found."


router = QueryRouter()
