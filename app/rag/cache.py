import redis.asyncio as aioredis
import hashlib
import json
from typing import Optional
from loguru import logger
from app.config import get_settings

settings = get_settings()


class CacheLayer:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            decode_responses=True,
        )
        logger.info("Redis cache connected.")

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    def _make_key(self, query: str, top_k: int) -> str:
        raw = f"{query.strip().lower()}::{top_k}"
        return "rag:" + hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, query: str, top_k: int) -> Optional[dict]:
        if not self._client:
            return None
        key = self._make_key(query, top_k)
        cached = await self._client.get(key)
        if cached:
            logger.debug(f"Cache HIT for key: {key[:16]}...")
            return json.loads(cached)
        logger.debug(f"Cache MISS for key: {key[:16]}...")
        return None

    async def set(self, query: str, top_k: int, value: dict) -> None:
        if not self._client:
            return
        key = self._make_key(query, top_k)
        await self._client.setex(key, settings.cache_ttl, json.dumps(value))
        logger.debug(f"Cached response for key: {key[:16]}...")

    async def ping(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False


cache = CacheLayer()
