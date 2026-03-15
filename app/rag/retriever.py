import asyncio
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from typing import List
from loguru import logger
from app.config import get_settings
from app.models.schemas import RetrievedChunk

settings = get_settings()


class HybridRetriever:
    def __init__(self):
        self.qdrant: AsyncQdrantClient = None
        self.sql_session: AsyncSession = None
        self.embedder: SentenceTransformer = None

    async def connect(self):
        # Qdrant
        self.qdrant = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info("Qdrant connected.")

        # MySQL (async)
        db_url = (
            f"mysql+aiomysql://{settings.mysql_user}:{settings.mysql_password}"
            f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}"
        )
        engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
        self.sql_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        logger.info("MySQL connected.")

        # Embedding model
        self.embedder = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        logger.info(f"Embedder loaded: {settings.embedding_model}")

    async def semantic_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Qdrant vector search using BGE-large embeddings."""
        loop = asyncio.get_event_loop()
        prompt = f"Represent this sentence for searching relevant passages: {query}"
        query_vector = await loop.run_in_executor(
            None,
            lambda: self.embedder.encode(prompt, normalize_embeddings=True).tolist()
        )

        results = await self.qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                content=r.payload.get("text", ""),
                source=r.payload.get("source", "qdrant"),
                score=round(r.score, 4),
                retrieval_type="semantic",
            )
            for r in results
        ]

    async def keyword_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """MySQL full-text / keyword exact-match search."""
        async with self.sql_session() as session:
            result = await session.execute(
                text(
                    "SELECT content, source, "
                    "MATCH(content) AGAINST(:q IN NATURAL LANGUAGE MODE) AS score "
                    "FROM documents "
                    "WHERE MATCH(content) AGAINST(:q IN NATURAL LANGUAGE MODE) > 0 "
                    "ORDER BY score DESC LIMIT :lim"
                ),
                {"q": query, "lim": top_k},
            )
            rows = result.fetchall()

        return [
            RetrievedChunk(
                content=row.content,
                source=row.source,
                score=round(float(row.score), 4),
                retrieval_type="keyword",
            )
            for row in rows
        ]

    def _reciprocal_rank_fusion(
        self,
        semantic: List[RetrievedChunk],
        keyword: List[RetrievedChunk],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        """Fuse semantic + keyword results using Reciprocal Rank Fusion (RRF)."""
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(semantic):
            key = chunk.content[:100]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            chunks[key] = chunk

        for rank, chunk in enumerate(keyword):
            key = chunk.content[:100]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            if key not in chunks:
                chunks[key] = chunk

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        fused = []
        for key in sorted_keys:
            c = chunks[key]
            c.score = round(scores[key], 6)
            fused.append(c)

        return fused

    async def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Main hybrid retrieval: semantic + keyword → RRF fusion."""
        semantic, keyword = await asyncio.gather(
            self.semantic_search(query, top_k),
            self.keyword_search(query, top_k),
        )
        fused = self._reciprocal_rank_fusion(semantic, keyword)
        logger.info(f"Retrieved {len(fused)} fused chunks (sem={len(semantic)}, kw={len(keyword)})")
        return fused[:top_k]

    async def ping_qdrant(self) -> bool:
        try:
            await self.qdrant.get_collections()
            return True
        except Exception:
            return False

retriever = HybridRetriever()
