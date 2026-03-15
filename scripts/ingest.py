"""
Data ingestion script — ArXiv / Wikipedia → Qdrant + MySQL

Usage:
    python scripts/ingest.py --source arxiv --limit 500 --topic "retrieval augmented generation"
    python scripts/ingest.py --source wikipedia --limit 100 --topic "large language models"
"""

import asyncio
import argparse
import arxiv
import wikipediaapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sqlalchemy import create_engine, text
from tqdm import tqdm
import uuid
import re
from loguru import logger

# Config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "hybrid_rag"
MYSQL_URL = "mysql+pymysql://raguser:ragpass@localhost:3306/ragdb"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i: i + size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def setup_mysql(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR(64) PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255),
                FULLTEXT KEY content_ft (content)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))
        conn.commit()
    logger.info("MySQL table ready.")


def setup_qdrant(client: QdrantClient):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info(f"Qdrant collection '{COLLECTION}' created.")
    else:
        logger.info(f"Qdrant collection '{COLLECTION}' already exists.")


def fetch_arxiv(topic: str, limit: int):
    logger.info(f"Fetching {limit} ArXiv papers on: '{topic}'")
    search = arxiv.Search(query=topic, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
    docs = []
    for paper in search.results():
        text = f"{paper.title}\n\n{paper.summary}"
        docs.append({"text": clean(text), "source": f"arxiv:{paper.entry_id}"})
    logger.info(f"Fetched {len(docs)} ArXiv documents.")
    return docs


def fetch_wikipedia(topic: str, limit: int):
    logger.info(f"Fetching Wikipedia pages on: '{topic}'")
    wiki = wikipediaapi.Wikipedia("hybrid-rag-demo/1.0", "en")
    page = wiki.page(topic)
    docs = []
    if page.exists():
        docs.append({"text": clean(page.text[:5000]), "source": f"wikipedia:{topic}"})
    logger.info(f"Fetched {len(docs)} Wikipedia pages.")
    return docs


def ingest(source: str, topic: str, limit: int):
    # Load models & clients
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    engine = create_engine(MYSQL_URL)

    setup_mysql(engine)
    setup_qdrant(qdrant)

    # Fetch docs
    if source == "arxiv":
        raw_docs = fetch_arxiv(topic, limit)
    else:
        raw_docs = fetch_wikipedia(topic, limit)

    # Chunk, embed, insert
    points = []
    sql_rows = []

    for doc in tqdm(raw_docs, desc="Processing documents"):
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            embedding = embedder.encode(
                f"Represent this sentence for searching relevant passages: {chunk}",
                normalize_embeddings=True,
            ).tolist()

            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": chunk, "source": doc["source"]},
            ))
            sql_rows.append({"id": doc_id, "content": chunk, "source": doc["source"]})

            # Batch upsert every 100
            if len(points) >= 100:
                qdrant.upsert(collection_name=COLLECTION, points=points)
                with engine.connect() as conn:
                    conn.execute(
                        text("INSERT IGNORE INTO documents (id, content, source) VALUES (:id, :content, :source)"),
                        sql_rows,
                    )
                    conn.commit()
                points.clear()
                sql_rows.clear()

    # Flush remaining
    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)
        with engine.connect() as conn:
            conn.execute(
                text("INSERT IGNORE INTO documents (id, content, source) VALUES (:id, :content, :source)"),
                sql_rows,
            )
            conn.commit()

    logger.success(f"Ingestion complete. Source={source}, Topic='{topic}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["arxiv", "wikipedia"], default="arxiv")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--topic", type=str, default="retrieval augmented generation")
    args = parser.parse_args()
    ingest(args.source, args.topic, args.limit)
