"""
VectorStore
===========
Qdrant-backed semantic storage for conversation chapters.

Why Qdrant and not in-memory similarity search?
  At 50 messages per chapter, a 500-message conversation has 10 chapters.
  At 40+ chapters, semantic retrieval matters — you want the 3 most
  relevant chapters for a query, not all 40.
  Qdrant handles this with approximate nearest neighbour search (ANN).

Why store chapter embeddings and not message embeddings?
  Chapter embeddings are computed once at ingest time and stored.
  Query time: embed the query, search Qdrant, retrieve top-K chapters.

Embedding model:
  Same all-MiniLM-L6-v2 used by the scorer. 384 dimensions.
"""

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from optimizer.chapter import Chapter

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lec_chapters"
EMBEDDING_DIM   = 384
TOP_K_DEFAULT   = 3

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _embed(text: str) -> list:
    embedder = get_embedder()
    vector = embedder.encode([text], normalize_embeddings=True)[0]
    return vector.tolist()


class VectorStore:
    """
    Qdrant-backed store for chapter embeddings.

    Usage:
        store = VectorStore()
        store.ensure_collection()
        store.store_chapter(conversation_id, chapter)
        chapters = store.retrieve_relevant_chapters(conversation_id, query, top_k=3)
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self._collection_ready = False

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist. Idempotent."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", COLLECTION_NAME)
        self._collection_ready = True

    def store_chapter(self, conversation_id: str, chapter: Chapter) -> str:
        """
        Embed and store a chapter in Qdrant.
        Embeds the compressed_content summary.
        Stores landmark messages in payload for verbatim hydration at query time.
        """
        if not self._collection_ready:
            self.ensure_collection()

        vector = _embed(chapter.compressed_content)

        payload = {
            "conversation_id": conversation_id,
            "chapter_id": chapter.chapter_id,
            "content_hash": chapter.content_hash,
            "compressed_content": chapter.compressed_content,
            "start_index": chapter.start_index,
            "end_index": chapter.end_index,
            "original_message_count": chapter.original_message_count,
            "compression_cost_usd": chapter.compression_cost_usd,
            "landmark_count": len(chapter.landmark_messages),
            "landmark_messages": [
                {
                    "index": lm.index,
                    "role": lm.role.value,
                    "content": lm.content,
                    "tool_call_id": lm.tool_call_id,
                }
                for lm in chapter.landmark_messages
            ],
        }

        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.info("Stored chapter %s for conv %s", chapter.chapter_id, conversation_id)
        return point_id

    def retrieve_relevant_chapters(
        self,
        conversation_id: str,
        query: str,
        top_k: int = TOP_K_DEFAULT,
    ) -> list:
        """
        Retrieve top-K most semantically relevant chapters for a query.
        Filters by conversation_id — no cross-conversation leakage.
        Returns chapters ordered by relevance (most relevant first).
        """
        if not self._collection_ready:
            self.ensure_collection()

        query_vector = _embed(query)

        from qdrant_client.models import QueryRequest
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        ).points

        chapters = []
        for result in results:
            chapter = self._payload_to_chapter(result.payload)
            if chapter:
                chapters.append(chapter)
        return chapters

    def get_all_chapters(self, conversation_id: str) -> list:
        """Retrieve ALL chapters for a conversation in chronological order."""
        if not self._collection_ready:
            self.ensure_collection()

        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id),
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
        )

        chapters = [self._payload_to_chapter(r.payload) for r in results]
        chapters = [c for c in chapters if c is not None]
        chapters.sort(key=lambda c: c.start_index)
        return chapters

    def chapter_count(self, conversation_id: str) -> int:
        """Count chapters stored for a conversation."""
        if not self._collection_ready:
            self.ensure_collection()
        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id),
                    )
                ]
            ),
            limit=1000,
        )
        return len(results)

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete all chapters for a conversation. Useful for testing."""
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id),
                    )
                ]
            ),
        )

    def _payload_to_chapter(self, payload: dict) -> Optional[Chapter]:
        """Reconstruct a Chapter from a Qdrant payload."""
        try:
            from optimizer.types import Message, MessageRole
            landmark_messages = [
                Message(
                    index=lm["index"],
                    role=MessageRole(lm["role"]),
                    content=lm["content"],
                    tool_call_id=lm.get("tool_call_id"),
                )
                for lm in payload.get("landmark_messages", [])
            ]
            return Chapter(
                chapter_id=payload["chapter_id"],
                content_hash=payload["content_hash"],
                compressed_content=payload["compressed_content"],
                landmark_messages=landmark_messages,
                original_message_count=payload["original_message_count"],
                start_index=payload["start_index"],
                end_index=payload["end_index"],
                compression_cost_usd=payload.get("compression_cost_usd", 0.0),
            )
        except Exception as exc:
            logger.error("Failed to reconstruct chapter from payload: %s", exc)
            return None
