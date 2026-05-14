"""
HierarchicalStore
=================
Unified interface combining ConversationStore and VectorStore.

This is the single entry point for the ingest-time compression pipeline.
It wires together:
  - ConversationStore: in-memory live tail + frozen chapter list
  - VectorStore: Qdrant-backed semantic retrieval of chapters
  - ChapterCompressor: Haiku compression at chapter boundaries

Why a unified interface?
  The optimizer should not need to know about two separate stores.
  HierarchicalStore presents one clean API:
    - add_message() for ingest path
    - get_context(query) for query path
  Internally it manages the boundary detection, compression,
  and Qdrant storage automatically.

Retrieval strategy:
  Short conversations (< 5 chapters): retrieve ALL chapters chronologically.
  Long conversations (5+ chapters): retrieve top-K by semantic similarity.

  Why the threshold?
    At 4 chapters (~200 messages), loading all chapters is cheap.
    At 10+ chapters (~500 messages), semantic retrieval saves
    significant context window space by filtering irrelevant chapters.
"""

import logging
from typing import Optional

from optimizer.chapter import Chapter
from optimizer.chapter_compressor import compress_to_chapter
from optimizer.conversation_store import ConversationStore
from optimizer.vector_store import VectorStore
from optimizer.types import Message

logger = logging.getLogger(__name__)

# Switch from chronological to semantic retrieval above this threshold
SEMANTIC_RETRIEVAL_THRESHOLD = 5
TOP_K_CHAPTERS = 3


class HierarchicalStore:
    """
    Unified store for ingest-time compression with semantic retrieval.

    Usage:
        store = HierarchicalStore()

        # Ingest path
        result = store.add_message(conv_id, message)
        # result tells you if a chapter was created

        # Query path
        chapters, live_tail = store.get_context(conv_id, query)
        # chapters are the most relevant frozen chapters
        # live_tail is the uncompressed recent messages
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self._conv_store = ConversationStore()
        self._vec_store = VectorStore(host=qdrant_host, port=qdrant_port)
        self._vec_store.ensure_collection()

    # ── Ingest path ───────────────────────────────────────────────

    def add_message(self, conversation_id: str, message: Message) -> dict:
        """
        Add a message to the conversation.

        If a chapter boundary is hit:
          1. Compress the live tail via Haiku
          2. Freeze the chapter in ConversationStore
          3. Embed and store in Qdrant for semantic retrieval

        Returns a status dict describing what happened.
        """
        self._conv_store.add_message(conversation_id, message)

        if not self._conv_store.should_close_chapter(conversation_id):
            return {
                "status": "message_added",
                "live_tail_length": self._conv_store.live_tail_length(conversation_id),
            }

        # Chapter boundary hit — compress and freeze
        _, live_tail = self._conv_store.get_context(conversation_id)

        chapter, cost = compress_to_chapter(live_tail)

        if not chapter:
            logger.warning(
                "Chapter compression failed for conv %s — live tail continues growing",
                conversation_id,
            )
            return {
                "status": "compression_failed",
                "live_tail_length": self._conv_store.live_tail_length(conversation_id),
            }

        # Freeze in ConversationStore
        self._conv_store.close_chapter(
            conversation_id=conversation_id,
            compressed_content=chapter.compressed_content,
            landmark_messages=chapter.landmark_messages,
            compression_cost_usd=cost,
        )

        # Store in Qdrant for semantic retrieval
        point_id = self._vec_store.store_chapter(conversation_id, chapter)

        logger.info(
            "Chapter %s created and stored in Qdrant (point: %s, cost: $%.6f)",
            chapter.chapter_id, point_id, cost,
        )

        return {
            "status": "chapter_created",
            "chapter_id": chapter.chapter_id,
            "messages_compressed": chapter.original_message_count,
            "landmarks_preserved": len(chapter.landmark_messages),
            "cost_usd": cost,
            "qdrant_point_id": point_id,
        }

    # ── Query path ────────────────────────────────────────────────

    def get_context(
        self,
        conversation_id: str,
        query: str,
        top_k: int = TOP_K_CHAPTERS,
    ) -> tuple[list[Chapter], list[Message]]:
        """
        Retrieve relevant chapters + live tail for a query.

        Retrieval strategy:
          < 5 chapters: return ALL chapters (chronological order)
          >= 5 chapters: return top-K by semantic similarity to query

        Why this threshold?
          Under 5 chapters (~250 messages), all chapters fit comfortably
          in context. Semantic filtering would risk missing relevant content.
          Above 5 chapters, semantic retrieval prevents context bloat.

        Returns:
          (chapters, live_tail)
          chapters: ordered by start_index (chronological)
          live_tail: uncompressed recent messages
        """
        _, live_tail = self._conv_store.get_context(conversation_id)
        chapter_count = self._vec_store.chapter_count(conversation_id)

        if chapter_count == 0:
            return [], live_tail

        if chapter_count < SEMANTIC_RETRIEVAL_THRESHOLD:
            # Short conversation — retrieve all chapters chronologically
            chapters = self._vec_store.get_all_chapters(conversation_id)
            logger.debug(
                "Chronological retrieval: %d chapters for conv %s",
                len(chapters), conversation_id,
            )
        else:
            # Long conversation — semantic top-K retrieval
            chapters = self._vec_store.retrieve_relevant_chapters(
                conversation_id, query, top_k=top_k
            )
            # Sort retrieved chapters chronologically for assembly
            chapters.sort(key=lambda c: c.start_index)
            logger.info(
                "Semantic retrieval: top-%d of %d chapters for conv %s",
                top_k, chapter_count, conversation_id,
            )

        return chapters, live_tail

    # ── Inspection ────────────────────────────────────────────────

    def chapter_count(self, conversation_id: str) -> int:
        return self._vec_store.chapter_count(conversation_id)

    def live_tail_length(self, conversation_id: str) -> int:
        return self._conv_store.live_tail_length(conversation_id)

    def reset(self, conversation_id: str) -> None:
        """Clear all state for a conversation. Useful for testing."""
        self._conv_store.reset(conversation_id)
        self._vec_store.delete_conversation(conversation_id)
