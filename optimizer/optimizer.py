"""
Context Optimizer
=================
Orchestrates the full pipeline:
  1. Detect query type (adaptive strategy)
  2. Score all messages (four signals, adaptive weights)
  3. Compress low-value groups (LLM summarisation)
  4. Assemble valid output thread
  5. Return OptimizedContext with full audit trail
"""

import time

from optimizer.assembler import Assembler
from optimizer.compressor import Compressor
from optimizer.scorer import RelevanceScorer, WEIGHT_PROFILES
from optimizer.types import (
    Message, MessageClass, OptimizedContext, QueryType, ScoredMessage
)


class ContextOptimizer:
    """
    Main entry point for the context optimization system.

    Usage:
        optimizer = ContextOptimizer()
        result = optimizer.optimize(messages, query)
        # result.messages → valid, compressed conversation thread
        # result.token_reduction_pct → how much smaller it is
    """

    def __init__(
        self,
        recency_lambda: float = 0.015,
        min_group_size: int = 2,
    ):
        self.scorer = RelevanceScorer(recency_lambda=recency_lambda)
        self.compressor = Compressor(min_group_size=min_group_size)
        self.assembler = Assembler()

    def optimize(
        self,
        messages: list[Message],
        query: str,
        query_type: QueryType | None = None,
    ) -> OptimizedContext:
        """
        Full pipeline: score → compress → assemble.

        Args:
            messages: Full conversation history
            query: Current user question
            query_type: Override auto-detection if known

        Returns:
            OptimizedContext with compressed messages + full metrics
        """
        start = time.perf_counter()

        if not messages:
            return self._empty_result()

        # Step 1: Detect query type (adaptive strategy)
        if query_type is None:
            query_type = self.scorer.detect_query_type(query)

        weights = WEIGHT_PROFILES[query_type]

        # Step 2: Score all messages
        scored: list[ScoredMessage] = self.scorer.score_all(
            messages, query, query_type
        )

        # Step 3: Compress
        compression = self.compressor.compress(scored)

        # Step 4: Count stats
        landmarks_preserved = sum(
            1 for sm in scored
            if sm.classification == MessageClass.LANDMARK
        )
        kept_verbatim = len(compression.kept_messages)

        original_tokens = sum(
            max(1, len(m.content) // 4) for m in messages
        )

        assembly_latency_ms = (time.perf_counter() - start) * 1000

        # Step 5: Assemble
        stats = {
            "kept_verbatim": kept_verbatim,
            "compressed_groups": compression.compressed_groups,
            "discarded": compression.discarded_count,
            "landmarks_preserved": landmarks_preserved,
        }

        result = self.assembler.assemble(
            kept_messages=compression.kept_messages,
            summary_messages=compression.summary_messages,
            original_token_count=original_tokens,
            compression_cost_usd=compression.total_cost_usd,
            query_type=query_type,
            scoring_weights=weights,
            assembly_latency_ms=round(assembly_latency_ms, 1),
            stats=stats,
        )

        return result

    def score_only(
        self, messages: list[Message], query: str
    ) -> list[ScoredMessage]:
        """Expose raw scoring for inspection and debugging."""
        return self.scorer.score_all(messages, query)

    def _empty_result(self) -> OptimizedContext:
        return OptimizedContext(
            messages=[],
            original_token_count=0,
            optimized_token_count=0,
            token_reduction_pct=0.0,
            kept_verbatim=0,
            compressed_groups=0,
            discarded=0,
            landmarks_preserved=0,
            assembly_latency_ms=0.0,
            compression_cost_usd=0.0,
            query_type=QueryType.GENERAL,
            scoring_weights=WEIGHT_PROFILES[QueryType.GENERAL],
        )


# ── Ingest-time compression ───────────────────────────────────────────────────

from optimizer.conversation_store import ConversationStore
from optimizer.chapter_compressor import compress_to_chapter


def ingest_message(
    store: ConversationStore,
    conversation_id: str,
    message: Message,
) -> dict:
    """
    Ingest path — called when a new message arrives.

    Adds the message to the live tail. If a chapter boundary is hit,
    compresses the live tail into a frozen Chapter automatically.

    Returns a status dict describing what happened.
    """
    store.add_message(conversation_id, message)

    if store.should_close_chapter(conversation_id):
        chapters, live_tail = store.get_context(conversation_id)

        # Compress the live tail into a chapter
        chapter, cost = compress_to_chapter(live_tail)

        if chapter:
            # Extract landmark messages from the live tail
            from optimizer.chapter_compressor import _is_landmark
            landmark_messages = [m for m in live_tail if _is_landmark(m)]

            store.close_chapter(
                conversation_id=conversation_id,
                compressed_content=chapter.compressed_content,
                landmark_messages=landmark_messages,
                compression_cost_usd=cost,
            )
            return {
                "status": "chapter_created",
                "chapter_id": chapter.chapter_id,
                "messages_compressed": chapter.original_message_count,
                "landmarks_preserved": len(landmark_messages),
                "cost_usd": cost,
            }
        else:
            return {
                "status": "compression_failed",
                "live_tail_length": store.live_tail_length(conversation_id),
            }

    return {
        "status": "message_added",
        "live_tail_length": store.live_tail_length(conversation_id),
    }


def optimize_with_store(
    store: ConversationStore,
    conversation_id: str,
    query: str,
    optimizer: "ContextOptimizer",
    query_type=None,
) -> "OptimizedContext":
    """
    Query path — called when user asks a question.

    Retrieves frozen chapters (byte-stable, cache hits) plus
    the live tail (recent, scored per query).

    Why score only the live tail?
      Frozen chapters are already compressed — re-scoring them
      would defeat the purpose of freezing. The chapter summary
      captures the meaning. Landmarks are hydrated verbatim.
      Only the live tail needs query-aware scoring.
    """
    chapters, live_tail = store.get_context(conversation_id)

    # Reconstruct frozen chapter messages (byte-stable)
    chapter_messages = []
    for chapter in chapters:
        chapter_messages.extend(chapter.to_messages())

    if not live_tail and not chapter_messages:
        return optimizer._empty_result()

    if not live_tail:
        # Only frozen chapters — no live tail to score
        # Assemble chapter messages directly
        from optimizer.types import QueryType
        from optimizer.scorer import WEIGHT_PROFILES
        qt = query_type or QueryType.GENERAL
        return optimizer.assembler.assemble(
            kept_messages=chapter_messages,
            summary_messages=[],
            original_token_count=sum(max(1, len(m.content) // 4) for m in chapter_messages),
            compression_cost_usd=0.0,
            query_type=qt,
            scoring_weights=WEIGHT_PROFILES[qt],
            assembly_latency_ms=0.0,
            stats={"kept_verbatim": len(chapter_messages), "compressed_groups": len(chapters),
                   "discarded": 0, "landmarks_preserved": 0},
        )

    # Score ONLY the live tail — this is the key difference from v1
    live_tail_result = optimizer.optimize(live_tail, query, query_type)

    # Combine: frozen chapters first, then scored live tail
    combined_messages = chapter_messages + live_tail_result.messages

    # Re-assemble combined context
    from optimizer.types import QueryType
    from optimizer.scorer import WEIGHT_PROFILES
    qt = live_tail_result.query_type
    original_tokens = sum(max(1, len(m.content) // 4) for m in chapter_messages) + live_tail_result.original_token_count

    return optimizer.assembler.assemble(
        kept_messages=combined_messages,
        summary_messages=[],
        original_token_count=original_tokens,
        compression_cost_usd=live_tail_result.compression_cost_usd,
        query_type=qt,
        scoring_weights=WEIGHT_PROFILES[qt],
        assembly_latency_ms=live_tail_result.assembly_latency_ms,
        stats={
            "kept_verbatim": len(chapter_messages) + live_tail_result.kept_verbatim,
            "compressed_groups": len(chapters) + live_tail_result.compressed_groups,
            "discarded": live_tail_result.discarded,
            "landmarks_preserved": live_tail_result.landmarks_preserved,
        },
    )
