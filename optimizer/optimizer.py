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
