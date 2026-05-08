"""
Assembler
=========
Takes kept messages + summary messages from the Compressor and
assembles a valid conversation thread that an LLM can consume.

Invariants enforced:
  1. Conversation starts with a user or system message
  2. No consecutive assistant messages without a user message between them
  3. Every tool_result has a preceding tool_use
  4. Summaries inserted in chronological position
  5. No orphaned assistant turns

If violations are detected, the assembler repairs them by:
  - Inserting bridging context markers
  - Removing truly orphaned turns
  - Never silently reordering (that would change meaning)
"""
import logging
logger = logging.getLogger(__name__)
from optimizer.types import Message, MessageRole, OptimizedContext, QueryType


class Assembler:
    """Assembles and validates the final optimized context."""

    def assemble(
        self,
        kept_messages: list[Message],
        summary_messages: list[Message],
        original_token_count: int,
        compression_cost_usd: float,
        query_type: QueryType,
        scoring_weights: dict,
        assembly_latency_ms: float,
        stats: dict,
    ) -> OptimizedContext:
        """
        Merge kept + summary messages, validate, and return OptimizedContext.
        """
        # 1. Merge by original index (chronological order)
        all_messages = kept_messages + summary_messages
        all_messages.sort(key=lambda m: m.index)

        # 2. Validate and repair
        valid_messages = self._validate_and_repair(all_messages)

        # 3. Count tokens
        optimized_tokens = sum(self._estimate_tokens(m.content) for m in valid_messages)
        reduction_pct = (
            round((1 - optimized_tokens / original_token_count) * 100, 1)
            if original_token_count > 0 else 0.0
        )

        return OptimizedContext(
            messages=valid_messages,
            original_token_count=original_token_count,
            optimized_token_count=optimized_tokens,
            token_reduction_pct=reduction_pct,
            kept_verbatim=stats.get("kept_verbatim", 0),
            compressed_groups=stats.get("compressed_groups", 0),
            discarded=stats.get("discarded", 0),
            landmarks_preserved=stats.get("landmarks_preserved", 0),
            assembly_latency_ms=assembly_latency_ms,
            compression_cost_usd=compression_cost_usd,
            query_type=query_type,
            scoring_weights=scoring_weights,
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_and_repair(self, messages: list[Message]) -> list[Message]:
        """
        Walk the message list and enforce conversation invariants.
        Returns a repaired list.
        """
        if not messages:
            return []

        repaired = []
        # First pass: build set of tool_call_ids that have matching tool_results.
        # Used in the second pass to detect orphan tool_use messages whose
        # tool_results were dropped during compression.
        responded_tool_ids = {
            m.tool_call_id
            for m in messages
            if m.role == MessageRole.TOOL_RESULT and m.tool_call_id is not None
        }

        # Ensure conversation starts with user or system message
        first_content = messages[0]
        if first_content.role == MessageRole.ASSISTANT:
            # Insert a bridging marker before the first assistant turn
            repaired.append(Message(
                index=-1,
                role=MessageRole.SYSTEM,
                content="[CONTEXT NOTE: Earlier conversation context has been compressed or omitted]",
            ))

        for i, msg in enumerate(messages):
            # Rule: tool_result must follow tool_use
            if msg.role == MessageRole.TOOL_RESULT:
                if not repaired or repaired[-1].role != MessageRole.TOOL_USE:
                    # Orphaned tool_result — skip it, it would confuse the LLM
                    continue
            # Rule: tool_use must have a matching tool_result.
            # If the response was dropped during compression, skip the orphan
            # to avoid leaving a tool call with no answer (which causes the
            # downstream LLM to hallucinate the result).
            if msg.role == MessageRole.TOOL_USE:
                if msg.tool_call_id not in responded_tool_ids:
                    continue

            # Rule: no consecutive assistant messages
            if msg.role == MessageRole.ASSISTANT and repaired:
                prev_role = repaired[-1].role
                if prev_role == MessageRole.ASSISTANT:
                    # Insert a bridging user turn
                    repaired.append(Message(
                        index=msg.index - 1,
                        role=MessageRole.USER,
                        content="[continued]",
                    ))

            repaired.append(msg)

        return repaired

    def _estimate_tokens(self, text: str) -> int:
        """~4 chars per token estimate."""
        return max(1, len(text) // 4)

    def to_anthropic_messages(self, optimized: OptimizedContext) -> list[dict]:
        """
        Convert OptimizedContext to the list[dict] format Anthropic's API expects.
        System messages are folded into a preamble string.
        """
        result = []
        for msg in optimized.messages:
            if msg.role == MessageRole.SYSTEM:
                # Inject as a user turn with a clear marker
                result.append({
                    "role": "user",
                    "content": f"<context_note>{msg.content}</context_note>",
                })
                # Add a minimal assistant acknowledgement to keep turn structure valid
                result.append({
                    "role": "assistant",
                    "content": "Understood.",
                })
            else:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        # Deduplicate consecutive same-role messages (safety net)
        deduped = []
        for m in result:
            if deduped and deduped[-1]["role"] == m["role"]:
                logger.warning("Assembler dedup fired: consecutive %s messages merged — check upstream invariant handling", m["role"])
                deduped[-1]["content"] += f"\n{m['content']}"
            else:
                deduped.append(m)

        return deduped
