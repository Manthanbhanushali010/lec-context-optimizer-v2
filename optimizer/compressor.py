"""
Compressor
==========
Takes scored messages and compresses them:
  - LANDMARK + RELEVANT → kept verbatim, untouched
  - COMPRESSIBLE groups → collapsed into a single summary message via LLM
  - NOISE → discarded (or folded into adjacent compressible group)

Groups consecutive compressible/noise messages together before summarising —
one LLM call per group, not per message.
"""

import os
import time
from dataclasses import dataclass

import anthropic

from optimizer.types import Message, MessageClass, MessageRole, ScoredMessage

_client = anthropic.Anthropic()

# Pricing for cost tracking (Haiku used for compression)
_HAIKU_INPUT_PRICE = 0.80 / 1_000_000
_HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000


@dataclass
class CompressionResult:
    """Output of the compression step."""
    kept_messages: list[Message]        # verbatim kept (landmark + relevant)
    summary_messages: list[Message]     # synthesised summaries replacing groups
    discarded_count: int
    compressed_groups: int
    total_cost_usd: float
    input_tokens_saved: int             # tokens in original compressible messages


class Compressor:
    """
    Compresses a scored conversation.
    Landmarks and relevant messages pass through untouched.
    Compressible/noise groups are summarised by Claude Haiku.
    """

    def __init__(self, min_group_size: int = 2):
        """
        min_group_size: minimum messages in a group before we bother summarising.
        Single noise messages are just discarded.
        """
        self.min_group_size = min_group_size

    def compress(self, scored_messages: list[ScoredMessage]) -> CompressionResult:
        """Main entry point. Returns compression result."""
        if not scored_messages:
            return CompressionResult([], [], 0, 0, 0.0, 0)

        # Separate into keep-as-is vs groups-to-compress
        groups = self._group_by_classification(scored_messages)

        kept: list[Message] = []
        summaries: list[Message] = []
        discarded_count = 0
        compressed_groups = 0
        total_cost = 0
        tokens_saved = 0

        for group_class, group_messages in groups:
            if group_class in (MessageClass.LANDMARK, MessageClass.RELEVANT):
                # Keep verbatim — these messages pass through unchanged
                kept.extend([sm.message for sm in group_messages])

            elif group_class == MessageClass.NOISE and len(group_messages) < self.min_group_size:
                # Single noise message — just discard
                discarded_count += len(group_messages)
                tokens_saved += sum(self._estimate_tokens(sm.message.content) for sm in group_messages)

            else:
                # Compressible group (or large noise group) → summarise
                msgs = [sm.message for sm in group_messages]
                original_tokens = sum(self._estimate_tokens(m.content) for m in msgs)

                summary, cost = self._summarise_group(msgs)

                if summary:
                    summaries.append(summary)
                    compressed_groups += 1
                    total_cost += cost
                    tokens_saved += original_tokens - self._estimate_tokens(summary.content)
                else:
                    # Summarisation failed — keep originals to be safe
                    kept.extend(msgs)

        return CompressionResult(
            kept_messages=kept,
            summary_messages=summaries,
            discarded_count=discarded_count,
            compressed_groups=compressed_groups,
            total_cost_usd=round(total_cost, 6),
            input_tokens_saved=max(0, tokens_saved),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _group_by_classification(
        self, scored: list[ScoredMessage]
    ) -> list[tuple[MessageClass, list[ScoredMessage]]]:
        """
        Group consecutive messages with the same broad class.
        LANDMARK and RELEVANT are kept as individual single-message groups.
        COMPRESSIBLE and NOISE are merged into runs.
        """
        groups: list[tuple[MessageClass, list[ScoredMessage]]] = []

        for sm in scored:
            cls = sm.classification

            # Landmarks and relevant are always their own group (never merged)
            if cls in (MessageClass.LANDMARK, MessageClass.RELEVANT):
                groups.append((cls, [sm]))
                continue

            # Compressible and noise are merged into runs
            compressible_class = MessageClass.COMPRESSIBLE  # treat noise same as compressible
            if groups and groups[-1][0] == compressible_class:
                groups[-1][1].append(sm)
            else:
                groups.append((compressible_class, [sm]))

        return groups

    def _summarise_group(self, messages: list[Message]) -> tuple[Message | None, float]:
        """
        Call Claude Haiku to summarise a group of low-value messages.
        Returns (summary Message, cost_usd).
        """
        if not messages:
            return None, 0.0

        # Format the group for the LLM
        formatted = "\n".join(
            f"{m.role.value.upper()}: {m.content[:500]}"
            for m in messages
        )

        prompt = f"""Summarise this conversation sub-thread in 1-2 concise sentences.
CRITICAL: Preserve any decisions, commitments, deadlines, or action items EXACTLY.
If there are no important facts, write a brief summary of the topic discussed.
Do NOT include filler or pleasantries. Output only the summary, no preamble.

Sub-thread:
{formatted}"""

        try:
            response = _client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            summary_text = response.content[0].text.strip()
            cost = (
                response.usage.input_tokens * _HAIKU_INPUT_PRICE
                + response.usage.output_tokens * _HAIKU_OUTPUT_PRICE
            )

            # Insert as a system-style marker so the LLM knows it's a summary
            summary_msg = Message(
                index=messages[0].index,  # position of first compressed message
                role=MessageRole.SYSTEM,
                content=f"[CONTEXT SUMMARY — {len(messages)} messages compressed]: {summary_text}",
                metadata={"compressed_from": len(messages), "original_indices": [m.index for m in messages]},
            )
            return summary_msg, cost

        except Exception as exc:
            # Graceful failure — return None so caller keeps originals
            print(f"Compression failed for group of {len(messages)}: {exc}")
            return None, 0.0

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)
