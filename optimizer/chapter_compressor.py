"""
ChapterCompressor
=================
Compresses a live tail of messages into a Chapter using Claude Haiku.

Why a separate module and not reusing Compressor?
  The existing Compressor works on ScoredMessages and produces summary
  messages for the assembler. ChapterCompressor works on raw Messages
  and produces a Chapter — a frozen, permanently stored artifact.
  Different input type, different output type, different purpose.
  Bundling them would mean one failure mode corrupts both paths.

Why Haiku and not Sonnet?
  Compression is a mechanical task — summarise these messages in 1-2
  sentences, preserve commitments exactly. Haiku is 4x cheaper and
  fast enough. Using Sonnet would eliminate most of the savings we
  are trying to create.

Landmark extraction:
  Before compressing, we scan the live tail for landmark messages
  using the same scorer patterns. Landmarks are extracted verbatim
  and stored in Chapter.landmark_messages — they are never passed
  to Haiku for summarisation. This is belt-and-suspenders: the scorer
  already classified them, but we verify again here because the cost
  of losing a commitment is higher than the cost of a redundant check.
"""

import logging
import re

import anthropic

from optimizer.chapter import Chapter
from optimizer.types import Message, MessageClass, MessageRole

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()

_HAIKU_INPUT_PRICE  = 0.80 / 1_000_000
_HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000

# Landmark patterns — same as scorer.py
# Duplicated here intentionally: chapter creation must not depend
# on the scorer being initialised (no embedding model needed)
_LANDMARK_PATTERNS = [
    r"\bwe (decided|agreed|confirmed|resolved|concluded|settled on)\b",
    r"\bfinal (answer|decision|call|choice)\b",
    r"\bapproved\b",
    r"\bsigned off\b",
    r"\bdeadline\b",
    r"\bdue date\b",
    r"\bcommit(ted|ment)?\b",
    r"\bbudget.{0,20}(approved|allocated|set).{0,20}[\d,]+",
    r"\b[£$€]\s*[\d,]+",
    r"\baction item\b",
    r"\bnext step\b",
]


def _is_landmark(message: Message) -> bool:
    """Quick landmark check — no embedding needed."""
    content_lower = message.content.lower()
    return any(re.search(p, content_lower) for p in _LANDMARK_PATTERNS)


def _format_for_haiku(messages: list[Message]) -> str:
    """Format messages for the Haiku summarisation prompt."""
    lines = []
    for m in messages:
        role = m.role.value.upper()
        content = m.content[:400]  # Truncate very long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def compress_to_chapter(messages: list[Message]) -> tuple[Chapter | None, float]:
    """
    Compress a list of messages into a frozen Chapter.

    Steps:
      1. Extract landmark messages verbatim (never summarised)
      2. Send non-landmark messages to Haiku for summarisation
      3. Create and return a frozen Chapter

    Returns:
      (Chapter, cost_usd) on success
      (None, 0.0) on failure — caller keeps originals

    Why return None on failure?
      Same pattern as Compressor._summarise_group().
      A failure here means the chapter boundary is missed this cycle.
      The live tail continues growing. Next boundary trigger retries.
      Never crash — always degrade gracefully.
    """
    if not messages:
        return None, 0.0

    # Step 1: Extract landmarks verbatim
    landmark_messages = [m for m in messages if _is_landmark(m)]
    non_landmark = [m for m in messages if not _is_landmark(m)]

    logger.info(
        "Compressing %d messages into chapter: %d landmarks, %d to summarise",
        len(messages), len(landmark_messages), len(non_landmark),
    )

    # Step 2: Summarise non-landmark messages via Haiku
    if non_landmark:
        formatted = _format_for_haiku(non_landmark)
        prompt = f"""Summarise this conversation segment in 2-3 concise sentences.
CRITICAL: Preserve any decisions, commitments, deadlines, budgets, or action items EXACTLY.
Focus on what was discussed and decided. Omit pleasantries and filler.
Output only the summary — no preamble, no labels.

Segment:
{formatted}"""

        try:
            response = _client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            compressed_content = response.content[0].text.strip()
            cost = (
                response.usage.input_tokens  * _HAIKU_INPUT_PRICE +
                response.usage.output_tokens * _HAIKU_OUTPUT_PRICE
            )
            logger.debug("Haiku summary: %s", compressed_content[:100])

        except Exception as exc:
            logger.error("Chapter compression failed: %s", exc)
            return None, 0.0
    else:
        # All messages were landmarks — no summarisation needed
        # Create a minimal summary noting only landmarks survived
        compressed_content = (
            f"[All {len(messages)} messages in this segment were landmark decisions "
            f"or commitments — preserved verbatim above.]"
        )
        cost = 0.0

    # Step 3: Create frozen Chapter
    chapter = Chapter.create(
        compressed_content=compressed_content,
        landmark_messages=landmark_messages,
        original_message_count=len(messages),
        start_index=messages[0].index,
        end_index=messages[-1].index,
        compression_cost_usd=cost,
    )

    logger.info(
        "Chapter %s created: %d msgs → summary + %d landmarks (cost: $%.6f)",
        chapter.chapter_id, len(messages), len(landmark_messages), cost,
    )
    return chapter, cost
