"""
ConversationStore
=================
In-memory store for chapters and live tail per conversation.

This is the persistence layer that makes ingest-time compression work.
It sits above the optimizer and manages two things per conversation:
  1. Frozen chapters — compressed, never change, cache-stable
  2. Live tail — recent messages not yet compressed into a chapter

Why in-memory for v2?
  Production would use Redis or Postgres. In-memory lets us prove
  the architecture works before adding infrastructure complexity.
  The interface is the same — swap the backend later.

Chapter boundary rules:
  A chapter closes when the live tail hits either:
  - 50 messages (count-based hygiene), OR
  - 4,000 tokens (economics — compression pays off at this threshold)
  Whichever comes first.
"""

import logging
from dataclasses import dataclass, field

from optimizer.chapter import Chapter
from optimizer.types import Message

logger = logging.getLogger(__name__)

# Chapter boundary thresholds
CHAPTER_MAX_MESSAGES = 50
CHAPTER_MAX_TOKENS   = 4_000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — 4 chars per token."""
    return max(1, len(text) // 4)


@dataclass
class ConversationState:
    """Internal state for one conversation."""
    conversation_id: str
    chapters: list[Chapter] = field(default_factory=list)
    live_tail: list[Message] = field(default_factory=list)

    def live_tail_tokens(self) -> int:
        return sum(_estimate_tokens(m.content) for m in self.live_tail)

    def should_close_chapter(self) -> bool:
        """True when live tail hits either boundary threshold."""
        if len(self.live_tail) >= CHAPTER_MAX_MESSAGES:
            logger.debug(
                "Chapter boundary hit: %d messages in live tail (limit %d)",
                len(self.live_tail), CHAPTER_MAX_MESSAGES,
            )
            return True
        tokens = self.live_tail_tokens()
        if tokens >= CHAPTER_MAX_TOKENS:
            logger.debug(
                "Chapter boundary hit: %d tokens in live tail (limit %d)",
                tokens, CHAPTER_MAX_TOKENS,
            )
            return True
        return False


class ConversationStore:
    """
    In-memory store for all active conversations.

    Usage:
        store = ConversationStore()

        # Ingest path — called when a message arrives
        store.add_message(conv_id, message)
        if store.should_close_chapter(conv_id):
            chapter = await store.close_chapter(conv_id, compressor)
            # chapter is now frozen and cache-stable

        # Query path — called when user asks a question
        chapters, live_tail = store.get_context(conv_id)
        # chapters are byte-stable (cache hits)
        # live_tail is scored per query
    """

    def __init__(self):
        self._conversations: dict[str, ConversationState] = {}

    # ── Ingest path ───────────────────────────────────────────────

    def add_message(self, conversation_id: str, message: Message) -> None:
        """
        Add a message to the live tail of a conversation.
        Called every time a new turn arrives — before checking boundaries.
        """
        state = self._get_or_create(conversation_id)
        state.live_tail.append(message)
        logger.debug(
            "Added message %d to conv %s (live tail: %d messages, ~%d tokens)",
            message.index, conversation_id,
            len(state.live_tail), state.live_tail_tokens(),
        )

    def should_close_chapter(self, conversation_id: str) -> bool:
        """Check whether the live tail has hit a chapter boundary."""
        state = self._conversations.get(conversation_id)
        if not state:
            return False
        return state.should_close_chapter()

    def close_chapter(
        self,
        conversation_id: str,
        compressed_content: str,
        landmark_messages: list[Message],
        compression_cost_usd: float = 0.0,
    ) -> Chapter:
        """
        Freeze the current live tail into a Chapter.

        The caller is responsible for running compression (Haiku call)
        and passing the result here. This keeps the store pure —
        no API calls, just state management.

        After this call:
        - The live tail is cleared
        - The chapter is frozen and appended to the chapter list
        - The chapter's content_hash guarantees byte-stability
        """
        state = self._get_or_create(conversation_id)

        if not state.live_tail:
            raise ValueError(f"Cannot close chapter: live tail is empty for {conversation_id}")

        start_index = state.live_tail[0].index
        end_index   = state.live_tail[-1].index
        original_count = len(state.live_tail)

        chapter = Chapter.create(
            compressed_content=compressed_content,
            landmark_messages=landmark_messages,
            original_message_count=original_count,
            start_index=start_index,
            end_index=end_index,
            compression_cost_usd=compression_cost_usd,
        )

        state.chapters.append(chapter)
        state.live_tail = []  # Clear — messages now frozen in chapter

        logger.info(
            "Chapter %s created for conv %s: %d messages → 1 summary + %d landmarks",
            chapter.chapter_id, conversation_id,
            original_count, len(landmark_messages),
        )
        return chapter

    # ── Query path ────────────────────────────────────────────────

    def get_context(
        self, conversation_id: str
    ) -> tuple[list[Chapter], list[Message]]:
        """
        Return (frozen_chapters, live_tail) for a conversation.

        frozen_chapters — byte-stable, cache hits, never re-scored
        live_tail       — recent messages, scored per query

        If conversation doesn't exist yet, returns ([], []).
        """
        state = self._conversations.get(conversation_id)
        if not state:
            return [], []
        return list(state.chapters), list(state.live_tail)

    # ── Inspection ────────────────────────────────────────────────

    def chapter_count(self, conversation_id: str) -> int:
        state = self._conversations.get(conversation_id)
        return len(state.chapters) if state else 0

    def live_tail_length(self, conversation_id: str) -> int:
        state = self._conversations.get(conversation_id)
        return len(state.live_tail) if state else 0

    def conversation_ids(self) -> list[str]:
        return list(self._conversations.keys())

    def reset(self, conversation_id: str) -> None:
        """Clear all state for a conversation. Useful for testing."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    # ── Private ───────────────────────────────────────────────────

    def _get_or_create(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id
            )
        return self._conversations[conversation_id]
