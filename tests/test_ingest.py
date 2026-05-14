"""
Tests for ingest-time compression.

These tests verify the new ingest path without making real API calls.
The chapter compressor is mocked — we test the store logic and
the optimize_with_store() wiring separately from Haiku.
"""

import pytest
from unittest.mock import patch, MagicMock

from optimizer.types import Message, MessageRole, QueryType
from optimizer.chapter import Chapter
from optimizer.conversation_store import ConversationStore, CHAPTER_MAX_MESSAGES
from optimizer.optimizer import ContextOptimizer, ingest_message, optimize_with_store


def msg(i, role, content):
    return Message(index=i, role=MessageRole(role), content=content)


def make_messages(n, start=0):
    """Generate n alternating user/assistant messages."""
    messages = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(msg(start + i, role, f"Message number {start + i} with some content."))
    return messages


# ── ConversationStore tests ───────────────────────────────────────────────────

class TestConversationStore:

    def setup_method(self):
        self.store = ConversationStore()

    def test_add_message_increases_live_tail(self):
        m = msg(0, "user", "Hello")
        self.store.add_message("conv1", m)
        assert self.store.live_tail_length("conv1") == 1

    def test_get_context_returns_empty_for_unknown_conv(self):
        chapters, tail = self.store.get_context("unknown")
        assert chapters == []
        assert tail == []

    def test_should_not_close_chapter_below_threshold(self):
        for m in make_messages(10):
            self.store.add_message("conv1", m)
        assert self.store.should_close_chapter("conv1") is False

    def test_should_close_chapter_at_50_messages(self):
        for m in make_messages(CHAPTER_MAX_MESSAGES):
            self.store.add_message("conv1", m)
        assert self.store.should_close_chapter("conv1") is True

    def test_close_chapter_clears_live_tail(self):
        for m in make_messages(10):
            self.store.add_message("conv1", m)
        self.store.close_chapter(
            conversation_id="conv1",
            compressed_content="Summary of 10 messages.",
            landmark_messages=[],
            compression_cost_usd=0.001,
        )
        assert self.store.live_tail_length("conv1") == 0

    def test_close_chapter_stores_chapter(self):
        for m in make_messages(10):
            self.store.add_message("conv1", m)
        self.store.close_chapter(
            conversation_id="conv1",
            compressed_content="Summary of 10 messages.",
            landmark_messages=[],
        )
        assert self.store.chapter_count("conv1") == 1

    def test_get_context_returns_chapters_and_tail(self):
        # Add 10 messages and freeze as chapter
        for m in make_messages(10):
            self.store.add_message("conv1", m)
        self.store.close_chapter("conv1", "Summary.", [])

        # Add 5 more to live tail
        for m in make_messages(5, start=10):
            self.store.add_message("conv1", m)

        chapters, tail = self.store.get_context("conv1")
        assert len(chapters) == 1
        assert len(tail) == 5

    def test_close_chapter_raises_on_empty_tail(self):
        with pytest.raises(ValueError):
            self.store.close_chapter("conv1", "Summary.", [])

    def test_reset_clears_conversation(self):
        for m in make_messages(5):
            self.store.add_message("conv1", m)
        self.store.reset("conv1")
        chapters, tail = self.store.get_context("conv1")
        assert chapters == []
        assert tail == []

    def test_multiple_conversations_isolated(self):
        self.store.add_message("conv1", msg(0, "user", "Hello conv1"))
        self.store.add_message("conv2", msg(0, "user", "Hello conv2"))
        assert self.store.live_tail_length("conv1") == 1
        assert self.store.live_tail_length("conv2") == 1
        assert self.store.conversation_ids() == ["conv1", "conv2"]


# ── Chapter tests ─────────────────────────────────────────────────────────────

class TestChapter:

    def test_create_sets_content_hash(self):
        chapter = Chapter.create(
            compressed_content="Summary of discussion.",
            landmark_messages=[],
            original_message_count=10,
            start_index=0,
            end_index=9,
        )
        assert chapter.content_hash is not None
        assert len(chapter.content_hash) == 64  # SHA256 hex

    def test_same_content_produces_same_hash(self):
        c1 = Chapter.create("Same content.", [], 5, 0, 4)
        c2 = Chapter.create("Same content.", [], 5, 0, 4)
        assert c1.content_hash == c2.content_hash

    def test_different_content_produces_different_hash(self):
        c1 = Chapter.create("Content A.", [], 5, 0, 4)
        c2 = Chapter.create("Content B.", [], 5, 0, 4)
        assert c1.content_hash != c2.content_hash

    def test_to_messages_includes_summary(self):
        chapter = Chapter.create("Summary here.", [], 5, 0, 4)
        messages = chapter.to_messages()
        assert len(messages) == 1
        assert "CHAPTER SUMMARY" in messages[0].content
        assert "Summary here." in messages[0].content

    def test_to_messages_includes_landmarks_verbatim(self):
        landmark = msg(2, "user", "We agreed on £40,000 budget.")
        chapter = Chapter.create("Summary.", [landmark], 5, 0, 4)
        messages = chapter.to_messages()
        # Should have summary + landmark
        assert len(messages) == 2
        contents = [m.content for m in messages]
        assert any("£40,000" in c for c in contents)

    def test_to_messages_sorted_by_index(self):
        lm1 = msg(1, "user", "We agreed on the deadline.")
        lm2 = msg(3, "assistant", "Confirmed. Budget approved.")
        chapter = Chapter.create("Summary.", [lm2, lm1], 5, 0, 4)
        messages = chapter.to_messages()
        indices = [m.index for m in messages]
        assert indices == sorted(indices)

    def test_chapter_id_format(self):
        chapter = Chapter.create("Summary.", [], 5, 0, 4)
        assert chapter.chapter_id.startswith("ch_0_4_")


# ── ingest_message() tests ────────────────────────────────────────────────────

class TestIngestMessage:

    def setup_method(self):
        self.store = ConversationStore()

    def test_returns_message_added_status(self):
        result = ingest_message(self.store, "conv1", msg(0, "user", "Hello"))
        assert result["status"] == "message_added"
        assert result["live_tail_length"] == 1

    @patch("optimizer.optimizer.compress_to_chapter")
    def test_triggers_chapter_creation_at_boundary(self, mock_compress):
        # Mock successful compression
        mock_chapter = Chapter.create("Summary.", [], 50, 0, 49)
        mock_compress.return_value = (mock_chapter, 0.001)

        # Add 50 messages to hit boundary
        for m in make_messages(CHAPTER_MAX_MESSAGES):
            result = ingest_message(self.store, "conv1", m)

        assert result["status"] == "chapter_created"
        assert result["messages_compressed"] == CHAPTER_MAX_MESSAGES

    @patch("optimizer.optimizer.compress_to_chapter")
    def test_compression_failure_returns_graceful_status(self, mock_compress):
        mock_compress.return_value = (None, 0.0)

        for m in make_messages(CHAPTER_MAX_MESSAGES):
            result = ingest_message(self.store, "conv1", m)

        assert result["status"] == "compression_failed"


# ── optimize_with_store() tests ───────────────────────────────────────────────

class TestOptimizeWithStore:

    def setup_method(self):
        self.store = ConversationStore()
        self.optimizer = ContextOptimizer()

    def test_empty_store_returns_empty_result(self):
        result = optimize_with_store(
            self.store, "conv1", "What did we decide?", self.optimizer
        )
        assert result.messages == []

    def test_live_tail_only_returns_optimized_result(self):
        # Add messages to live tail without hitting boundary
        for m in make_messages(10):
            self.store.add_message("conv1", m)

        result = optimize_with_store(
            self.store, "conv1", "What did we decide?", self.optimizer
        )
        assert result.messages is not None
        assert result.original_token_count > 0

    def test_frozen_chapters_included_in_output(self):
        # Create a chapter manually
        for m in make_messages(10):
            self.store.add_message("conv1", m)
        self.store.close_chapter("conv1", "Summary of first 10 messages.", [])

        # Add live tail
        for m in make_messages(5, start=10):
            self.store.add_message("conv1", m)

        result = optimize_with_store(
            self.store, "conv1", "What is the status?", self.optimizer
        )
        # Output should contain chapter summary
        contents = [m.content for m in result.messages]
        assert any("CHAPTER SUMMARY" in c for c in contents)

    def test_landmark_in_chapter_preserved_in_output(self):
        landmark = msg(5, "user", "We agreed on £40,000 budget.")
        for m in make_messages(10):
            self.store.add_message("conv1", m)
        self.store.close_chapter("conv1", "Summary.", [landmark])

        for m in make_messages(5, start=10):
            self.store.add_message("conv1", m)

        result = optimize_with_store(
            self.store, "conv1", "What was the budget?", self.optimizer
        )
        contents = [m.content for m in result.messages]
        assert any("£40,000" in c for c in contents)
