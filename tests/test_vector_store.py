
import pytest
import uuid
from optimizer.vector_store import VectorStore
from optimizer.chapter import Chapter
from optimizer.types import Message, MessageRole


def make_chapter(content, start, end, landmarks=None):
    return Chapter.create(
        compressed_content=content,
        landmark_messages=landmarks or [],
        original_message_count=end - start + 1,
        start_index=start,
        end_index=end,
    )


def landmark_msg(i, content):
    return Message(index=i, role=MessageRole.USER, content=content)


class TestVectorStore:

    def setup_method(self):
        self.store = VectorStore()
        self.store.ensure_collection()
        self.conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"

    def teardown_method(self):
        self.store.delete_conversation(self.conv_id)

    def test_store_and_retrieve_chapter(self):
        chapter = make_chapter("We discussed database options and chose PostgreSQL.", 0, 9)
        self.store.store_chapter(self.conv_id, chapter)
        results = self.store.retrieve_relevant_chapters(
            self.conv_id, "What database did we choose?", top_k=1
        )
        assert len(results) == 1
        assert results[0].chapter_id == chapter.chapter_id

    def test_retrieve_returns_most_relevant_first(self):
        c1 = make_chapter("Team discussed project timeline and deadlines.", 0, 9)
        c2 = make_chapter("We agreed on PostgreSQL for the database layer.", 10, 19)
        c3 = make_chapter("Budget approved at 42000 GBP for the project.", 20, 29)
        self.store.store_chapter(self.conv_id, c1)
        self.store.store_chapter(self.conv_id, c2)
        self.store.store_chapter(self.conv_id, c3)
        results = self.store.retrieve_relevant_chapters(
            self.conv_id, "What database technology did we select?", top_k=1
        )
        assert len(results) == 1
        assert results[0].chapter_id == c2.chapter_id

    def test_landmark_messages_preserved_through_storage(self):
        landmark = landmark_msg(5, "We agreed on 42000 GBP budget.")
        chapter = make_chapter("Budget discussion.", 0, 9, landmarks=[landmark])
        self.store.store_chapter(self.conv_id, chapter)
        results = self.store.retrieve_relevant_chapters(
            self.conv_id, "What was the budget?", top_k=1
        )
        assert len(results) == 1
        assert len(results[0].landmark_messages) == 1
        assert "42000" in results[0].landmark_messages[0].content

    def test_get_all_chapters_returns_chronological(self):
        c1 = make_chapter("First segment.", 0, 9)
        c2 = make_chapter("Second segment.", 10, 19)
        c3 = make_chapter("Third segment.", 20, 29)
        self.store.store_chapter(self.conv_id, c1)
        self.store.store_chapter(self.conv_id, c3)
        self.store.store_chapter(self.conv_id, c2)
        all_chapters = self.store.get_all_chapters(self.conv_id)
        assert len(all_chapters) == 3
        indices = [c.start_index for c in all_chapters]
        assert indices == sorted(indices)

    def test_chapter_count(self):
        assert self.store.chapter_count(self.conv_id) == 0
        self.store.store_chapter(self.conv_id, make_chapter("Summary.", 0, 9))
        assert self.store.chapter_count(self.conv_id) == 1

    def test_content_hash_stable_after_roundtrip(self):
        chapter = make_chapter("Stable content for cache testing.", 0, 9)
        original_hash = chapter.content_hash
        self.store.store_chapter(self.conv_id, chapter)
        results = self.store.get_all_chapters(self.conv_id)
        assert len(results) == 1
        assert results[0].content_hash == original_hash

    def test_no_cross_conversation_leakage(self):
        other_conv = f"other_{uuid.uuid4().hex[:8]}"
        try:
            self.store.store_chapter(
                self.conv_id,
                make_chapter("Conv1 database PostgreSQL discussion.", 0, 9)
            )
            self.store.store_chapter(
                other_conv,
                make_chapter("Conv2 cooking recipes discussion.", 0, 9)
            )
            results = self.store.retrieve_relevant_chapters(
                self.conv_id, "database", top_k=5
            )
            other_results = self.store.retrieve_relevant_chapters(
                other_conv, "database", top_k=5
            )
            ids = [r.chapter_id for r in results]
            other_ids = [r.chapter_id for r in other_results]
            assert not set(ids) & set(other_ids)
        finally:
            self.store.delete_conversation(other_conv)

    def test_delete_conversation_removes_all_chapters(self):
        self.store.store_chapter(self.conv_id, make_chapter("Summary 1.", 0, 9))
        self.store.store_chapter(self.conv_id, make_chapter("Summary 2.", 10, 19))
        assert self.store.chapter_count(self.conv_id) == 2
        self.store.delete_conversation(self.conv_id)
        assert self.store.chapter_count(self.conv_id) == 0
