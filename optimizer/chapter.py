"""
Chapter
=======
A frozen, compressed span of conversation history.

Once created, a Chapter never changes. This byte-stability is what
allows Anthropic's prefix cache to work — frozen chapters produce
identical prompt prefixes on every request.
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import time

from optimizer.types import Message


@dataclass
class Chapter:
    """A frozen span of compressed conversation history."""

    chapter_id: str
    content_hash: str
    compressed_content: str
    landmark_messages: list[Message]
    original_message_count: int
    start_index: int
    end_index: int
    created_at: float = field(default_factory=time.time)
    compression_cost_usd: float = 0.0

    @classmethod
    def create(
        cls,
        compressed_content: str,
        landmark_messages: list[Message],
        original_message_count: int,
        start_index: int,
        end_index: int,
        compression_cost_usd: float = 0.0,
    ) -> "Chapter":
        content_hash = hashlib.sha256(compressed_content.encode()).hexdigest()
        chapter_id = f"ch_{start_index}_{end_index}_{content_hash[:8]}"
        return cls(
            chapter_id=chapter_id,
            content_hash=content_hash,
            compressed_content=compressed_content,
            landmark_messages=landmark_messages,
            original_message_count=original_message_count,
            start_index=start_index,
            end_index=end_index,
            compression_cost_usd=compression_cost_usd,
        )

    def to_messages(self) -> list[Message]:
        from optimizer.types import MessageRole
        result = []
        summary_msg = Message(
            index=self.start_index,
            role=MessageRole.SYSTEM,
            content=(
                f"[CHAPTER SUMMARY — {self.original_message_count} messages compressed, "
                f"indices {self.start_index}-{self.end_index}]: {self.compressed_content}"
            ),
            metadata={
                "chapter_id": self.chapter_id,
                "content_hash": self.content_hash,
                "is_frozen": True,
            },
        )
        result.append(summary_msg)
        for lm in self.landmark_messages:
            result.append(lm)
        result.sort(key=lambda m: m.index)
        return result
