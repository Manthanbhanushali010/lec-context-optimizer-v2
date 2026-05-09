"""
Shared data types for the context optimizer.
Every component speaks these types — nothing else crosses module boundaries.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class MessageClass(str, Enum):
    """How the optimizer classifies each message."""
    LANDMARK = "landmark"       # Decision, commitment, deadline — always keep verbatim
    RELEVANT = "relevant"       # Related to query — keep verbatim
    COMPRESSIBLE = "compressible"  # Loosely related — summarise as group
    NOISE = "noise"             # Irrelevant — discard or one-liner


class QueryType(str, Enum):
    """Detected intent of the current query — drives adaptive scoring weights."""
    FACTUAL = "factual"         # "What did we decide about X?" — needs exact facts
    ANALYTICAL = "analytical"   # "What were the trade-offs?" — needs broad coverage
    STATUS = "status"           # "What's the current state of X?" — needs recent context
    GENERAL = "general"         # Fallback — balanced weights


@dataclass
class Message:
    """A single turn in a conversation."""
    index: int                          # Position in original conversation (0-based)
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None  # For tool_use / tool_result linking
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredMessage:
    """A message with its relevance breakdown."""
    message: Message
    keyword_score: float        # 0-1: exact term overlap with query
    semantic_score: float       # 0-1: embedding cosine similarity
    recency_score: float        # 0-1: exponential decay by position
    landmark_score: float       # 0-1: decision/commitment/deadline signal
    composite_score: float      # weighted combination
    classification: MessageClass
    landmark_reason: Optional[str] = None  # e.g. "contains decision marker: 'we agreed'"


@dataclass
class OptimizedContext:
    """The output of the full optimization pipeline."""
    messages: list[Message]             # Assembled, valid conversation thread
    original_token_count: int
    optimized_token_count: int
    token_reduction_pct: float
    kept_verbatim: int                  # count of messages kept as-is
    compressed_groups: int              # count of message groups summarised
    discarded: int                      # count of messages dropped
    landmarks_preserved: int
    assembly_latency_ms: float
    compression_cost_usd: float
    query_type: QueryType
    scoring_weights: dict               # which weights were used (for audit)


@dataclass
class EvalResult:
    """Result of one head-to-head comparison: full vs optimised context."""
    conversation_id: str
    query: str
    query_type: str
    full_answer: str
    optimized_answer: str
    full_score: float           # LLM judge score 0-9
    optimized_score: float      # LLM judge score 0-9
    optimized_wins: bool        # optimized_score >= full_score
    token_reduction_pct: float
    assembly_latency_ms: float
    compression_cost_usd: float
    full_context_cost_usd: float
    optimized_context_cost_usd: float
    net_saving_usd: float
def estimate_tokens(text: str) -> int:
    """Estimate token count — ~4 chars per token. Replace with tiktoken for production accuracy."""
    return max(1, len(text) // 4)