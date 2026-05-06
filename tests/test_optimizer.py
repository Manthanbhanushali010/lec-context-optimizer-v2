"""Tests for LEC Context Optimizer — run with: pytest tests/ -v"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from unittest.mock import MagicMock
from optimizer.types import Message, MessageRole, MessageClass, QueryType
from optimizer.scorer import RelevanceScorer
from optimizer.assembler import Assembler
from optimizer.optimizer import ContextOptimizer


def _make_mock_embedder():
    mock = MagicMock()
    def fake_encode(texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            v = np.zeros(384)
            v[len(t) % 384] = 1.0
            v[0] = 0.3
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs)
    mock.encode = fake_encode
    return mock


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    monkeypatch.setattr("optimizer.scorer.get_embedder", _make_mock_embedder)


def msg(i, role, content):
    return Message(index=i, role=MessageRole(role), content=content)


SAMPLE_CONV = [
    msg(0, "user", "What database should we use?"),
    msg(1, "assistant", "What are the requirements?"),
    msg(2, "user", "We need strong consistency and JSON support."),
    msg(3, "assistant", "PostgreSQL with JSONB is a good fit."),
    msg(4, "user", "We decided to go with PostgreSQL for the new service."),
    msg(5, "assistant", "Great choice. I will handle the schema design."),
    msg(6, "user", "Sure, sounds good."),
    msg(7, "assistant", "The deadline for the schema is Thursday."),
    msg(8, "user", "Perfect."),
    msg(9, "assistant", "I will also set up PgBouncer for connection pooling."),
]


class TestQueryTypeDetection:
    def setup_method(self):
        self.scorer = RelevanceScorer()

    def test_factual_detection(self):
        qt = self.scorer.detect_query_type("What did we decide about the database?")
        assert qt == QueryType.FACTUAL

    def test_analytical_detection(self):
        qt = self.scorer.detect_query_type("Why did we choose this approach and analyse the trade-offs?")
        assert qt == QueryType.ANALYTICAL

    def test_status_detection(self):
        qt = self.scorer.detect_query_type("What is the current status of the project?")
        assert qt == QueryType.STATUS

    def test_general_fallback(self):
        qt = self.scorer.detect_query_type("Tell me something.")
        assert qt == QueryType.GENERAL


class TestRelevanceScorer:
    def setup_method(self):
        self.scorer = RelevanceScorer()

    def test_scores_all_messages(self):
        scored = self.scorer.score_all(SAMPLE_CONV, "What database did we choose?")
        assert len(scored) == len(SAMPLE_CONV)

    def test_scores_in_range(self):
        scored = self.scorer.score_all(SAMPLE_CONV, "What database did we choose?")
        for sm in scored:
            assert 0.0 <= sm.composite_score <= 1.0
            assert 0.0 <= sm.keyword_score <= 1.0
            assert 0.0 <= sm.recency_score <= 1.0
            assert 0.0 <= sm.landmark_score <= 1.0

    def test_landmark_detected(self):
        scored = self.scorer.score_all(SAMPLE_CONV, "What was decided?")
        decision_msg = scored[4]  # "We decided to go with PostgreSQL"
        assert decision_msg.landmark_score > 0.5
        assert decision_msg.classification == MessageClass.LANDMARK

    def test_recency_decay(self):
        scored = self.scorer.score_all(SAMPLE_CONV, "anything at all here")
        assert scored[-1].recency_score > scored[0].recency_score

    def test_deadline_detected_as_landmark(self):
        scored = self.scorer.score_all(SAMPLE_CONV, "when is the deadline?")
        deadline_msg = scored[7]  # "The deadline for the schema is Thursday"
        assert deadline_msg.landmark_score > 0.0

    def test_adaptive_weights_factual(self):
        from optimizer.scorer import WEIGHT_PROFILES
        w = WEIGHT_PROFILES[QueryType.FACTUAL]
        assert w["landmark"] > w["recency"]
        assert w["keyword"] >= w["semantic"]

    def test_adaptive_weights_analytical(self):
        from optimizer.scorer import WEIGHT_PROFILES
        w = WEIGHT_PROFILES[QueryType.ANALYTICAL]
        assert w["semantic"] > w["keyword"]

    def test_adaptive_weights_status(self):
        from optimizer.scorer import WEIGHT_PROFILES
        w = WEIGHT_PROFILES[QueryType.STATUS]
        assert w["recency"] > w["keyword"]

    def test_empty_conversation(self):
        scored = self.scorer.score_all([], "any query")
        assert scored == []

    def test_weights_sum_to_one(self):
        from optimizer.scorer import WEIGHT_PROFILES
        for qt, w in WEIGHT_PROFILES.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 0.001, f"{qt} weights sum to {total}"


class TestAssembler:
    def setup_method(self):
        self.assembler = Assembler()

    def test_orphaned_assistant_gets_bridge(self):
        messages = [
            msg(0, "assistant", "Here is some info."),
            msg(1, "user", "Thanks."),
        ]
        result = self.assembler.assemble(
            kept_messages=messages, summary_messages=[],
            original_token_count=100, compression_cost_usd=0.0,
            query_type=QueryType.GENERAL, scoring_weights={},
            assembly_latency_ms=10.0, stats={},
        )
        assert result.messages[0].role in (MessageRole.USER, MessageRole.SYSTEM)

    def test_to_anthropic_messages_format(self):
        result = self.assembler.assemble(
            kept_messages=SAMPLE_CONV[:5], summary_messages=[],
            original_token_count=200, compression_cost_usd=0.0,
            query_type=QueryType.GENERAL, scoring_weights={},
            assembly_latency_ms=5.0, stats={},
        )
        thread = self.assembler.to_anthropic_messages(result)
        assert isinstance(thread, list)
        for m in thread:
            assert "role" in m
            assert "content" in m
            assert m["role"] in ("user", "assistant")

    def test_token_reduction_calculated(self):
        result = self.assembler.assemble(
            kept_messages=SAMPLE_CONV[:3], summary_messages=[],
            original_token_count=500, compression_cost_usd=0.0,
            query_type=QueryType.GENERAL, scoring_weights={},
            assembly_latency_ms=5.0, stats={},
        )
        assert result.token_reduction_pct > 0

    def test_empty_input(self):
        result = self.assembler.assemble(
            kept_messages=[], summary_messages=[],
            original_token_count=0, compression_cost_usd=0.0,
            query_type=QueryType.GENERAL, scoring_weights={},
            assembly_latency_ms=1.0, stats={},
        )
        assert result.messages == []

    def test_orphan_tool_use_dropped(self):
        """tool_use with no matching tool_result should be dropped."""
        messages = [
            Message(index=0, role=MessageRole.USER, content="Run the search tool"),
            Message(index=1, role=MessageRole.TOOL_USE, content="search query", tool_call_id="call_001"),
            Message(index=2, role=MessageRole.ASSISTANT, content="Let me check that for you"),
        ]
        result = self.assembler._validate_and_repair(messages)
        roles = [m.role for m in result]
        assert MessageRole.TOOL_USE not in roles, "Orphan tool_use should be dropped"


class TestOptimizer:
    def setup_method(self):
        self.optimizer = ContextOptimizer()

    def test_returns_optimized_context(self):
        result = self.optimizer.optimize(SAMPLE_CONV, "What database did we pick?")
        assert result.messages is not None
        assert result.original_token_count > 0

    def test_query_type_detected_factual(self):
        result = self.optimizer.optimize(SAMPLE_CONV, "What did we decide about the database?")
        assert result.query_type == QueryType.FACTUAL

    def test_query_type_detected_status(self):
        result = self.optimizer.optimize(SAMPLE_CONV, "What is the current status?")
        assert result.query_type == QueryType.STATUS

    def test_empty_conversation(self):
        result = self.optimizer.optimize([], "any query")
        assert result.messages == []
        assert result.original_token_count == 0

    def test_optimized_smaller_than_original(self):
        long_conv = []
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            long_conv.append(msg(i, role, f"Message number {i} with some content here."))
        result = self.optimizer.optimize(long_conv, "What did we decide?")
        assert result.optimized_token_count <= result.original_token_count

    def test_scoring_weights_in_result(self):
        result = self.optimizer.optimize(SAMPLE_CONV, "What database did we pick?")
        assert isinstance(result.scoring_weights, dict)
        assert len(result.scoring_weights) == 4