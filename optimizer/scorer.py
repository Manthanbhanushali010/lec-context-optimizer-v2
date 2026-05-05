"""
Relevance Scorer
================
Scores every message in a conversation against the current query.

Four signals:
  A. Keyword match    — exact term overlap (fast, handles terminology)
  B. Semantic sim     — embedding cosine similarity (handles synonyms)
  C. Recency decay    — exponential decay by position (recent = more relevant)
  D. Landmark detect  — decision/commitment/deadline patterns (always important)

Adaptive weights shift based on detected query type:
  FACTUAL     → boost landmark + keyword   (need exact facts)
  ANALYTICAL  → boost semantic             (need broad coverage)
  STATUS      → boost recency              (need current state)
  GENERAL     → balanced defaults
"""

import math
import re
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from optimizer.types import (
    Message, MessageClass, MessageRole, QueryType, ScoredMessage
)

# ── Embedding model (loaded once, reused across all calls) ───────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


# ── Landmark detection patterns ──────────────────────────────────────────────
_DECISION_PATTERNS = [
    r"\bwe (decided|agreed|confirmed|resolved|concluded|settled on)\b",
    r"\b(decision|agreement|conclusion|resolution)\b.{0,30}\b(is|was|will be)\b",
    r"\bgoing with\b",
    r"\bfinal (answer|decision|call|choice)\b",
    r"\bapproved\b",
    r"\bsigned off\b",
]

_COMMITMENT_PATTERNS = [
    r"\bdeadline\b",
    r"\bdue date\b",
    r"\b(will|shall|must|need to|have to)\b.{0,40}\bby\b",
    r"\bcommit(ted|ment)?\b",
    r"\bbudget.{0,20}(is|was|approved|set|allocated).{0,20}[£$€]\s*[\d,]+",
    r"\b[£$€]\s*[\d,]+.{0,20}(approved|allocated|budget|spend)",
    r"\baction item\b",
    r"\bTODO\b",
    r"\bnext step\b",
    r"\byou('ll| will) (handle|own|lead|take care of)\b",
]

_LANDMARK_PATTERNS = _DECISION_PATTERNS + _COMMITMENT_PATTERNS

# ── Query type detection ──────────────────────────────────────────────────────
_FACTUAL_SIGNALS = [
    r"\bwhat (did|was|is|were)\b",
    r"\bwhen (did|was|is)\b",
    r"\bwho (decided|agreed|said|confirmed)\b",
    r"\bwhich (option|choice|decision)\b",
    r"\bhow much\b",
    r"\bwhat.*decide\b",
]

_ANALYTICAL_SIGNALS = [
    r"\bwhy\b",
    r"\btrade.?off\b",
    r"\bpros? and cons?\b",
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\bexplain\b",
    r"\banalyse\b",
    r"\banalyze\b",
    r"\bcompare\b",
    r"\bwhat were the (reasons|considerations|factors)\b",
]

_STATUS_SIGNALS = [
    r"\bcurrent(ly)?\b",
    r"\blatest\b",
    r"\bright now\b",
    r"\bstatus\b",
    r"\bprogress\b",
    r"\bupdate\b",
    r"\bwhere (are|do) we\b",
    r"\bwhat('s| is) happening\b",
]


# ── Adaptive weight profiles ──────────────────────────────────────────────────
WEIGHT_PROFILES = {
    QueryType.FACTUAL: {
        "keyword": 0.30,
        "semantic": 0.15,
        "recency": 0.10,
        "landmark": 0.45,
    },
    QueryType.ANALYTICAL: {
        "keyword": 0.15,
        "semantic": 0.50,
        "recency": 0.15,
        "landmark": 0.20,
    },
    QueryType.STATUS: {
        "keyword": 0.20,
        "semantic": 0.25,
        "recency": 0.45,
        "landmark": 0.10,
    },
    QueryType.GENERAL: {
        "keyword": 0.20,
        "semantic": 0.30,
        "recency": 0.20,
        "landmark": 0.30,
    },
}

# ── Classification thresholds ─────────────────────────────────────────────────
THRESHOLDS = {
    "landmark_override": 0.75,   # landmark_score above this → always LANDMARK class
    "keep": 0.55,                # composite above this → RELEVANT (keep verbatim)
    "compress": 0.30,            # composite above this → COMPRESSIBLE (summarise)
    # below compress threshold → NOISE (discard)
}


class RelevanceScorer:
    """
    Scores all messages in a conversation against the current query.
    Returns a list of ScoredMessage, one per input message.
    """

    def __init__(self, recency_lambda: float = 0.015):
        """
        recency_lambda: controls decay speed.
          0.01 = gentle (old messages still somewhat relevant)
          0.03 = steep (only recent messages matter)
        """
        self.recency_lambda = recency_lambda
        self._embedder = None  # lazy load

    # ── Public API ────────────────────────────────────────────────────────────

    def score_all(
        self,
        messages: list[Message],
        query: str,
        query_type: Optional[QueryType] = None,
    ) -> list[ScoredMessage]:
        """Score every message. Returns same-length list as input."""
        if not messages:
            return []

        if query_type is None:
            query_type = self.detect_query_type(query)

        weights = WEIGHT_PROFILES[query_type]
        n = len(messages)

        # Batch embed all messages + query in one call (efficient)
        embedder = get_embedder()
        texts = [m.content for m in messages]
        all_embeddings = embedder.encode(texts + [query], normalize_embeddings=True)
        message_embeddings = all_embeddings[:n]
        query_embedding = all_embeddings[n]

        # Pre-tokenise query for keyword scoring
        query_terms = self._tokenise(query)

        scored = []
        for i, msg in enumerate(messages):
            # A: Keyword
            kw = self._keyword_score(msg.content, query_terms)

            # B: Semantic
            sem = float(np.dot(message_embeddings[i], query_embedding))
            sem = max(0.0, min(1.0, sem))

            # C: Recency (position from end of conversation)
            position_from_end = n - 1 - i
            rec = math.exp(-self.recency_lambda * position_from_end)

            # D: Landmark
            lm, lm_reason = self._landmark_score(msg.content)

            # Composite
            composite = (
                weights["keyword"] * kw
                + weights["semantic"] * sem
                + weights["recency"] * rec
                + weights["landmark"] * lm
            )
            composite = min(1.0, composite)

            # Classify
            classification = self._classify(composite, lm)

            scored.append(ScoredMessage(
                message=msg,
                keyword_score=round(kw, 4),
                semantic_score=round(sem, 4),
                recency_score=round(rec, 4),
                landmark_score=round(lm, 4),
                composite_score=round(composite, 4),
                classification=classification,
                landmark_reason=lm_reason,
            ))

        return scored

    def detect_query_type(self, query: str) -> QueryType:
        """Classify the query into one of four types using pattern matching."""
        q = query.lower()

        factual = sum(1 for p in _FACTUAL_SIGNALS if re.search(p, q))
        analytical = sum(1 for p in _ANALYTICAL_SIGNALS if re.search(p, q))
        status = sum(1 for p in _STATUS_SIGNALS if re.search(p, q))

        scores = {
            QueryType.FACTUAL: factual,
            QueryType.ANALYTICAL: analytical,
            QueryType.STATUS: status,
        }
        best_type, best_score = max(scores.items(), key=lambda x: x[1])

        if best_score == 0:
            return QueryType.GENERAL
        return best_type

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> set[str]:
        """Lowercase word tokens, strip punctuation, remove stopwords."""
        _STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "need",
            "i", "we", "you", "they", "he", "she", "it", "this", "that",
            "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "about", "what", "how", "when", "where", "who", "which",
        }
        tokens = re.findall(r"\b[a-z][a-z0-9]*\b", text.lower())
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}

    def _keyword_score(self, content: str, query_terms: set[str]) -> float:
        """TF-style keyword overlap score."""
        if not query_terms:
            return 0.0
        content_terms = self._tokenise(content)
        if not content_terms:
            return 0.0
        overlap = query_terms & content_terms
        # Jaccard-style: overlap / union, but capped by query coverage
        precision = len(overlap) / len(content_terms) if content_terms else 0
        recall = len(overlap) / len(query_terms) if query_terms else 0
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _landmark_score(self, content: str) -> tuple[float, Optional[str]]:
        """
        Detect decision/commitment/deadline language.
        Returns (score 0-1, reason string or None).
        """
        content_lower = content.lower()
        matches = []
        for pattern in _LANDMARK_PATTERNS:
            m = re.search(pattern, content_lower)
            if m:
                matches.append(m.group(0))

        if not matches:
            return 0.0, None

        # More matches = higher confidence it's a landmark
        score = min(1.0, 0.5 + 0.25 * len(matches))
        reason = f"contains: {', '.join(matches[:2])}"
        return score, reason

    def _classify(self, composite: float, landmark_score: float) -> MessageClass:
        """Map scores to a MessageClass."""
        if landmark_score >= THRESHOLDS["landmark_override"]:
            return MessageClass.LANDMARK
        if composite >= THRESHOLDS["keep"]:
            return MessageClass.RELEVANT
        if composite >= THRESHOLDS["compress"]:
            return MessageClass.COMPRESSIBLE
        return MessageClass.NOISE
