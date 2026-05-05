# Architecture Decisions — LEC Context Optimizer

## System Overview

A three-tier filtering pipeline: Relevance Scorer → Compressor → Assembler. Takes a full conversation history and current query, returns a valid, compressed conversation thread 40-60% smaller than the original, with proven quality parity.

```
Tier 1 — Structural Preservation (Landmark Detection)
  Non-negotiable messages locked regardless of score.
  Decisions, commitments, deadlines, action items.

Tier 2 — Dynamic Relevance (Weighted Scoring)
  Four signals combined with adaptive weights per query type.
  Keyword match + semantic similarity + recency decay + landmark score.

Tier 3 — Lossy Compression (LLM Summarisation)
  Low-value contiguous groups collapsed into single summary messages.
  Claude Haiku used as compression model.
```

---

## Decision 1: Local Embeddings (sentence-transformers) over API Embeddings

**Chosen:** `all-MiniLM-L6-v2` via sentence-transformers, running locally

**Rejected:** Anthropic or OpenAI embedding API

**Why:**
The assembly latency target is p95 < 500ms for conversations up to 100 messages. An API round-trip for embeddings adds 100-300ms per call before any compression logic runs. With local inference, embedding a 100-message conversation takes ~40ms total on CPU. The quality difference between MiniLM and API embeddings is negligible for conversational text similarity — both encode semantic proximity well enough to distinguish relevant from irrelevant turns.

**Trade-off:** MiniLM is a 80MB download on first run. In production with Docker, this is baked into the image. Not a real constraint.

---

## Decision 2: Adaptive Weights per Query Type over Fixed Weights

**Chosen:** Four weight profiles (FACTUAL, ANALYTICAL, STATUS, GENERAL) that shift scoring emphasis based on detected query intent

**Rejected:** Single fixed weight set (e.g. keyword=0.25, semantic=0.35, recency=0.20, landmark=0.20)

**Why:**
Fixed weights fail on edge cases that are actually common in production. A factual query ("what was the deadline we agreed?") needs the landmark and keyword signals boosted — the answer is a specific fact that may be 50 messages old. Recency is irrelevant. An analytical query ("what were the trade-offs we discussed?") needs broad semantic coverage — boosting keyword would miss synonyms and paraphrases. A status query ("where are we on the frontend?") needs recency boosted — the most recent messages dominate.

One fixed weight set would compromise all three. The adaptive profiles are implemented as a lookup table (four dicts), not ML — deterministic, debuggable, and tunable.

**Trade-off:** Requires a query classifier. Implemented as regex pattern matching — fast and transparent. Downside: if the query is ambiguous, it falls back to GENERAL (balanced weights), which is conservative rather than wrong.

---

## Decision 3: Group Summarisation over Per-Message Summarisation

**Chosen:** Consecutive low-scoring messages grouped into blocks, one LLM call per block

**Rejected:** One LLM summarisation call per low-scoring message

**Why:**
Per-message summarisation is expensive and loses context between adjacent messages. A back-and-forth scheduling discussion across 8 messages is best summarised as a unit — "team agreed to meet Tuesday at 3pm." Summarising each message individually produces 8 disconnected summaries that are harder for the downstream LLM to reason over than one clean sentence. Grouping also cuts compression LLM calls from O(n_low_value_messages) to O(n_groups), which can be 5-10× cheaper.

**Trade-off:** Group boundaries are determined by classification changes (landmark/relevant → compressible/noise). If a landmark appears inside a noise block, it breaks the group, which is the correct behaviour.

---

## Decision 4: Structured Error Returns in Compressor over Exceptions

**Chosen:** Compressor returns `None` on summarisation failure, caller keeps originals

**Rejected:** Raise exception on Anthropic API failure

**Why:**
The compressor is called inside a scoring loop. If one group fails to summarise (API timeout, rate limit), crashing the entire optimization is the wrong trade-off. The safe fallback is: keep the original messages for that group. The context is slightly larger than optimal but valid. This is the difference between a demo and a production system.

---

## Decision 5: Assembler Validates and Repairs over Asserting Validity

**Chosen:** Assembler detects invariant violations (orphaned assistant turns, missing user before assistant) and repairs them by inserting bridging markers

**Rejected:** Assert that input is valid and raise if not

**Why:**
After scoring and compression, the remaining message set may have gaps. An assistant turn might be selected without its preceding user turn if the user turn scored below threshold. Rather than crashing (which would make the whole pipeline fail) or silently producing invalid output (which would confuse the downstream LLM), the assembler inserts a `[CONTEXT NOTE: earlier context compressed]` system message. The downstream LLM sees a valid thread with an explicit signal that context was omitted.

---

## Rejected Alternatives

| Alternative | Why Rejected |
|---|---|
| BART zero-shot classifier for landmark detection | 400MB model, 200ms inference — too slow for assembly latency target. Regex patterns catch 90% of landmarks at zero latency cost |
| Qdrant vector store for message retrieval | Overkill for conversations under 500 messages. Adds a required running service. Correct for 500+ message scale — first roadmap item |
| Truncation (drop oldest messages) | Destroys landmarks regardless of position. Loses critical decisions made early in conversation. Not defensible against the quality metric |
| Full conversation summarisation | One LLM call to summarise everything loses structure entirely. Tool call pairs break. Downstream LLM loses thread of recent decisions |
| tiktoken for token counting | Correct for exact counts, adds a dependency. Our ~4 chars/token estimate is sufficient for relative comparison and costs nothing |

---

## Concurrency and Production Considerations

Each POST /optimize request is stateless — all state lives in the function call scope. FastAPI + uvicorn handle request concurrency natively. The bottleneck at scale is the Anthropic API calls in the Compressor. At 100 concurrent users each triggering compression, this hits rate limits.

**Mitigation (roadmap):** Asyncio semaphore limiting concurrent Anthropic calls, exponential backoff on 429s, and an embedding cache keyed by message content hash (most messages in a long conversation don't change between queries).

---

## System Diagram

```
POST /optimize
{messages: [...], query: "..."}
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  ContextOptimizer                    │
│                                                      │
│  1. detect_query_type(query)                         │
│     → FACTUAL / ANALYTICAL / STATUS / GENERAL        │
│                                                      │
│  2. scorer.score_all(messages, query, query_type)    │
│     For each message:                                │
│       keyword_score  = F1(query_terms ∩ msg_terms)   │
│       semantic_score = cosine(embed(msg), embed(q))  │
│       recency_score  = exp(-λ × position_from_end)   │
│       landmark_score = regex_match(msg)              │
│       composite      = w1·kw + w2·sem + w3·rec + w4·lm │
│       classification = LANDMARK|RELEVANT|COMPRESS|NOISE │
│                                                      │
│  3. compressor.compress(scored_messages)             │
│     LANDMARK + RELEVANT → kept verbatim              │
│     COMPRESSIBLE groups → Haiku summarisation        │
│     NOISE (small) → discarded                        │
│                                                      │
│  4. assembler.assemble(kept, summaries, stats)       │
│     Sort by original index                           │
│     Repair invariant violations                      │
│     Return OptimizedContext                          │
└─────────────────────────────────────────────────────┘
        │
        ▼
{optimized_messages, token_reduction_pct,
 landmarks_preserved, assembly_latency_ms,
 compression_cost_usd, query_type_detected}
```
