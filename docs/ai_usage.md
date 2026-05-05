# AI Usage Note

## The Honest Breakdown

I used Claude (claude.ai) as a primary implementation collaborator throughout this project. I'm being specific because the assignment asked for judgment, not a clean-hands claim.

---

## What AI Generated

**Scaffolding and boilerplate:** File structure, FastAPI route signatures, Pydantic models, TypedDict definitions, `__init__.py` files, pytest configuration. I specified what each module needed to do — AI translated that into Python structure.

**Algorithm first drafts:** The AST-based calculator, the cosine similarity scoring loop, the exponential recency decay formula, and the regex pattern list for landmark detection were all AI-generated first drafts based on my architectural spec. Two of these required correction after testing (landmark patterns, sandbox import handling).

**Test structure:** The test class organisation and individual test method stubs were AI-generated. The test cases themselves — what to assert, which edge cases to cover, what constitutes a meaningful test — were my decisions.

**Docstrings:** Largely AI-generated from surrounding code context.

---

## What Is Mine

**All architectural decisions:** The three-tier filtering model (structural preservation → dynamic relevance → lossy compression), the choice of local embeddings over API embeddings, the adaptive weight profiles per query type, the group summarisation strategy, the assembler repair logic — every decision in `docs/architecture.md` was made by me for reasons I can defend.

**The scoring weight profiles:** The specific numbers in `WEIGHT_PROFILES` — why FACTUAL gives landmark 0.45 and recency 0.10, why ANALYTICAL gives semantic 0.50 — were my decisions, made by reasoning about what each query type needs, then verified against the eval results.

**The eval dataset design:** The three conversations (backend DB architecture, Q3 roadmap, production incident), their embedded landmarks, and the 10+ queries covering factual/analytical/status types were designed by me. The coverage is intentional — each conversation has different landmark density, different query types, and different noise ratios.

**The evaluation methodology:** The head-to-head structure (full context vs. optimised context, LLM judge, three scoring dimensions), the win rate metric, the net cost analysis formula — these were my design. AI wrote the Python that executes it.

**Failure analysis:** The three bugs documented in the report (landmark pattern coverage, sandbox import collision, query type classifier ties) were found by me during testing and diagnosed by me before being fixed. AI assisted with the fix implementation in two of three cases.

---

## How I Verified What AI Produced

Every function has at least one test. I did not merge AI-generated code without running `pytest tests/ -v`. When tests failed (two did on first run), I read the error, diagnosed the root cause, and fixed it — I did not ask AI to fix blindly.

For the scoring logic specifically: I manually traced through the weight calculation for three sample messages before trusting the output. For the assembler: I inserted print statements to verify that invariant repair was triggering on the correct cases.

The eval runner I ran end-to-end with a real API key before finalising the submission. The numbers in the report are from actual runs, not estimated.

---

## Summary

AI wrote roughly 60% of the lines of code. It wrote 0% of the architectural decisions, 0% of the eval design, and 0% of the failure analysis. The judgment about what to build, why it's structured this way, and what the numbers mean is mine. That's the part I expect to defend in the technical call.
