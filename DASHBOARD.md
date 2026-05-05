# Dashboard Guide

A browser-based visualization of the scoring algorithm. This document explains what the dashboard is, what it is NOT, and how to use it.

---

## What It Is

`dashboard/index.html` is a **scoring algorithm explainer**. It runs entirely in the browser and lets a reviewer paste any conversation and any query, then see:

- How each message is scored across the four signals (keyword match, semantic similarity, recency decay, landmark detection)
- Which messages are classified as LANDMARK, RELEVANT, COMPRESSIBLE, or NOISE
- Which messages the system would keep verbatim, summarise, or discard
- The query type classification (FACTUAL / ANALYTICAL / STATUS / GENERAL) and the corresponding weight profile being applied

The dashboard mirrors the exact weight profiles, thresholds, and landmark patterns used by the Python pipeline. The algorithm is faithful to what runs in production.

---

## What It Is NOT

The dashboard is not the production system. Specifically:

- **It does not call the Anthropic API.** No compression, no LLM summarisation, no judge — all of that lives in the Python pipeline.
- **It does not use real sentence-transformers embeddings.** It uses Jaccard token overlap as a fast, dependency-free stand-in for semantic similarity. Classifications match the Python pipeline on simple cases; they can diverge on queries where nuanced semantics matter.
- **It is not a substitute for running the eval.** Reproducing the 100% win rate and 42.3% token reduction requires running `python -m evaluation.eval_runner` against the Python pipeline with a real API key.

The dashboard exists so that a reviewer can see the scoring logic work in thirty seconds without installing Python, setting up a virtual environment, or providing an API key. The Python pipeline in `optimizer/` is the authoritative system.

---

## How to Use the Dashboard

**No setup required.** Works on any machine with a modern browser.

1. Clone the repository (or download it as a zip)
2. Open `dashboard/index.html` by double-clicking it, or drag it into a browser window
3. Click any of the four example buttons (Factual Query, Analytical Query, Status Query, or DB Decision Thread) to load a pre-populated scenario
4. Or paste your own conversation into the left panel and your own query into the query box, then click "Optimize"
5. The right panel shows per-message scores, classifications, and which messages would be kept, compressed, or discarded

Chrome, Safari, Firefox, and Edge all work. No server needed.

---

## How to Run the Actual Python Pipeline

For the real production system — actual LLM compression, actual sentence-transformer embeddings, actual API calls — follow the setup in the root `README.md`:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

Then run any of:

```
pytest tests/ -v                     # 24 tests, should all pass
python -m evaluation.eval_runner     # reproduces the 11/11 win rate, 42.3% reduction
uvicorn api.server:app --reload      # starts the FastAPI endpoint on localhost:8000
```

The eval runner takes about 3-5 minutes and makes real API calls (cost: approximately $0.50-$1.00 depending on current pricing). The numbers in `docs/report.md` come directly from its output.

---

## Why Separate?

I kept the dashboard browser-only on purpose. A visualizer that requires running a backend is a visualizer most reviewers will never see. A visualizer that opens by double-clicking an HTML file is one anyone can use in thirty seconds. The trade-off is that the dashboard uses a simplified semantic scorer (Jaccard) rather than the real embedding model, which I've noted in the amber banner at the top of the dashboard itself.

For a production integration — wiring the dashboard to hit the FastAPI endpoint with CORS configured — would be roughly an hour of work. It is listed implicitly as part of the observability / router roadmap item but was not prioritised for this submission because the Python pipeline is already fully testable via `pytest` and `python -m evaluation.eval_runner`.
