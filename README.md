# SHL Assessment Recommender

A stateless FastAPI service for recommending and comparing SHL Individual Test Solutions from a scraped catalog.

## What Is Included

- `scripts/scrape_catalog.py` scrapes SHL catalog listing and detail pages into `data/catalog.json`.
- `app/main.py` exposes `GET /health` and `POST /chat`.
- `app/retriever.py` builds or loads a hybrid semantic + BM25 index once at startup.
- `app/chat.py` reconstructs conversation state from the request history on every call.
- `eval.py` replays public traces and computes Recall@10.
- `tests/` covers routing, validation, and response behavior.
- `APPROACH.md` is the two-page implementation summary.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/scrape_catalog.py
uvicorn app.main:app --reload
```

Then call:

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"I need a Java assessment for a mid-level backend developer, under 45 minutes."}]}'
```

## Configuration

The service runs without an LLM by using deterministic retrieval and template responses. To enable LLM selection from retrieved candidates, set:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4.1-mini
```

By default, retrieval uses BM25 + TF-IDF so the API starts quickly. To use local sentence-transformer embeddings with FAISS, set:

```bash
export USE_SENTENCE_TRANSFORMERS=true
```

The code validates every recommendation against `data/catalog.json`, so generated names or URLs that are not in the catalog are dropped.

## Evaluation

Place trace JSON files in `traces/`, each with:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "expected": ["Assessment Name A", "Assessment Name B"]
}
```

Run:

```bash
python eval.py --trace-dir traces --base-url http://127.0.0.1:8000
```

## Notes

- The API is intentionally stateless. The full `messages` array is the only conversation state.
- The retriever index is loaded once at process startup, not rebuilt per request.
- If `sentence-transformers` or `faiss` are unavailable, retrieval falls back to TF-IDF + BM25 so development remains lightweight.
