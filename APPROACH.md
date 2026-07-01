# Approach

## Catalog And Context

The system uses `data/catalog.json` as the single source of truth. The scraper in `scripts/scrape_catalog.py` visits the SHL product catalog, follows pagination, filters for Individual Test Solutions, and then fetches each assessment detail page. Each record stores the assessment name, URL, test type codes, remote testing support, adaptive/IRT support, duration, and description/job-relevance text.

The application never lets the model invent catalog data. Recommendation URLs are looked up from `catalog.json` after candidate names are chosen. If a model returns a name that cannot be matched to the catalog, it is removed.

## Retrieval

Retrieval is hybrid:

- Semantic retrieval uses sentence-transformer embeddings plus FAISS when installed.
- Keyword retrieval uses BM25 over names, test types, durations, and descriptions.
- A TF-IDF fallback keeps the service usable in constrained local environments.

At request time, both semantic and keyword results are combined and re-ranked. Only the top candidates are passed to the response builder or optional LLM layer. This supports both broad job-description recommendations and exact-name comparison queries.

## Stateless Conversation Handling

Every `/chat` request receives the full message history. The service reconstructs state from that transcript on each call:

- It classifies the latest user turn as recommendation, refinement, comparison, off-topic/injection, or chitchat.
- It extracts a requirements slot-set from the full transcript: role, skills, seniority, duration, test types, remote-testing preference, and adaptive/IRT preference.
- It asks one targeted clarifying question when the slot-set is too sparse.
- It re-runs retrieval whenever the user refines the request, so no server-side session state is needed.

## Prompt And Output Control

The service can run without an LLM. If `LLM_PROVIDER=openai` is configured, the model receives only retrieved candidates plus strict JSON instructions. Regardless of provider, the final response is validated against the local catalog.

The returned schema is:

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": ["K"],
      "duration_minutes": 30,
      "remote_testing": true,
      "adaptive_irt": false,
      "reason": "string"
    }
  ],
  "end_of_conversation": false
}
```

## Evaluation

`eval.py` replays trace files turn-by-turn against a running local service and computes Recall@10 from final recommendations. The same harness can be extended with adversarial probes for prompt injection, off-topic requests, mid-conversation refinements, and named-assessment comparisons.

## Known Tradeoffs

The deterministic fallback is deliberately conservative. It is fast, testable, and robust under a 30-second timeout, but a configured LLM can produce more nuanced justifications when the retrieved candidate set is good. The critical safety behavior remains the same in both modes: all final assessment names and URLs must validate against `catalog.json`.
