# Coffee Guide: RAG with Elasticsearch

A retrieval-augmented generation (RAG) app that answers questions about choosing coffee beans at the store
and brewing them at home. **Elasticsearch** runs hybrid retrieval (BM25 keyword search plus kNN vector
search, merged with reciprocal rank fusion) over a curated coffee knowledge base, and **Claude** writes
answers that cite the guide sections they came from.

**[View the technical presentation →](https://rachnamunagala6.github.io/coffee-guide-rag/)**

**Stack:** Elasticsearch 8.17 · Python · sentence-transformers (all-MiniLM-L6-v2) · Anthropic Claude API · FastAPI · Docker · HTML/JavaScript

Example questions:
- "What should I look for on a bag of coffee at the grocery store?"
- "I like smooth, chocolatey coffee. Which beans should I buy?"
- "How do I make coffee with a French press?"
- "My pour-over tastes sour. What am I doing wrong?"
- "Should I keep my beans in the freezer?"

---

## How it works

```
docs/*.md ──► ingest.py ──► split by heading ─► embed ─► Elasticsearch index
                                                              │
question ──► ask.py ──► search.py (BM25 + kNN ─► RRF) ────────┘──► top 5 sections ─► Claude ─► answer + sources
```

1. **ingest.py** splits each guide by its headings, so every chunk is one topic
   (for example "French press" or "Check the roast date first"). It embeds each chunk and stores
   the text and vector in Elasticsearch.
2. **search.py** runs keyword search (BM25, with extra weight on section headings and English
   stemming so "grinding" matches "grind") and vector search (matching by meaning, so "my coffee
   tastes like vinegar" finds the "sour" troubleshooting section). It merges both rankings with
   reciprocal rank fusion.
3. **ask.py** gives the top sections to Claude with instructions to act as a friendly coffee
   guide: direct answer first, practical amounts and steps, citations, and no invented specifics.

## The knowledge base (`docs/`)

| Guide | Covers |
|---|---|
| `01-buying-beans-at-the-store.md` | Roast dates, whole vs ground, packaging, labels, Arabica vs Robusta, decaf, certifications, price, a store checklist |
| `02-choosing-flavor-roast-and-origin.md` | Roast levels, flavor by country, processing methods, matching beans to your taste and brewer |
| `03-storing-beans.md` | Containers, fridge vs freezer, signs of stale coffee |
| `04-grinding.md` | Burr vs blade, grind size chart by method, adjusting grind |
| `05-brewing-basics.md` | Coffee-to-water ratio, measuring without a scale, water, temperature, bloom, extraction |
| `06-brew-methods.md` | Step-by-step drip, pour-over, Chemex, French press, AeroPress, moka pot, espresso, cold brew, iced, Turkish |
| `07-troubleshooting-taste.md` | Fixes for sour, bitter, weak, flat, muddy and other problems |

`docs/index.html` is the presentation page served by GitHub Pages; the ingest script only reads
`.md` and `.txt` files, so it is ignored when indexing.

Edit these files or add new `.md` files (use `#` for the title and `##` for each topic), then
re-run `python ingest.py docs/`. Re-ingesting a file replaces its old chunks.

## Setup

Requires Docker and Python 3.10+.

```bash
# 1. Start Elasticsearch (wait ~30s, then check http://localhost:9200)
docker compose up -d

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env      # then put your ANTHROPIC_API_KEY in .env

# 4. Index the coffee guides (first run downloads the ~90MB embedding model)
python ingest.py docs/ --recreate
```

## Use it in your browser

```bash
python app.py        # then open http://localhost:8000  (Ctrl+C to stop)
```

The page has example questions, keeps the conversation so follow-ups work ("what about for
espresso?"), and shows which guide sections each answer came from (hover a red number to see it).
The code is `app.py` (web server) and `static/index.html` (the page).

## Use it in the terminal

```bash
python ask.py                                            # interactive, shows example questions
python ask.py "how much coffee per cup for a drip machine?"
python search.py "french press grind size"               # inspect retrieval only, no LLM call
```

If an answer looks wrong, run the same question through `search.py` first. If the right section
isn't in the top results, it's a retrieval problem (wording in the guides, chunking or the embedding
model); if it is, it's the prompt in `ask.py`.

## Settings (`.env`)

- `ES_INDEX` (default `coffee-guide`): index name.
- `TOP_K` (default 5): how many guide sections Claude sees.
- `CHUNK_WORDS` / `CHUNK_OVERLAP`: only matter for sections longer than `CHUNK_WORDS`, which get split further.
- `EMBED_MODEL`: `all-MiniLM-L6-v2` is small and fast. `BAAI/bge-small-en-v1.5` usually retrieves better. Re-run with `--recreate` after changing it.

## Next steps

- **Measure retrieval:** recall@5 and MRR on a labeled set of real questions, before tuning anything else.
- **Rerank:** score the top 20 fused hits with a cross-encoder, then keep the best 5.
- **Embed in the cluster:** Elastic inference endpoints instead of a local Python model.
- **Native hybrid search:** Elasticsearch's retriever API with built-in RRF, where the version and license allow.
- **Metadata filters:** brew method or roast as keyword fields, filtering both retrievers.
- **Production hardening:** Elasticsearch security and API keys, streamed responses, caching repeat questions.
