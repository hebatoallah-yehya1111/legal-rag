# Arabic Legal Document Q&A

A RAG system that answers questions about the Egyptian Civil Code
(القانون المدني المصري) in Arabic, with every answer cited by article
number rather than chunk id or page number.

## Status

Scaffolding stage: project structure, API skeleton, schema, and test
harness are in place. The ingestion pipeline (`src/legal_rag/ingestion.py`)
is stubbed out and waiting on the source PDF - see `TODO`s inside that file.

## Quickstart (3 commands)

```bash
pip install -e ".[dev]"
uvicorn legal_rag.api:app --reload --port 8000
curl http://localhost:8000/health
```

Or with Docker:

```bash
docker compose up
curl http://localhost:8000/health
```

## Project layout

```
src/legal_rag/
  schema.py      # Article, AskRequest, AskResponse, HealthResponse
  config.py      # centralized settings (env-overridable)
  ingestion.py   # Step 0: PDF -> structured per-article JSON (TODO)
  rag.py         # embedding + vector store + generation (TODO)
  api.py         # FastAPI app: /ask, /health
tests/
  test_api.py
  test_ingestion.py
data/
  raw/           # source PDF (DVC-tracked)
  processed/     # articles.json, vector_store (DVC-tracked)
dvc.yaml         # build_corpus -> build_index pipeline
```

## Architecture

```
PDF (data/raw/)
  -> ingestion.py  [extract, normalize, validate]
  -> articles.json (data/processed/, DVC-tracked)
  -> rag.py build_index()  [embed per article, upsert to vector store]
  -> vector_store (data/processed/)
  -> api.py /ask  [retrieve top-k articles, generate cited answer]
```
## Development

```bash
pytest -v
ruff check src/ tests/
```
