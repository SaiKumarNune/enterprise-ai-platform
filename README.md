# Enterprise AI Platform

Backend foundation for an Enterprise AI Knowledge & Workflow Automation Platform.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`sentence-transformers` may download `all-MiniLM-L6-v2` (384 dimensions) the first time you process a document or search. That uses local CPU and RAM. Tests never load the model.

## Run the API

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Open http://127.0.0.1:8000/docs.

## PostgreSQL with pgvector

Local Compose uses `pgvector/pgvector:pg17` (PostgreSQL 17) so it can reuse the existing `postgres_data` volume. Do **not** run `docker compose down -v` when changing this image.

```bash
docker compose up -d db
docker compose exec db psql -U enterprise_user -d enterprise_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Database migrations

```bash
alembic current
alembic upgrade head
```

## Document and search API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/documents` | Metadata-only create |
| `GET` | `/documents` | List with `skip`/`limit` |
| `GET` | `/documents/{id}` | Fetch one document |
| `PATCH` | `/documents/{id}` | `filename` and `content_type` only |
| `DELETE` | `/documents/{id}` | Metadata, chunks, best-effort file cleanup |
| `POST` | `/documents/upload` | PDF or UTF-8 text |
| `GET` | `/documents/{id}/download` | Download stored file |
| `POST` | `/documents/{id}/process` | Extract, chunk, embed, persist |
| `GET` | `/documents/{id}/chunks` | Ordered chunk list |
| `POST` | `/search` | Semantic search over chunk embeddings |
| `POST` | `/rag/ask` | Citation-grounded RAG answer from retrieved chunks |

`POST /documents/{id}/process` is synchronous. It extracts text, chunks it, embeds all new chunks, then replaces previous chunks in one commit. If embedding fails, old chunks are left unchanged and status becomes `failed`.

`PATCH` cannot change `status`.

### Semantic search

`POST /search` embeds the query with the same model and ranks `document_chunks` by pgvector **cosine distance**. The JSON `similarity` field is **cosine similarity** (`1 - cosine_distance`) for L2-normalized 384-d vectors. Chunks without embeddings are excluded.

No ANN index (HNSW/IVFFlat) is created yet. Exact nearest-neighbor search is enough for local development volumes.

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vacation policy", "top_k": 5}'
```

Optional `document_id` limits results to one document (`404` if that document does not exist). `top_k` is 1–50 (default 5).

## RAG question answering

`POST /rag/ask` is **synchronous**. It embeds the question, retrieves chunks via the existing pgvector search, builds a bounded context window, and asks a local Ollama model. Retrieved text is passed as untrusted data, not as system instructions.

Ollama is installed in this environment (`0.33.3`) but the daemon was not running and no model was pulled during this change. Default model is `llama3.2:3b` (fits ~16 GB Apple Silicon). Do not download during tests.

```bash
ollama serve
ollama pull llama3.2:3b
```

Configure `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`, `RAG_CONTEXT_CHARS`, and `RAG_MIN_SIMILARITY` in `.env`.

```bash
curl -X POST http://127.0.0.1:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees receive?", "top_k": 5}'
```

Response fields: `answer`, `sources` (`source_id`, `document_id`, `chunk_id`, `page_number`, `similarity`), `retrieval_count`, `grounded`, `model`. `similarity` is cosine similarity. The answer cites sources as `[S1]`, `[S2]`. Unsupported citations are stripped; uncited model output is replaced with an insufficient-information message.

This is not a guarantee of factual correctness. Prompting and citation checks reduce hallucinations; they do not eliminate them. Latency is limited by local embedding + Ollama generation on the request thread (`503` if Ollama is down, `504` on timeout).

## Run tests

```bash
pytest
```

Tests use a fake embedder, an in-memory session, and temporary directories. They do not use development PostgreSQL and they do not download the MiniLM model.
