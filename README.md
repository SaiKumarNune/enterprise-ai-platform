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

Configure `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`, `RAG_CONTEXT_CHARS`, `RAG_MIN_SIMILARITY`, and `RAG_RETRIEVE_MULTIPLIER` in `.env`.

`RAG_MIN_SIMILARITY` (default `0.2`) drops weak neighbors before generation. Higher values increase abstention and reduce noisy context; lower values increase recall and can add irrelevant chunks. `RAG_RETRIEVE_MULTIPLIER` fetches extra candidates (capped at 20) then filters, dedupes, and assigns stable `[S1]…` IDs in similarity order. `/search` is unchanged and does not use this threshold.

### RAG architecture

1. Embed the question with MiniLM.
2. Retrieve extra neighbors from pgvector (same cosine search as `/search`).
3. Filter by `RAG_MIN_SIMILARITY`, drop near-duplicates (keep longer evidence), apply a character budget.
4. Generate a short synthesized answer with a local Ollama model. Document text is wrapped as untrusted data.
5. Parse `[S#]` citations, strip unknown IDs, and flag missing citations on factual answers.
6. Return `answer`, `sources`, `grounded`, `citation_valid`, and `abstained`.

`grounded` / `citation_valid` are heuristics. They do not prove the answer is factually correct.

### Offline evaluation

```bash
python eval/run_rag_eval.py
python eval/run_rag_eval.py --output /tmp/rag_eval.json
```

Uses synthetic documents in `eval/datasets/rag_quality_v1.json` plus fake embeddings/LLM. It does not write to PostgreSQL or download models. Mock aggregates are not MiniLM or Ollama quality scores.

| Metric | Definition |
|---|---|
| Recall@k | Gold chunk IDs found in the top-k retrieved chunks |
| Citation validity rate | Remaining citations map to retrieved source IDs |
| Abstention accuracy | Abstention matches `expect_abstain` |
| Mean latency ms | Mock-runner wall time, not Ollama latency |

```bash
curl -X POST http://127.0.0.1:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees receive?", "top_k": 5}'
```

Response fields: `answer`, `sources` (`source_id`, `document_id`, `chunk_id`, `chunk_index`, `page_number`, `similarity`), `retrieval_count`, `grounded`, `citation_valid`, `abstained`, `model`. `similarity` is cosine similarity. The answer cites sources as `[S1]`, `[S2]`, or grouped `[S1, S4]`. Unknown IDs are stripped, remaining citations always match `sources`, and `citation_valid` is false if any invalid ID was generated or a factual answer lacks citations. Stripping an invalid ID does not make the response citation-valid. Factual answers without citations stay in the response with `grounded=false` and `citation_valid=false`; they are not rewritten into the insufficient-information fallback.

This is not a guarantee of factual correctness. Prompting and citation checks reduce some failure modes; they do not eliminate hallucinations or prove that a cited claim is true. Latency is limited by local embedding + Ollama generation on the request thread (`503` if Ollama is down, `504` on timeout).

## Agent workflows (development only)

`POST /agents/run` starts a **synchronous** LangGraph `StateGraph`: classify/plan → retrieve (existing RAG search) → prepare answer or draft → approval gate or finalize.

These endpoints are **unauthenticated and DEVELOPMENT-ONLY**. Do not expose them as an organizational approval system.

LangGraph orchestrates in-process nodes and does **not** use an in-memory checkpointer as source of truth. The durable record is the `agent_workflows` PostgreSQL row (status, draft, sources, decision, simulated result, `graph_state` snapshot). Process restart is recovered from that row, not from a MemorySaver.

```text
START
  ├─ classify_or_plan
  │     → retrieve_context  (knowledge_search, read-only)
  │         → prepare_answer_or_draft
  │               ├─ informational / clarification / abstain → finalize → END
  │               └─ draft leave_request → approval_gate → END (pending_approval)
  └─ after API decision
        ├─ approve → execute_approved_action (simulated DB record only) → finalize
        └─ reject  → reject_gate → finalize
```

Allowlisted tools: `knowledge_search`, `prepare_approval_request`, `execute_approved_action`. Planner JSON is untrusted; unknown tools and malformed output fall back to deterministic routing. The model cannot authorize execution. Only `POST /agents/{id}/approve` sets `authorized=True` after a row lock on `pending_approval`.

Simulated execution writes `execution_result.simulated=true` on the workflow row. It does **not** send email, call external APIs, or change documents.

```bash
alembic upgrade head   # revision e7a1b0c2d3f4
```

```bash
curl -X POST http://127.0.0.1:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{"request": "Review the vacation policy and prepare an approval request for three days off.", "top_k": 5}'

curl http://127.0.0.1:8000/agents/$WORKFLOW_ID

curl -X POST http://127.0.0.1:8000/agents/$WORKFLOW_ID/approve \
  -H "Content-Type: application/json" -d '{}'

curl -X POST http://127.0.0.1:8000/agents/$WORKFLOW_ID/reject \
  -H "Content-Type: application/json" -d '{}'
```

Example pending response (abridged): `status=pending_approval`, `draft.action_type=leave_request`, `draft.days=3`, `draft.reason=Not provided` when the user omitted a reason, `sources=[{source_id, document_id, chunk_id, ...}]`, `execution_result=null`. After approve: `status=completed`, `execution_result.simulated=true`. Drafts do not invent dates, reasons, or names; missing optional fields use `Not provided`, and missing required days asks for clarification.

`AGENT_MAX_STEPS` (default `8`) caps graph recursion. `/search` and `/rag/ask` contracts are unchanged.

## Run tests

```bash
pytest
```

Tests use a fake embedder, an in-memory session, and temporary directories. They do not use development PostgreSQL and they do not download the MiniLM model.
