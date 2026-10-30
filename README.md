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

Set `JWT_SECRET` in `.env` to a unique value of at least 32 characters. The API refuses to issue or verify tokens if the secret is missing, shorter than 32 characters, or equal to the `.env.example` placeholder. Do not reuse the placeholder.

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

Phase 10 revision `b0c1d2e3f4a5` (tenants, document owner/ACL, audit) revises `a9b0c1d2e3f4`. Additive only; it does not drop tables or reset volumes.

**Existing data:** if `users`, `documents`, or `agent_workflows` already have rows, set `LEGACY_TENANT_ID`. Do **not** set `LEGACY_DOCUMENT_OWNER_ID` (blanket assignment is rejected). Map only verified documents:

```bash
LEGACY_TENANT_ID=<new-or-chosen-tenant-uuid> \
LEGACY_TENANT_NAME=Legacy \
LEGACY_DOCUMENT_OWNERS_BY_FILENAME='{"vacation_policy.txt":"<user-uuid>","Sai_Kumar_Nune_AI.pdf":"<user-uuid>"}' \
alembic upgrade head
```

Unmapped files such as `employee_handbook.pdf` stay **quarantined** (`owner_id` null). They are hidden from `/documents`, `/search`, `/rag/ask`, and agents, including ordinary tenant-admin access, until a tenant admin reviews metadata at `GET /admin/documents/quarantine` and assigns an owner with `POST /admin/documents/{id}/assign-owner`. That review response does not include file content or `storage_path`.

Look up IDs first:

```bash
docker compose exec db psql -U enterprise_user -d enterprise_ai -c "SELECT id, email FROM users;"
docker compose exec db psql -U enterprise_user -d enterprise_ai -c "SELECT id, filename FROM documents;"
```

Rollback (destroys tenant/ACL/audit schema added by this revision, not document rows themselves):

```bash
alembic downgrade a9b0c1d2e3f4
```

Greenfield databases create tenant `00000000-0000-4000-8000-000000000001` (`default`). Set `DEFAULT_TENANT_ID` to that value for self-registration.

## Authentication and RBAC

Roles are `employee`, `manager`, and `admin`. Self-registration always creates an **employee**. Privileged roles cannot be set through public HTTP APIs. Authorization reloads the user from the database on every request; JWT claims are not trusted for role checks.

| Method | Path | Who |
|---|---|---|
| `POST` | `/auth/register` | Public. Creates an employee. |
| `POST` | `/auth/login` | Public. JSON `{email, password}` returns a bearer access token. |
| `POST` | `/auth/token` | Public. OAuth2 password form (`username` = email) for Swagger Authorize. |
| `GET` | `/auth/me` | Authenticated user. |
| `POST` | `/agents/run` | Any active user. New workflows are owned by the caller. |
| `GET` | `/agents/{workflow_id}` | Owner, or manager/admin. Employees cannot read others' workflows. Historical rows with `owner_id=null` are readable by manager/admin only. |
| `POST` | `/agents/{id}/approve` | Manager or admin. |
| `POST` | `/agents/{id}/reject` | Manager or admin. |

`/health` and `/db-health` remain unauthenticated. Document, search, RAG, and agent routes require a JWT. Admins are **tenant-scoped** and cannot read or approve another tenant's resources.

### Document permission matrix

Privileges are `read`, `write`, and `manage` (`manage` includes write and read). Decisions use the authenticated user loaded from the database, never request-supplied IDs.

| Actor | Own documents | Same-tenant documents | Other tenants |
|---|---|---|---|
| Employee | manage (as owner) | none unless shared via ACL | none (404) |
| Manager | manage (as owner) | no automatic document access; can approve **same-tenant** workflows | none (404) |
| Admin | manage | manage assigned documents in **their** tenant; quarantined docs are **not** readable | none (404) |

`PUT /documents/{id}/permissions` and `DELETE /documents/{id}/permissions/{user_id}` require `manage`. Users cannot grant privileges to themselves. Shares cannot cross tenants. Vector search applies this filter **in the retrieval query** before ranking and `top_k`.

### Development-only admin bootstrap

There is no public role-upgrade endpoint. Promote a local admin from the project root after migrations:

```bash
BOOTSTRAP_ADMIN_ENABLED=true \
BOOTSTRAP_ADMIN_EMAIL=admin@localhost \
BOOTSTRAP_ADMIN_PASSWORD='choose-a-long-local-password' \
PYTHONPATH=backend python -m app.bootstrap_admin
```

Unset those variables afterward. This is not an HR identity provider.

### Swagger testing sequence

1. Copy `.env.example` to `.env` if needed, set a unique `JWT_SECRET`, run `alembic upgrade head`, start Postgres and `uvicorn`.
2. Open http://127.0.0.1:8000/docs.
3. `POST /auth/register`:

```json
{"email": "employee@example.com", "password": "correct-horse-battery"}
```

4. `POST /auth/login` with the same JSON. Copy `access_token`.
5. Click **Authorize**, paste the token (or use `/auth/token` with username `employee@example.com` and the password).
6. `GET /auth/me` — role is `employee`.
7. `POST /agents/run`:

```json
{"request": "Review the vacation policy and prepare an approval request for three days off.", "top_k": 5}
```

8. `GET /agents/{workflow_id}` as the employee.
9. `POST /agents/{workflow_id}/approve` as the employee — expect **403**.
10. Bootstrap a manager/admin with the CLI above (or a second user whose role was set only via bootstrap). Login as that user, Authorize with the new token.
11. `POST /agents/{workflow_id}/approve` — manager/admin only, **same tenant**. A second approve returns **409**. Stale invalid pending drafts return `needs_clarification` and are not executed.
12. As the employee, `POST /documents/upload` a text file, then `POST /documents/{id}/process`. `GET /documents` should list only that file.
13. As a second employee in the same tenant, `GET /documents/{id}` should be **404** until the owner `PUT /documents/{id}/permissions` with `{"user_id":"<colleague-id>","privilege":"read"}`.
14. As an admin from another tenant, `GET` of that document and `POST /agents/{id}/approve` of the first tenant's workflow should be **404**.
15. After migrating legacy data, tenant admin `GET /admin/documents/quarantine` should list `employee_handbook.pdf` metadata only. Ordinary `GET /documents/{id}` stays **404** until `POST /admin/documents/{id}/assign-owner`.

Sample login:

```bash
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "employee@example.com", "password": "correct-horse-battery"}'

curl -X POST http://127.0.0.1:8000/agents/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"request": "How many vacation days do full-time employees receive?", "top_k": 5}'
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
| `POST` | `/documents/{id}/process` | Extract, chunk, embed, persist (write) |
| `GET` | `/documents/{id}/permissions` | List ACL grants (manage) |
| `PUT` | `/documents/{id}/permissions` | Grant `read`/`write`/`manage` (manage) |
| `DELETE` | `/documents/{id}/permissions/{user_id}` | Revoke ACL grant (manage) |
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

`POST /agents/run` starts a **synchronous** LangGraph graph with four distinct agents. Agent endpoints require a JWT. Simulated execution is **not** an HR system integration and must not be treated as a production approval system.

| Agent | Responsibility |
|---|---|
| Planner | Structured task plan (`informational`, `clarification`, or allowlisted `leave_request`). Rejects unknown tools/actions. Never authorizes execution. |
| Research | Existing `knowledge_search` / RAG retrieval with optional `document_id`. Returns source IDs, chunk IDs, and similarity. Synthesizes cited informational answers. Does not invent sources. |
| Validator | Citation IDs vs retrieved evidence, missing required days, policy vs user personal fields, and simple allotment checks. `semantic_verification` is always false. |
| Action | Simulated `leave_request` drafts from user-provided details only. Execution runs only after `POST /agents/{id}/approve`. |

```text
START
  ├─ planner_agent → research_agent
  │     ├─ informational / clarification → validator_agent → finalize
  │     └─ leave_request → action_agent (draft) → validator_agent
  │           ├─ validation pass → approval_gate → END (pending_approval)
  │           └─ validation fail → finalize (needs_clarification / abstain)
  └─ after API decision
        ├─ approve → action_agent (simulated execute) → finalize
        └─ reject  → reject_gate → finalize
```

LangGraph is in-process orchestration only. PostgreSQL `agent_workflows` (including `agent_trace` and `graph_state`) is the durable source of truth. An in-memory checkpointer is not used for authorization. `AGENT_MAX_STEPS` (default 8) bounds the graph.

`agent_trace` is a list of `{agent, outcome, summary, issues[]}`. It does not include prompts, chain-of-thought, or document excerpt text.

Allowlisted tools: `knowledge_search`, `prepare_approval_request`, `execute_approved_action`. Only `POST /agents/{id}/approve` sets `authorized=True` after a row lock on `pending_approval`. Simulated execution writes `execution_result.simulated=true` and does not send email or change documents.

```bash
alembic upgrade head   # head b0c1d2e3f4a5 (tenants, ACL, audit); prior a9b0c1d2e3f4
```

```bash
curl -X POST http://127.0.0.1:8000/agents/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"request": "Review the vacation policy and prepare an approval request for three days off.", "top_k": 5}'

curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/agents/$WORKFLOW_ID

curl -X POST http://127.0.0.1:8000/agents/$WORKFLOW_ID/approve \
  -H "Authorization: Bearer $MANAGER_TOKEN" \
  -H "Content-Type: application/json" -d '{}'

curl -X POST http://127.0.0.1:8000/agents/$WORKFLOW_ID/reject \
  -H "Authorization: Bearer $MANAGER_TOKEN" \
  -H "Content-Type: application/json" -d '{}'
```

Example pending response (abridged): `status=pending_approval` only after the validator parses annual allowance, required reason/date rules, and citation IDs from retrieved evidence. Drafts do not invent dates, reasons, or names. If policy requires a reason or planned dates and the user omitted them, or if requested days exceed the parsed annual allowance, the workflow is `needs_clarification` with issue codes such as `days_exceed_policy`, `missing_reason`, `missing_dates`, or `policy_rule_unverified`. Annual entitlement is not treated as remaining balance (`employee_balance_unknown` warning).

**Already pending workflows:** Approve now re-runs these checks. A stale `pending_approval` row created before this patch (for example a 25-day draft against a 10-day policy) is moved to `needs_clarification` and is **not** executed. Reject those rows, or GET them after a failed approve, then submit a new `/agents/run` with complete user details.

`AGENT_MAX_STEPS` (default `8`) caps graph recursion. `/search` and `/rag/ask` contracts are unchanged.

### Local verification (Swagger)

1. Informational threshold: `POST /agents/run` with `When do vacation requests require manager approval?` and the vacation policy `document_id`. Expect `completed` and wording `3 or more consecutive days`, not `exceeding 3 days`.
2. Incomplete leave: `POST /agents/run` with `Review the vacation policy and prepare an approval request for three days off.` Expect `needs_clarification` plus `missing_reason` and `missing_dates` when the policy requires those fields.
3. Over-limit leave: `POST /agents/run` with `Review the vacation policy and prepare an approval request for 25 days off.` Expect `needs_clarification` and `days_exceed_policy`. A complete 3-day request with reason and dates within the parsed annual cap can reach `pending_approval`.

## Run tests

```bash
pytest
```

Tests use a fake embedder, an in-memory session, and temporary directories. They do not use development PostgreSQL and they do not download the MiniLM model. Agent tests authenticate with JWT secrets set only in the pytest process.
