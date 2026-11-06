# Enterprise AI Knowledge & Workflow Platform

A **local** FastAPI backend that shows how an experienced AI engineer would structure retrieval-augmented generation, hybrid search, multi-agent workflows, and evaluation—without pretending this is a hosted product.

It is a portfolio and review artifact: JWT/RBAC, tenant isolation, citation-grounded RAG, LangGraph agents with human approval, in-process MCP tool calls, and deterministic eval runners. It is **not** deployed, not a public demo, and not a production HR or identity system.

## Capabilities

| Area | What exists today |
|---|---|
| API | FastAPI + Pydantic v2, OpenAPI at `/docs`, local HTML UI at `/app/` |
| Auth | Argon2 passwords, HS256 JWT, optional OIDC (authorization code + PKCE), invite-only registration, DB-owned RBAC |
| Tenancy | Per-request DB principal, document ACL, quarantined unowned files, SQL filters before ranking |
| Documents | PDF/text upload; sync process by default or `wait=false` jobs + worker |
| Retrieval | Semantic (pgvector cosine), lexical (PostgreSQL FTS `ts_rank_cd`), hybrid RRF |
| RAG | Authorized retrieve, evidence-sufficiency check, Ollama generation, `[S#]` citations, abstention |
| Agents | LangGraph planner / research / validator / action; durable `agent_workflows` rows |
| MCP | Official Python SDK, in-process only; identity bound server-side |
| Approval | Manager/admin approve or reject; `execute_approved_action` is not an MCP tool |
| Ticketing | Optional mock adapter (`TICKETING_ENABLED=false`); prepare via MCP, create only after API approval |
| Observability | JSON request logs, Prometheus `/metrics`, `/health/live` and `/health/ready` |
| Evaluation | Offline mock + opt-in live Ollama and hosted OpenAI-compatible RAG; agents and MCP stay in-process |
| CI | GitHub Actions with isolated pgvector PostgreSQL 17 |

## Architecture

```mermaid
flowchart TB
  subgraph clients [Clients]
    OpenAPI[OpenAPI / HTTP]
    LocalUI[Jinja2 + HTMX /app]
    EvalRunners[Eval runners]
  end

  subgraph api [FastAPI]
    Auth[JWT + optional OIDC]
    DocsAPI[Documents]
    SearchAPI["/search"]
    RagAPI["/rag/ask"]
    AgentAPI["/agents"]
    Health["/health /metrics"]
  end

  subgraph authz [Authorization]
    Principal[DB-backed Principal]
    ACL[Owner + ACL + quarantine]
  end

  subgraph retrieval [Retrieval]
    Lexical[PostgreSQL FTS ts_rank_cd]
    Semantic[pgvector cosine]
    RRF[RRF fusion]
    Sufficiency[Evidence sufficiency]
  end

  subgraph agents [LangGraph]
    Planner[Planner]
    Research[Research]
    Validator[Validator]
    Action[Action draft]
    Gate[Approval gate]
  end

  subgraph mcp [In-process MCP]
    Knowledge[knowledge_search]
    Prepare[prepare_approval_request]
    PrepareTicket[prepare_ticket]
    CreateTicket[create_ticket gated]
  end
  MockTickets[In-process mock tickets]

  PG[(PostgreSQL 17 + pgvector + GIN FTS)]
  MiniLM[MiniLM embeddings]
  Ollama[Ollama llama3.2:3b]
  Logs[JSON logs + Prometheus]

  OpenAPI --> Auth
  LocalUI --> Auth
  Auth --> Principal
  Principal --> ACL
  ACL --> DocsAPI
  ACL --> SearchAPI
  ACL --> RagAPI
  ACL --> AgentAPI
  DocsAPI --> PG
  DocsAPI --> MiniLM
  SearchAPI --> Lexical
  SearchAPI --> Semantic
  Lexical --> PG
  Semantic --> PG
  Semantic --> MiniLM
  Lexical --> RRF
  Semantic --> RRF
  RagAPI --> RRF
  RagAPI --> Sufficiency
  Sufficiency --> Ollama
  AgentAPI --> Planner
  Planner --> Research
  Research --> Knowledge
  Knowledge --> RRF
  Research --> Validator
  Planner --> Action
  Action --> Prepare
  Action --> PrepareTicket
  Action --> Validator
  Validator --> Gate
  Gate --> PG
  Gate --> CreateTicket
  CreateTicket --> MockTickets
  EvalRunners -.-> api
  api --> Logs
```

Rationale for RRF vs BM25, in-process MCP, tenancy, approval gates, and local models: [docs/design-decisions.md](docs/design-decisions.md). Eval details: [eval/README.md](eval/README.md). Latest local verification: [docs/final-audit.md](docs/final-audit.md).

## Stack

Python 3.13, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL 17 + pgvector, LangGraph, sentence-transformers MiniLM, Ollama, official MCP SDK, Prometheus client. Tests use fake embeddings, scripted LLMs, and an in-memory session unless `TEST_DATABASE_URL` points at an isolated `*_test` database.

## Setup

Requires Python 3.13, Docker (for PostgreSQL), and optionally Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `JWT_SECRET` in `.env` to a unique value of at least 32 characters. The API refuses tokens if the secret is missing, too short, or still the `.env.example` placeholder.

```bash
docker compose up -d db
docker compose exec db psql -U enterprise_user -d enterprise_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Open http://127.0.0.1:8000/docs for the API, or http://127.0.0.1:8000/app/login for the local HTML UI (password login, then documents / search / RAG / agents). The UI calls the same FastAPI handlers and does not bypass tenant ACL or RBAC.

### Local UI

The UI is Jinja2 templates plus a small HTMX-style helper under `/app/`. It stores the existing HS256 access token in a signed session cookie (not a separate identity system).

```bash
uvicorn app.main:app --reload --app-dir backend
# http://127.0.0.1:8000/app/login
```

Document upload from the UI enqueues `POST /documents/{id}/process?wait=false`. Start a worker if you want jobs to leave `queued`:

```bash
PYTHONPATH=backend python -m app.worker
```

There is no hosted frontend, CDN app, or public demo.

Local Compose uses `pgvector/pgvector:pg17` so it can reuse the existing `postgres_data` volume. Do **not** run `docker compose down -v` unless you intend to destroy that volume.

Schema head is `e3f4a5b6c7d8` (invitations, `external_identities`, OIDC login state; `users.password_hash` nullable for IdP-only accounts). Additive only; it does not drop tables. Greenfield databases create tenant `00000000-0000-4000-8000-000000000001` (`default`). Set `DEFAULT_TENANT_ID` to that value for open self-registration.

Existing rows: set `LEGACY_TENANT_ID` and map verified documents only (`LEGACY_DOCUMENT_OWNERS` or `LEGACY_DOCUMENT_OWNERS_BY_FILENAME`). Do **not** set `LEGACY_DOCUMENT_OWNER_ID`. Unmapped files stay quarantined (`owner_id` null) until a tenant admin assigns an owner. Details remain in Alembic revision comments and the previous operations notes below.

### RAG generation (optional)

Tests never start Ollama. For live answers:

```bash
ollama serve
ollama pull llama3.2:3b
```

`POST /documents/{id}/process` stays **synchronous by default** (`wait=true`, HTTP 200, existing clients unchanged). Pass `wait=false` to enqueue a PostgreSQL-backed job (HTTP 202) and poll `GET /documents/{id}/processing-jobs/{job_id}`. Run a separate worker; it is not Celery and does not use Redis:

```bash
alembic upgrade head   # includes d2e3f4a5b6c7 document_processing_jobs
PYTHONPATH=backend python -m app.worker
# or one shot:
PYTHONPATH=backend python -m app.worker --once
```

Embedding uses local CPU/RAM; the first MiniLM load may download `all-MiniLM-L6-v2`. `/rag/ask` remains synchronous on the API process.

### Auth bootstrap

`POST /auth/register` always creates an **employee**. Role is taken from the application database, never from an identity-provider claim. There is no public role-upgrade endpoint.

Password login (`POST /auth/login`, `POST /auth/token`) stays enabled for local development.

Invite-only registration (optional):

```bash
AUTH_INVITE_ONLY=true
```

A tenant admin creates invites with `POST /admin/invitations`. The plaintext token is returned **once**. Register with `invite_token`. Missing or mismatched invites return 403. The new user joins the **admin’s tenant**, not `DEFAULT_TENANT_ID`.

Optional OIDC (authorization code + PKCE, state, nonce). Disabled by default. No extra Docker IdP is bundled; point at any standards-compliant issuer you already run:

```bash
OIDC_ENABLED=true
OIDC_ISSUER=https://your-idp.example
OIDC_CLIENT_ID=enterprise-ai
# OIDC_CLIENT_SECRET=   # confidential clients only
OIDC_REDIRECT_URI=http://127.0.0.1:8000/auth/oidc/callback
OIDC_SCOPES=openid email
# OIDC_DISCOVERY_URL=   # optional; default is {issuer}/.well-known/openid-configuration
```

`GET /auth/oidc/login` redirects to the issuer. `GET /auth/oidc/callback` validates the ID token (issuer, audience, RS256 signature via JWKS, expiration, nonce) and returns the same local HS256 JWT as password login. Mapping key is `(issuer, sub)` stored in `external_identities`. IdP `role` / `groups` claims are ignored. New OIDC users still require an invitation when `AUTH_INVITE_ONLY=true`.

```bash
BOOTSTRAP_ADMIN_ENABLED=true \
BOOTSTRAP_ADMIN_EMAIL=admin@localhost \
BOOTSTRAP_ADMIN_PASSWORD='choose-a-long-local-password' \
PYTHONPATH=backend python -m app.bootstrap_admin
```

Unset those variables afterward.

## Tests

Default tests do not use development PostgreSQL `enterprise_ai` and do not download MiniLM.

```bash
python3 -m pytest
```

Isolated PostgreSQL tests (ACL, FTS, hybrid SQL) need a database whose name ends in `_test`. They refuse `enterprise_ai`:

```bash
PYTHONPATH=backend python3 -m tests.ensure_test_database
TEST_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@127.0.0.1:5432/enterprise_ai_acl_test' python3 -m pytest
```

```bash
python eval/run_rag_eval.py
python eval/run_retrieval_eval.py
python eval/run_agent_eval.py
python eval/run_mcp_eval.py
```

Live eval is opt-in (`--mode live`) and never falls back to mocks. Hosted eval (`--mode hosted`) is also opt-in and stays `not_run` without `EVAL_HOSTED_LLM_*`. See [eval/README.md](eval/README.md).

### Last verified local results

Recorded **2026-11-06** on this machine after Section 8 (final verification). Mock eval and mock bench numbers are not production measurements.

| Check | Result |
|---|---|
| Isolated pytest `TEST_DATABASE_URL=…/enterprise_ai_acl_test` | **292 passed, 0 skipped**, 13.88s, no Pydantic `lifespan` warning |
| `eval/run_rag_eval.py` (`offline_mock`) | **passed=true**, n=11, HashEmbedder/DatasetLLM |
| `eval/run_retrieval_eval.py` (`offline_mock_hash_embedder`) | **passed=true** |
| `eval/run_agent_eval.py` (`offline_mock`) | **passed=true**, n=13 |
| `eval/run_mcp_eval.py` | **passed=true**, n=12 |
| `bench/run_bench.py --mode mock` | **completed**; scenarios labeled `http_mock` / `worker_mock` / `in_memory` |
| Fresh Alembic on `enterprise_ai_migrate_test` | **head `e3f4a5b6c7d8`** (11 revisions, linear) |
| Live RAG / hosted LLM / live bench | **Not run** |

CI (`.github/workflows/ci.yml`) runs pytest against `pgvector/pgvector:pg17` with `enterprise_ai_acl_test`. Treat CI as the authority for isolated SQL tests when it is green.

## HTTP surface (condensed)

JSON document, search, RAG, and agent routes require a Bearer JWT except `/health`, `/health/live`, `/health/ready`, `/db-health`. `/metrics` is optional and may require `METRICS_TOKEN`. The `/app` HTML UI stores that JWT in a signed session cookie and still calls the same authorized handlers.

| Area | Routes |
|---|---|
| Auth | `POST /auth/register`, `/auth/login`, `/auth/token`; `GET /auth/me`; optional `GET /auth/oidc/login` and `/auth/oidc/callback`; admin `POST/GET /admin/invitations` |
| Local UI | `/app/login`, `/app/`, `/app/documents`, `/app/search`, `/app/ask`, `/app/agents` |
| Documents | upload/process/chunks, ACL `PUT/DELETE .../permissions`, admin quarantine |
| Search | `POST /search` (`semantic` \| `lexical` \| `hybrid`) |
| RAG | `POST /rag/ask` |
| Agents | `POST /agents/run`, `GET /agents/{id}`, manager `approve` / `reject` |

Roles: employees own their workflows and documents; managers/admins approve **same-tenant** workflows; admins manage assigned documents in **their** tenant only. Cross-tenant access is 404. Retrieval authorization is applied in SQL before `top_k`.

`POST /search` default mode follows `RETRIEVAL_MODE` (hybrid). Hit JSON shape is unchanged when `retrieval_mode` is omitted. Hybrid `similarity` is the RRF score; semantic mode reports cosine. `RAG_MIN_SIMILARITY` applies only to semantic RAG. No HNSW/IVFFlat index is installed.

Agent execution is **simulated**. `execution_result.simulated=true` means no email and no HR write. Ticket creation is off unless `TICKETING_ENABLED=true`; even then it writes only to an in-process mock, never a vendor API.

## Local benchmarks

Bounded laptop measurements are in [`bench/`](bench/README.md). Mock mode never opens PostgreSQL or Ollama. Database and live modes require an isolated `*_test` URL (`BENCH_DATABASE_URL` / `EVAL_DATABASE_URL` / `TEST_DATABASE_URL`) and refuse `enterprise_ai`.

```bash
python bench/run_bench.py --mode mock --requests 40 --concurrency 4 --jobs 8
```

Do not mix HashEmbedder/ScriptedLLM rows with MiniLM/Ollama rows.

## Limitations

- Not deployed. No public URL, no multi-host MCP, no Celery/Redis queue.
- RAG still runs on the API request thread. Async document processing requires a running worker; jobs are not processed by uvicorn alone.
- No Redis, Celery, result cache, or pgvector ANN index (no measured justification at current corpus size).
- OIDC is optional and off by default. It is not a production IdP. There is no refresh-token flow, no RP-initiated logout, and no IdP-administered application roles. The OIDC callback returns the same local HS256 JWT as password login. The `/app` UI stores that app JWT in a signed session cookie; that cookie is not an IdP session.
- Invite-only mode is configuration, not a full identity-governance product. Invitation tokens are hashed at rest; the plaintext token is shown once.
- The `/app` UI is local-only: no hosting, no public demo, no OIDC login screen (password JWT only). Agent “inbox” is lookup-by-id because there is no list-workflows API.
- Hybrid lexical rank is FTS `ts_rank_cd`, not BM25.
- RAG `grounded` / citation checks and evidence-term overlap are heuristics, not NLI.
- Offline eval uses HashEmbedder and scripted LLMs; those scores are not MiniLM or Ollama quality.
- Simulated leave execution is not an HR integration.
- Ticketing is a local mock. No Jira/ServiceNow credentials. Tickets are not durable across process restarts. `create_ticket` is listed in MCP but cannot run without API approval plus `TICKETING_ENABLED=true`.
- Default pytest skips isolated PostgreSQL tests unless `TEST_DATABASE_URL` is set.

## License

No `LICENSE` file is in the repository yet. **MIT** is the usual choice for a portfolio backend (short, familiar to reviewers). **Apache-2.0** is reasonable if you want an explicit patent grant. Confirm which to add before a file is committed.
