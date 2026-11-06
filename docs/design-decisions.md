# Design decisions

This document records why the platform is built the way it is. It is not a product roadmap.

## Reciprocal Rank Fusion instead of BM25

Lexical ranking uses PostgreSQL `english` full-text search (`plainto_tsquery` + `ts_rank_cd`) on a generated `tsvector` column with a GIN index. Hybrid search fuses that list with pgvector cosine neighbors using Reciprocal Rank Fusion (RRF, default `k=60`).

PostgreSQL `ts_rank` / `ts_rank_cd` is **not BM25**. It does not implement Robertson–Spärck Jones IDF or BM25 term-frequency saturation. True BM25 would require an extra extension (for example ParadeDB `pg_search`) or an in-process scorer.

RRF was chosen because:

- Both legs already return ranked lists. RRF needs only ranks, not calibrated scores.
- Tenant, owner, and ACL predicates stay in SQL **before** `ORDER BY` / `LIMIT` on each leg. An in-process BM25 pass over a candidate dump would either leak unauthorized rows into Python or duplicate authorization logic.
- Exact cosine search plus FTS is enough for local document volumes. No HNSW/IVFFlat index is created yet; ANN would be a later, measured change.

## In-process MCP

Research and Action call `knowledge_search`, `prepare_approval_request`, and `prepare_ticket` through the official Python MCP SDK (`FastMCP` + in-memory session). There is no extra Docker service, TCP port, or public MCP route.

Identity is bound server-side from the JWT `Principal`. Tool arguments cannot set `user_id`, `tenant_id`, `role`, `authorized`, `requester_user_id`, `idempotency_key`, or `workflow_id`. `execute_approved_action` is **not** an MCP tool; only `POST /agents/{id}/approve` can authorize simulated leave execution after a row lock.

`create_ticket` is listed as an MCP tool so the approval path can call it through the same SDK, but the tool refuses unless `McpExecutionContext.execution_authorized` is set. Only the Python helper `invoke_create_ticket` (used after API approve) sets that flag. The LLM cannot pass it as an argument. Tenant and requester IDs always come from the bound principal. Idempotency is `(tenant_id, workflow_id)` inside an in-process mock adapter. `TICKETING_ENABLED` defaults to false; when it is false, approve records `created=false` and does not invent a vendor call.

This is a local, in-process control-plane pattern. It is not a multi-host MCP deployment, not Jira/ServiceNow, and should not be described as one.

## Tenant isolation

Every authenticated request reloads the user from PostgreSQL. JWT claims are not trusted for role or tenant. Admins are tenant-scoped. Cross-tenant reads and approvals return **404**, not 403, to avoid confirming that another tenant’s resource exists.

Document access is owner, ACL grant, or (for assigned documents) tenant admin. Rows with `owner_id IS NULL` are quarantined: hidden from ordinary document, search, RAG, and agent paths until a tenant admin assigns an owner. Retrieval joins `documents` and applies that predicate in SQL before ranking.

Public self-registration always creates an **employee** in `DEFAULT_TENANT_ID` unless an invitation token is presented (then the user joins the inviting admin’s tenant). Privileged roles are not settable through public HTTP APIs. `AUTH_INVITE_ONLY=true` rejects registration and first-time OIDC provisioning without a valid, unexpired, email-matching invite.

## Optional OIDC

OIDC is an optional authorization-code login (`GET /auth/oidc/login` → IdP → `GET /auth/oidc/callback`). It uses PKCE S256, a one-time server-side `state` (HMAC), and a `nonce` bound into the ID token. The ID token is verified with **RS256** only: issuer, audience (`OIDC_CLIENT_ID`), signature (JWKS from discovery), `exp`/`iat`, and nonce. `alg=none` and HMAC-with-IdP-secret confusion are rejected by pinning RS256.

Verified identities map to local users **only** by `(issuer, subject)` in `external_identities`. Matching email is **not** an automatic account-link: an OIDC subject is never attached to an existing password-based (or any existing) local user just because the emails match. That case returns **409**. A new local employee is created only when the email is unused. IdP `role`, `groups`, or `realm_access` claims are never copied onto `users.role`. Existing `(issuer, subject)` mappings stay bound to the original `user_id` and tenant; a later email or invite cannot steal the identity.

Invitation tokens expire, are hashed at rest, and are redeemed with a row lock plus `UPDATE … WHERE consumed_at IS NULL` so concurrent requests cannot use the same invite twice.

Password JWT login remains for local development. OIDC-only users may have a null `password_hash` and cannot use password login.

This is not a hosted IdP, not SSO for third-party apps, and not a substitute for tenant RBAC in PostgreSQL. Discovery and JWKS fetches require network access to the configured issuer; tests inject an in-process fake provider.

## Human approval gates

LangGraph orchestrates planner → research → (action draft) → validator in-process. PostgreSQL `agent_workflows` is the durable source of truth. An in-memory checkpointer is not used for authorization.

Leave-request drafts stay `pending_approval` until a **manager or admin in the same tenant** approves or rejects. Approve takes a row lock, re-validates the draft against retrieved policy, and only then sets `authorized=True` for simulated execution. Duplicate approve/reject returns **409**. Simulated execution sets `execution_result.simulated=true` and does not send email, mutate HR systems, or change documents.

`create_ticket` drafts use the same approve/reject gate and the same tenant-scoped manager/admin check. Ticket field validation does not require leave-policy excerpts. Leave-request validation is unchanged.

Stale drafts that fail current validation move to `needs_clarification` and are not executed.

## Local-model tradeoffs

Default generation is Ollama `llama3.2:3b`. Embeddings are `all-MiniLM-L6-v2` (384 dimensions) on CPU. Tests and default eval runners never download those models; they use `HashEmbedder` and scripted LLMs.

| Choice | Benefit | Cost |
|---|---|---|
| Local Ollama | No paid API key; data stays on the machine | Quality and latency are hardware-bound; daemon must be running |
| MiniLM | Fast, small, deterministic dimension | Weaker than larger embedders on paraphrase |
| HashEmbedder in tests/eval | CI without downloads | Mock retrieval scores are not MiniLM quality |
| DatasetLLM / SequenceLLM | Deterministic RAG and agent tests | Not a measure of llama3.2 faithfulness |
| Evidence-sufficiency filter | Abstain when retrieved text does not share enough content terms with the question | Lexical overlap, not NLI; paraphrases with no shared terms can over-abstain |

Live eval (`--mode live`) requires an isolated `*_test` database, a cached MiniLM, and a reachable Ollama model. It never falls back to mocks. Hosted OpenAI-compatible eval (`--mode hosted`) is optional configuration (`EVAL_HOSTED_LLM_*`) and is `not_run` unless those variables plus an isolated database are set. Mock, live, and hosted reports stay separately labeled and are not production accuracy.

## Document processing jobs

`POST /documents/{id}/process` remains synchronous unless the client sends `wait=false`. Async work is a row in `document_processing_jobs`, claimed with `FOR UPDATE SKIP LOCKED`, retried with a lease and `max_attempts`, and executed by `python -m app.worker`. Redis/Celery are not used. Caching and pgvector ANN indexes are not added; there is no benchmark showing they help at current corpus size.

## What this platform is not

- Not a deployed or multi-tenant SaaS.
- Not an HR, hosted IdP, or ticketing production integration. Optional OIDC is a local callback against a configured issuer. Mock tickets exist only in process memory.
- Not BM25, not a public MCP server, and not a guarantee of factual RAG answers.
- Citation validity and `grounded` flags are heuristics.
