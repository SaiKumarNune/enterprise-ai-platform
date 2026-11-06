# Section 8 final audit

Local verification of `enterprise-ai-platform` after Sections 1–7. This is not a production certification, not a hosted demo, and not a claim of model accuracy.

Recorded **2026-11-06** on this machine.

## Verified capabilities

- FastAPI API plus local `/app` UI; JWT/RBAC; optional OIDC + invite-only registration.
- Tenant isolation, document ACL, quarantine of unowned rows, SQL filters before ranking.
- Hybrid retrieval (PostgreSQL FTS `ts_rank_cd` + pgvector cosine + RRF).
- Citation-grounded RAG with evidence-sufficiency abstention; local Ollama optional.
- LangGraph leave-request and mock ticket drafts; manager/admin same-tenant approval; simulated execution only.
- In-process MCP: identity from `Principal`; `execute_approved_action` is not an MCP tool; `create_ticket` is gated.
- PostgreSQL document jobs with `SKIP LOCKED`, lease heartbeat, owner-only complete/fail.
- Offline eval (RAG/agent/MCP/retrieval) and mock bench with labeled layers.

## Test and eval results

| Check | Result |
|---|---|
| Isolated pytest (`enterprise_ai_acl_test`) | **292 passed, 0 skipped**, 13.88s |
| RAG eval `offline_mock` | **passed=true**, n=11, dataset v1.2 |
| Retrieval eval `offline_mock_hash_embedder` | **passed=true** |
| Agent eval `offline_mock` | **passed=true**, n=13, dataset v1.1 |
| MCP eval `offline_inprocess_mcp` | **passed=true**, n=12, dataset v1.1 |
| Bench `--mode mock` | **completed**; `http_mock` / `worker_mock` / `in_memory` |
| Fresh migrate `enterprise_ai_migrate_test` | linear history to **`e3f4a5b6c7d8`** |

Development database `enterprise_ai` was not used for tests, eval, bench, or this migrate run.

## Security boundaries reviewed

- JWT role/tenant reloaded from PostgreSQL; IdP claims do not set RBAC.
- Cross-tenant reads/approvals return 404.
- MCP argument guard rejects identity/authorization keys.
- Ticket create requires API approval (`execution_authorized`) and `TICKETING_ENABLED`.
- Job succeed/fail requires the claiming `worker_id`; stale workers cannot finish another worker’s lease.

## Changes in this section

- Call `FastMCPSettings.model_rebuild()` so pydantic-settings no longer warns on the MCP `lifespan` forward reference.
- Ignore `saikumar-portfolio/` in `.gitignore`.
- Clarify README: OIDC has no IdP session/refresh flow; `/app` stores the app JWT in a signed cookie.
- Record this audit.

No production RAG, retrieval, agent graph, or MCP tool semantics were changed. No commit or push. No deploy.

## Unresolved limitations

- Not deployed. No public URL.
- Mock eval/bench scores are HashEmbedder/scripted LLM and laptop timings, not MiniLM/Ollama/hosted quality and not production SLOs.
- Tickets are in-process mocks; not durable; not Jira/ServiceNow.
- Leave execution is simulated.
- OIDC is optional and incomplete as an identity product (no refresh tokens, no RP logout).
- UI has no OIDC login screen and no workflow inbox list.
- Lexical rank is `ts_rank_cd`, not BM25. No ANN index.
- Live/hosted eval and live bench were not run in this audit.
- Uncommitted Section 1–7 work remains in the working tree; nothing was committed here.
