# Evaluation

Offline mock evaluation never uses PostgreSQL, Ollama, or MiniLM downloads.

```bash
python eval/run_rag_eval.py
python eval/run_rag_eval.py --output /tmp/rag_eval.json
python eval/run_retrieval_eval.py
python eval/run_agent_eval.py
python eval/run_mcp_eval.py
```

JSON and Markdown are both written when `--output` is set (`.json` also emits `.md` and the reverse).

## Live mode (opt-in)

Live RAG/retrieval requires an isolated database name ending in `_test` (never `enterprise_ai`), a locally cached MiniLM model, and for RAG a reachable Ollama model. Live mode **does not** fall back to HashEmbedder or DatasetLLM.

```bash
export EVAL_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@127.0.0.1:5432/enterprise_ai_acl_test'
python eval/run_rag_eval.py --mode live --output /tmp/rag_live.json
python eval/run_retrieval_eval.py --mode live --output /tmp/retrieval_live.json
# Optional LLM-as-judge (not a CI gate):
python eval/run_rag_eval.py --mode live --llm-judge
```

Hosted OpenAI-compatible generation is a third labeled mode. It is `not_run` unless `EVAL_HOSTED_LLM_BASE_URL` and `EVAL_HOSTED_LLM_MODEL` are set, plus an isolated `*_test` database and cached MiniLM. An API key is optional. Hosted mode never falls back to DatasetLLM or Ollama.

```bash
# Example using a local compatible proxy (no paid vendor required):
export EVAL_HOSTED_LLM_BASE_URL='http://127.0.0.1:11434/v1'
export EVAL_HOSTED_LLM_MODEL='llama3.2:3b'
python eval/run_rag_eval.py --mode hosted --output /tmp/rag_hosted.json
```

If live or hosted dependencies are missing the runner prints `status: not_run` and exits 2.

Agent live Ollama planning is not enabled by default (planner JSON is nondeterministic). MCP evaluation is in-process and does not need PostgreSQL.

Reports keep `mode` labels separate: `offline_mock`, `live`, `hosted_openai_compatible`, or `*_not_run`. Do not mix those aggregates. Failures are grouped by scenario with a heuristic `failure_layer` of `retrieval`, `generation`, or `authorization`.

## Metric definitions

| Metric | Meaning |
|---|---|
| Recall@k | Gold chunk IDs found in the top-k retrieved chunks. Null when a case has no gold IDs. |
| MRR | Reciprocal rank of the first gold chunk. |
| Citation validity rate | Share of answers whose remaining `[S#]` citations map to retrieved sources. |
| Citation precision | Share of cited chunks that are gold evidence. |
| Abstention accuracy | Share of cases where the system abstained iff `expect_abstain` is true. |
| Answer/evidence claim coverage | Deterministic substring overlap with `reference_claims`. Not NLI and not proven faithfulness. |
| LLM-as-judge faithfulness | Optional live-only labeled heuristic. Excluded from default regression gates. |
| Mean latency ms | Wall-clock time in this runner. |

Deterministic regression example: `python eval/run_rag_eval.py --baseline /tmp/prev.json --max-drop recall_at_k=0.15`

Do not treat mock aggregates as MiniLM, Ollama, or hosted-model quality. Mock, live, and hosted reports are separate `mode` values and are not production accuracy.

`rag_quality_v1.json` version 1.2 covers answerable, paraphrase, ambiguous, conflicting, outdated, missing-evidence, and prompt-injection cases. `agent_orchestration_v1.json` version 1.1 adds ticket prepare/approve/reject, unauthorized execution, and disabled integration. `mcp_security_v1.json` version 1.1 adds ticket tool authorization and identity-boundary cases. Retrieval datasets and metrics are unchanged.
