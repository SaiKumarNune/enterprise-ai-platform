# Local performance benchmarks

Bounded, reproducible measurements for search, RAG, retrieval, and document-processing jobs. This is not a cloud load test and does not start Redis, Celery, or ANN indexes.

## Safety

- Default **mock** mode never opens PostgreSQL or Ollama.
- **database** and **live** require `BENCH_DATABASE_URL`, `EVAL_DATABASE_URL`, or `TEST_DATABASE_URL` whose database name ends in `_test`.
- The runner refuses `enterprise_ai` and other development/system database names.
- `isolated_session_factory` may drop/recreate tables **inside that isolated database only**.

Recommended dedicated database: `enterprise_ai_bench_test`.

## Modes

| Mode | Embedder | LLM | PostgreSQL |
|---|---|---|---|
| `mock` | HashEmbedder | ScriptedLLM | no |
| `database` | HashEmbedder | ScriptedLLM | isolated `*_test` |
| `live` | MiniLM (must already be cached) | Ollama only in `rag_ollama` | isolated `*_test` |
| `auto` | runs mock, then database/live if configured | mixed; compare only matching `layer` labels | |

Do not mix `http_mock`, `sql_database`, and `http_live_ollama` numbers as if they were the same experiment.

## Reproduce

```bash
python bench/run_bench.py --mode mock --requests 40 --concurrency 4 --jobs 8
python bench/run_bench.py --mode mock --output /tmp/bench_mock.json
```

Database-only (HashEmbedder, no Ollama):

```bash
# Create an isolated DB once. Never point this at enterprise_ai.
PYTHONPATH=backend python -c "from tests.postgres_isolation import ensure_isolated_database; ensure_isolated_database('enterprise_ai_bench_test')"
export BENCH_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@127.0.0.1:5432/enterprise_ai_bench_test'
python bench/run_bench.py --mode database --requests 40 --concurrency 4 --jobs 8 --output /tmp/bench_db.json
```

Live MiniLM / optional Ollama (`--mode live` exits 2 when MiniLM or the isolated DB is missing; Ollama absence only skips `rag_ollama`):

```bash
python bench/run_bench.py --mode live --requests 20 --concurrency 4 --live-rag-requests 5 --output /tmp/bench_live.json
```

`--requests`, `--concurrency`, `--jobs`, and `--live-rag-requests` cap work. Keep them small on a laptop.

## Metrics

Each completed scenario reports `rps`, `success_rps`, `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms`, `error_rate`, and `elapsed_s`. Job scenarios also report `job_throughput_per_s`, `jobs_enqueued`, `jobs_succeeded`, and `queue_drain_s`. `rss_delta_bytes` is a coarse `ru_maxrss` delta, not container memory.
