# RAG offline evaluation

Synthetic, non-sensitive cases live in `datasets/rag_quality_v1.json`.

```bash
python eval/run_rag_eval.py
python eval/run_rag_eval.py --output /tmp/rag_eval.json
```

Default mode uses fake embeddings and a scripted LLM. It does not use PostgreSQL, Ollama, or MiniLM downloads.

## Metric definitions

| Metric | Meaning |
|---|---|
| Recall@k | Gold chunk IDs found in the top-k retrieved chunks. Null when a case has no gold IDs. |
| Citation validity rate | Share of answers whose remaining `[S#]` citations map to retrieved sources. |
| Abstention accuracy | Share of cases where the system abstained iff `expect_abstain` is true. |
| Mean latency ms | Wall-clock time in this mock runner, not production Ollama latency. |

These are automated heuristics. They are not human graded answer quality and do not prove factual correctness.

Offline mock retrieval uses a lexical hash embedder, so paraphrase Recall@k can collapse unless the question shares tokens with the gold chunk. The scripted LLM cites `[S1]` when any context is selected. Do not treat mock aggregates as MiniLM or Ollama quality.
