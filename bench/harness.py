"""Concurrent timing helpers for local benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from bench.metrics import TimingSummary, rss_bytes, summarize

Work = Callable[[], float]


def run_pool(work: Work, *, n: int, concurrency: int) -> tuple[TimingSummary, int]:
    rss_before = rss_bytes()
    latencies: list[float] = []
    errors = 0
    started = time.perf_counter()
    workers = max(1, min(concurrency, n))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work) for _ in range(n)]
        for future in as_completed(futures):
            try:
                latencies.append(float(future.result()))
            except Exception:
                errors += 1
    elapsed = time.perf_counter() - started
    rss_after = rss_bytes()
    return summarize(latencies, errors=errors, elapsed_s=elapsed, n=n), max(0, rss_after - rss_before)


def time_call(work: Work) -> float:
    started = time.perf_counter()
    work()
    return (time.perf_counter() - started) * 1000.0
