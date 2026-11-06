"""Latency and throughput summaries. No external stats libraries."""

from __future__ import annotations

import math
import resource
import sys
from dataclasses import dataclass


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[low], 3)
    weight = rank - low
    return round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 3)


def rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


@dataclass(frozen=True)
class TimingSummary:
    n: int
    errors: int
    elapsed_s: float
    rps: float
    success_rps: float
    error_rate: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    mean_ms: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "errors": self.errors,
            "elapsed_s": round(self.elapsed_s, 4),
            "rps": round(self.rps, 3),
            "success_rps": round(self.success_rps, 3),
            "error_rate": round(self.error_rate, 4),
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
        }


def summarize(latencies_ms: list[float], *, errors: int, elapsed_s: float, n: int) -> TimingSummary:
    elapsed_s = max(elapsed_s, 1e-9)
    successes = len(latencies_ms)
    mean_ms = round(sum(latencies_ms) / successes, 3) if successes else None
    return TimingSummary(
        n=n,
        errors=errors,
        elapsed_s=elapsed_s,
        rps=n / elapsed_s,
        success_rps=successes / elapsed_s,
        error_rate=errors / n if n else 0.0,
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        mean_ms=mean_ms,
    )
