"""JSON and Markdown evaluation reports plus optional baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(report: dict[str, Any], output: Path | None) -> str:
    text = json.dumps(report, indent=2, default=str)
    if output is None:
        return text
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".md":
        markdown = markdown_report(report)
        output.write_text(markdown, encoding="utf-8")
        json_path = output.with_suffix(".json")
        json_path.write_text(text + "\n", encoding="utf-8")
    else:
        output.write_text(text + "\n", encoding="utf-8")
        markdown_path = output.with_suffix(".md")
        markdown_path.write_text(markdown_report(report), encoding="utf-8")
    return text


def markdown_report(report: dict[str, Any]) -> str:
    aggregates = report.get("aggregates") or {}
    definitions = report.get("metric_definitions") or {}
    lines = [
        f"# {report.get('title') or report.get('dataset') or 'Evaluation'}",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- Dataset version: `{report.get('dataset_version')}`",
        f"- Pass: `{report.get('passed')}`",
        f"- Status: `{report.get('status', 'completed')}`",
    ]
    if report.get("embedder"):
        lines.append(f"- Embedder: `{report['embedder']}`")
    if report.get("llm"):
        lines.append(f"- LLM: `{report['llm']}`")
    if report.get("limitations"):
        lines.append("")
        lines.append("## Limitations")
        lines.append("")
        for item in report["limitations"]:
            lines.append(f"- {item}")
    if definitions:
        lines.extend(["", "## Metrics", "", "| Metric | Value | Definition |", "|---|---|---|"])
        for key, definition in definitions.items():
            lines.append(f"| {key} | {aggregates.get(key)} | {definition} |")
    notes = report.get("notes") or []
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    breakdown = report.get("scenario_breakdown")
    if isinstance(breakdown, dict) and breakdown:
        lines.extend(["", "## Scenario breakdown", "", "| Scenario | n | passed | failed | failure layers |", "|---|---:|---:|---:|---|"])
        for name, bucket in breakdown.items():
            layers = bucket.get("failure_layers") or {}
            layer_text = ", ".join(f"{key}={value}" for key, value in layers.items()) or "—"
            lines.append(
                f"| {name} | {bucket.get('n')} | {bucket.get('passed')} | {bucket.get('failed')} | {layer_text} |"
            )
    grouped_failures = report.get("failures_by_scenario")
    if isinstance(grouped_failures, dict) and grouped_failures:
        lines.extend(["", "## Failures by scenario", ""])
        for scenario, rows in grouped_failures.items():
            lines.append(f"- `{scenario}`")
            for row in rows:
                lines.append(f"  - `{row.get('case_id')}` layer=`{row.get('failure_layer')}`")
    scenarios = report.get("scenarios")
    if isinstance(scenarios, list) and scenarios:
        lines.extend(["", "## Scenarios", ""])
        for item in scenarios:
            lines.append(
                f"- `{item.get('id')}` layer=`{item.get('layer')}` "
                f"status={item.get('status')} rps={item.get('rps')} "
                f"p50_ms={item.get('p50_ms')} p95_ms={item.get('p95_ms')} "
                f"p99_ms={item.get('p99_ms')} error_rate={item.get('error_rate')}"
            )
    nested = report.get("reports")
    if isinstance(nested, list):
        lines.extend(["", "## Nested reports", ""])
        for item in nested:
            lines.append(f"- mode=`{item.get('mode')}` status={item.get('status')}")
    cases = report.get("cases")
    if isinstance(cases, list):
        lines.extend(["", "## Cases", ""])
        for case in cases:
            expected = case.get("expected")
            observed = case.get("observed", case)
            passed = case.get("passed")
            layer = case.get("failure_layer")
            layer_bit = f" layer={layer}" if layer else ""
            lines.append(
                f"- `{case.get('case_id')}` passed={passed}{layer_bit} expected={expected} observed={observed}"
            )
    elif isinstance(cases, dict):
        lines.extend(["", "## Cases", ""])
        for mode, rows in cases.items():
            lines.append(f"### {mode}")
            for case in rows:
                lines.append(f"- `{case.get('case_id')}` {case}")
    lines.append("")
    return "\n".join(lines)


def compare_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_drop: dict[str, float],
) -> list[str]:
    """Compare deterministic aggregate metrics. Does not use LLM-as-judge scores."""
    failures: list[str] = []
    current_agg = current.get("aggregates") or {}
    baseline_agg = baseline.get("aggregates") or {}
    for metric, allowed_drop in max_drop.items():
        if metric in {"llm_judge_faithfulness", "llm_as_judge"}:
            continue
        left = _scalar(current_agg, metric)
        right = _scalar(baseline_agg, metric)
        if left is None or right is None:
            continue
        if left + 1e-9 < right - allowed_drop:
            failures.append(
                f"{metric} dropped from {right} to {left} (allowed drop {allowed_drop})"
            )
    return failures


def _scalar(aggregates: dict[str, Any], metric: str) -> float | None:
    value = aggregates.get(metric)
    if isinstance(value, dict):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
