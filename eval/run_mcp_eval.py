#!/usr/bin/env python3
"""MCP tool integration evaluation using the in-process client/server path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402
from mcp.types import TextContent  # noqa: E402

from app.mcp.client import invoke_knowledge_search, invoke_prepare_approval_request  # noqa: E402
from app.mcp.context import McpExecutionContext  # noqa: E402
from app.mcp.errors import McpToolError  # noqa: E402
from app.mcp.server import EXPOSED_TOOLS, build_mcp_server  # noqa: E402
from app.services.authorization import Principal  # noqa: E402
from eval.reporting import write_report  # noqa: E402
from tests.auth_helpers import own_session_documents, seed_tenant, seed_user  # noqa: E402
from tests.fake_embeddings import HashEmbedder  # noqa: E402
from tests.test_agents import _session_with_policy  # noqa: E402

DATASET = Path(__file__).resolve().parent / "datasets" / "mcp_security_v1.json"


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(dataset_path: Path) -> dict:
    dataset = load_dataset(dataset_path)
    session, document = _session_with_policy()
    owner = seed_user(session)
    own_session_documents(session, owner)
    embedder = HashEmbedder()
    results = []
    for case in dataset["cases"]:
        results.append(_run_case(case, session, document, owner, embedder))
    listed = _list_tools(McpExecutionContext(principal=Principal.from_user(owner), db=session, embedder=embedder))
    passed = all(item["passed"] for item in results) and set(listed) == EXPOSED_TOOLS
    return {
        "title": "MCP evaluation",
        "dataset": dataset_path.name,
        "dataset_version": dataset.get("version"),
        "mode": "offline_inprocess_mcp",
        "status": "completed",
        "passed": passed,
        "metric_definitions": {
            "tool_success_rate": "Share of cases whose success/error matched expect_success.",
            "schema_rejection_rate": "Share of schema/security cases that returned isError.",
            "exposed_tools": "MCP list_tools names; execute_approved_action must be absent.",
        },
        "aggregates": {
            "n_cases": len(results),
            "tool_success_rate": round(sum(1 for item in results if item["passed"]) / len(results), 4),
            "exposed_tools": listed,
        },
        "cases": results,
        "limitations": [
            "Uses the official MCP in-memory transport against build_mcp_server; not a network MCP port.",
            "Authorization is the bound Principal, not model-supplied identity fields.",
        ],
        "notes": ["These cases call ClientSession.call_tool / invoke_* helpers, not knowledge_search() directly."],
    }


def _run_case(case: dict, session, document, owner, embedder) -> dict:
    principal_user = owner
    if case.get("inactive"):
        principal_user = seed_user(session, is_active=False, tenant_id=owner.tenant_id)
    if case.get("cross_tenant"):
        principal_user = seed_user(session, tenant_id=seed_tenant(session, name="OtherMCP").id)
    context = McpExecutionContext(
        principal=Principal.from_user(principal_user),
        db=session,
        embedder=embedder,
    )
    expected_success = bool(case.get("expect_success"))
    error_text = None
    success = False
    payload = None
    tool = case["tool"]
    try:
        if "arguments" in case:
            result = _raw_call(context, tool, case["arguments"])
            success = not result.isError
            if result.isError and result.content and isinstance(result.content[0], TextContent):
                error_text = result.content[0].text
            elif not result.isError:
                payload = result.structuredContent
        elif tool == "knowledge_search":
            payload = invoke_knowledge_search(
                db=session,
                embedder=embedder,
                principal=Principal.from_user(principal_user),
                question=case.get("question") or "vacation",
                top_k=5,
                document_id=document.id,
            )
            success = True
        elif tool == "prepare_approval_request":
            payload = invoke_prepare_approval_request(
                principal=Principal.from_user(principal_user),
                request=case["request"],
                raw_draft=case["raw_draft"],
                valid_source_ids={"S1"},
                action_type="leave_request",
            )
            success = payload is not None
        else:
            result = _raw_call(context, tool, case.get("arguments") or {})
            success = not result.isError
            if result.isError and result.content:
                error_text = result.content[0].text
    except McpToolError as exc:
        success = False
        error_text = exc.message

    passed = success == expected_success
    needle = case.get("expect_error_substring")
    if needle:
        passed = (not success) and needle.lower() in (error_text or "").lower()
    if case.get("expect_selected"):
        passed = passed and bool((payload or {}).get("selected_count"))
    if case.get("expect_requires_approval"):
        passed = passed and bool((payload or {}).get("requires_approval"))
    if case.get("forbid_execution_fields"):
        passed = passed and "simulated" not in (payload or {})
    return {
        "case_id": case["id"],
        "type": case["type"],
        "expected": {"success": expected_success, "error_substring": needle},
        "observed": {"success": success, "error": error_text, "payload_keys": list((payload or {}).keys())},
        "passed": passed,
    }


def _raw_call(context: McpExecutionContext, name: str, arguments: dict):
    async def _run():
        server = build_mcp_server(context)
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(_run())


def _list_tools(context: McpExecutionContext) -> list[str]:
    async def _run():
        server = build_mcp_server(context)
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            return [tool.name for tool in listed.tools]

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate in-process MCP tools.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_eval(args.dataset)
    print(write_report(report, args.output))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
