import uuid
from datetime import datetime, timezone

from app.models.workflow import AgentWorkflow
from app.services.agent_tools import NOT_PROVIDED, parse_planner_output
from app.services.agent_workflow import approve_agent_workflow
from app.services.agents.action import prepare_leave_draft
from app.services.agents.planner import build_task_plan, run_planner
from app.services.agents.research import run_research, synthesize_informational_answer
from app.services.agents.validator import (
    correct_threshold_wording,
    extract_vacation_policy_rules,
    ground_informational_answer,
    validate_workflow,
)
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import SequenceLLM
from tests.test_agents import PLAN_DRAFT, PLAN_INFO, PLAN_UNKNOWN_TOOL, _session_with_policy
from tests.test_documents import FakeSession

STRICT_VACATION_POLICY = (
    "Employees may request up to 10 days of paid vacation annually. "
    "Vacation requests must be submitted at least 7 days before planned leave. "
    "Requests for 3 or more consecutive days require manager approval. "
    "Employees must specify the number of leave days and the reason. "
    "No leave request may be finalized without explicit manager approval."
)


def test_planner_agent_builds_bounded_unauthorizing_plan() -> None:
    llm = SequenceLLM([PLAN_INFO])
    plan = run_planner(llm, "How many vacation days do employees receive?")
    assert plan["intent"] == "informational"
    assert plan["authorized"] is False
    assert plan["task_plan"]["authorized"] is False
    assert plan["task_plan"]["steps"] == ["research", "validator"]
    assert "execute_approved_action" not in plan["task_plan"]["steps"]


def test_planner_agent_falls_back_on_malformed_and_unknown_tools() -> None:
    malformed = run_planner(SequenceLLM(["not json"]), "How many vacation days?")
    assert malformed["planner_fallback"] is True
    assert malformed["intent"] == "informational"
    unknown = parse_planner_output(PLAN_UNKNOWN_TOOL, "prepare an approval request for three days off")
    plan = build_task_plan(unknown)
    assert plan["planner_fallback"] is True
    assert plan["authorized"] is False
    assert plan["intent"] == "draft_action"


def test_research_agent_returns_retrieved_source_metadata() -> None:
    from app.services.authorization import Principal
    from tests.auth_helpers import own_session_documents, seed_user

    session, document = _session_with_policy()
    user = seed_user(session)
    own_session_documents(session, user)
    result = run_research(
        session,
        HashEmbedder(),
        question="How many vacation days do full-time employees receive?",
        top_k=5,
        document_id=str(document.id),
        principal=Principal.from_user(user),
    )
    assert result["selected_count"] == 1
    source = result["sources"][0]
    assert source["source_id"] == "S1"
    assert source["chunk_id"] == str(session.chunks[0].id)
    assert "similarity" in source
    assert result["excerpts"][0]["content"]


def test_research_agent_does_not_invent_sources_when_empty() -> None:
    from app.services.authorization import Principal
    from tests.auth_helpers import seed_user

    session = FakeSession()
    user = seed_user(session)
    result = run_research(
        session,
        HashEmbedder(),
        question="How many vacation days?",
        top_k=5,
        document_id=None,
        principal=Principal.from_user(user),
    )
    assert result["sources"] == []
    assert result["excerpts"] == []
    assert result["retrieval_count"] == 0


def test_validator_agent_flags_invalid_citations_without_claiming_facts() -> None:
    result = validate_workflow(
        {
            "intent": "informational",
            "request": "How many vacation days?",
            "answer": "Employees receive 15 days [S9].",
            "sources": [{"source_id": "S1"}],
            "citation_valid": True,
        }
    )
    assert result["validation"]["passed"] is False
    assert result["validation"]["semantic_verification"] is False
    assert any(issue["code"] == "unknown_citation" for issue in result["validation"]["issues"])
    assert result["citation_valid"] is False


def test_validator_agent_flags_missing_required_days() -> None:
    result = validate_workflow(
        {
            "intent": "draft_action",
            "request": "Prepare an approval request for time off.",
            "draft": None,
            "excerpts": [{"source_id": "S1", "content": "Employees receive 15 vacation days."}],
            "sources": [{"source_id": "S1"}],
        }
    )
    assert result["validation"]["passed"] is False
    assert any(issue["code"] == "missing_required_days" for issue in result["validation"]["issues"])


def test_validator_agent_flags_days_exceeding_policy() -> None:
    result = validate_workflow(
        {
            "intent": "draft_action",
            "request": "Prepare an approval request for 25 vacation days off.",
            "draft": {"days": 25, "reason": NOT_PROVIDED, "source_ids": ["S1"]},
            "excerpts": [{"source_id": "S1", "content": "Full-time employees receive 15 vacation days."}],
            "sources": [{"source_id": "S1"}],
            "citation_valid": True,
        }
    )
    assert result["validation"]["passed"] is False
    assert any(issue["code"] == "days_exceed_policy" for issue in result["validation"]["issues"])


def test_policy_parser_reads_up_to_n_days_and_ignores_notice_windows() -> None:
    rules = extract_vacation_policy_rules([{"content": STRICT_VACATION_POLICY}])
    assert rules["annual_days"] == 10
    assert rules["advance_notice_days"] == 7
    assert rules["consecutive_manager_threshold"] == 3
    assert rules["reason_required"] is True
    assert rules["dates_required"] is True
    assert rules["balance_verified"] is False
    assert rules["late_submission_requires_approval"] is False


def test_validator_flags_25_days_against_10_day_allowance() -> None:
    result = validate_workflow(
        {
            "intent": "draft_action",
            "request": "Review the vacation policy and prepare an approval request for 25 days off.",
            "draft": {"days": 25, "reason": NOT_PROVIDED, "start_date": NOT_PROVIDED, "source_ids": ["S1"]},
            "excerpts": [{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
            "sources": [{"source_id": "S1"}],
            "citation_valid": True,
        }
    )
    codes = {issue["code"] for issue in result["validation"]["issues"]}
    assert result["validation"]["passed"] is False
    assert "days_exceed_policy" in codes
    assert "missing_reason" in codes
    assert "missing_dates" in codes


def test_validator_flags_three_day_request_missing_required_details() -> None:
    result = validate_workflow(
        {
            "intent": "draft_action",
            "request": "Prepare an approval request for three days off.",
            "draft": {
                "days": 3,
                "reason": NOT_PROVIDED,
                "start_date": NOT_PROVIDED,
                "end_date": NOT_PROVIDED,
                "source_ids": ["S1"],
            },
            "excerpts": [{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
            "sources": [{"source_id": "S1"}],
        }
    )
    codes = {issue["code"] for issue in result["validation"]["issues"]}
    assert result["validation"]["passed"] is False
    assert "missing_reason" in codes
    assert "missing_dates" in codes
    assert "days_exceed_policy" not in codes


def test_validator_fails_closed_when_annual_limit_is_ambiguous() -> None:
    result = validate_workflow(
        {
            "intent": "draft_action",
            "request": "Prepare an approval request for three days off.",
            "draft": {"days": 3, "reason": "surgery recovery", "start_date": "2026-11-10", "source_ids": ["S1"]},
            "excerpts": [{"source_id": "S1", "content": "Employees may take vacation with manager approval."}],
            "sources": [{"source_id": "S1"}],
        }
    )
    assert result["validation"]["passed"] is False
    assert any(issue["code"] == "policy_rule_unverified" for issue in result["validation"]["issues"])


def test_validator_accepts_complete_leave_request_within_allowance() -> None:
    result = validate_workflow(
        {
            "intent": "draft_action",
            "request": (
                "Prepare an approval request for three days off because surgery recovery "
                "from 2026-11-10 to 2026-11-12."
            ),
            "draft": {
                "days": 3,
                "reason": "surgery recovery",
                "start_date": "2026-11-10",
                "end_date": "2026-11-12",
                "source_ids": ["S1"],
            },
            "excerpts": [{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
            "sources": [{"source_id": "S1"}],
            "citation_valid": True,
        }
    )
    codes = {issue["code"] for issue in result["validation"]["issues"]}
    assert result["validation"]["passed"] is True
    assert "days_exceed_policy" not in codes
    assert "employee_balance_unknown" in codes
    assert result["validation"]["semantic_verification"] is False


def test_late_submission_is_not_an_approval_trigger() -> None:
    answer = (
        "Vacation requests require manager approval for requests of 3 or more consecutive days "
        "and for all requests that are not submitted at least 7 days in advance [S1]."
    )
    result = ground_informational_answer(
        answer,
        [{"content": STRICT_VACATION_POLICY}],
    )
    lowered = result["answer"].lower()
    assert result["abstain"] is False
    assert "3 or more consecutive days" in result["answer"]
    assert "not submitted" not in lowered
    assert "in advance" not in lowered
    assert "[S1]" in result["answer"]
    assert any(issue["code"] == "unsupported_approval_trigger" for issue in result["issues"])
    check = validate_workflow(
        {
            "intent": "informational",
            "request": "When do vacation requests require manager approval?",
            "answer": answer,
            "excerpts": [{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
            "sources": [{"source_id": "S1"}],
            "citation_valid": True,
        }
    )
    assert "not submitted" not in check["answer"].lower()
    assert check["validation"]["semantic_verification"] is False


def test_genuine_late_submission_approval_rule_is_preserved() -> None:
    excerpts = [
        {
            "content": (
                "Requests submitted fewer than 7 days before planned leave require manager approval. "
                "Requests for 3 or more consecutive days require manager approval."
            )
        }
    ]
    answer = (
        "Vacation requests require manager approval for requests of 3 or more consecutive days "
        "and for all requests that are not submitted at least 7 days in advance [S1]."
    )
    result = ground_informational_answer(answer, excerpts)
    assert result["abstain"] is False
    assert "not submitted" in result["answer"].lower()
    rules = extract_vacation_policy_rules(excerpts)
    assert rules["late_submission_requires_approval"] is True


def test_threshold_wording_corrects_exceeding_three_days() -> None:
    answer = "Requests exceeding 3 days require manager approval [S1]."
    excerpts = [{"content": "Requests for 3 or more consecutive days require manager approval."}]
    corrected, changed = correct_threshold_wording(answer, excerpts)
    assert changed is True
    assert "3 or more consecutive days" in corrected
    assert "exceeding 3 days" not in corrected.lower()


def test_research_synthesis_corrects_exceeding_threshold() -> None:
    llm = SequenceLLM(["Requests exceeding 3 days require manager approval [S1]."])
    result = synthesize_informational_answer(
        llm,
        question="When is manager approval required?",
        excerpts=[{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
        sources=[{"source_id": "S1"}],
    )
    assert "3 or more consecutive days" in result["answer"]
    assert "exceeding 3" not in result["answer"].lower()


def test_stale_pending_draft_is_revalidated_on_approve() -> None:
    now = datetime.now(timezone.utc)
    workflow = AgentWorkflow(
        id=uuid.uuid4(),
        request="Review the vacation policy and prepare an approval request for 25 days off.",
        status="pending_approval",
        intent="draft_action",
        draft={"days": 25, "reason": NOT_PROVIDED, "start_date": NOT_PROVIDED, "source_ids": ["S1"]},
        sources=[{"source_id": "S1", "document_id": str(uuid.uuid4()), "chunk_id": str(uuid.uuid4()), "chunk_index": 0, "page_number": 1, "similarity": 0.9}],
        retrieval_count=1,
        created_at=now,
        updated_at=now,
        graph_state={
            "request": "Review the vacation policy and prepare an approval request for 25 days off.",
            "intent": "draft_action",
            "excerpts": [{"source_id": "S1", "content": STRICT_VACATION_POLICY}],
            "sources": [{"source_id": "S1"}],
            "draft": {"days": 25, "reason": NOT_PROVIDED, "start_date": NOT_PROVIDED, "source_ids": ["S1"]},
            "status": "pending_approval",
        },
    )
    from app.services.authorization import Principal
    from tests.auth_helpers import seed_user

    session = FakeSession()
    actor = seed_user(session, role="manager")
    workflow.owner_id = actor.id
    workflow.tenant_id = actor.tenant_id
    session.add(workflow)
    result = approve_agent_workflow(
        session,
        HashEmbedder(),
        SequenceLLM(["unused"]),
        workflow.id,
        actor_id=actor.id,
        principal=Principal.from_user(actor),
    )
    assert result.status == "needs_clarification"
    assert result.execution_result is None
    assert result.approval_decision is None


def test_action_agent_prepare_does_not_invent_personal_details() -> None:
    llm = SequenceLLM(
        [
            '{"reason":"personal reasons","start_date":"2026-11-03","days":3,"source_ids":["S1"]}'
        ]
    )
    draft = prepare_leave_draft(
        llm,
        request="Prepare an approval request for three days off.",
        excerpts=[{"source_id": "S1", "content": "Employees receive 15 vacation days."}],
        sources=[{"source_id": "S1", "document_id": "d", "chunk_id": "c", "page_number": 1}],
        action_type="leave_request",
    )
    assert draft is not None
    assert draft["days"] == 3
    assert draft["reason"] == NOT_PROVIDED
    assert draft["start_date"] == NOT_PROVIDED
