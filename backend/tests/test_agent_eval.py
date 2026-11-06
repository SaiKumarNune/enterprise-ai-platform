from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_agent_eval import DATASET, run_eval


def test_offline_agent_eval_covers_routing_and_gates() -> None:
    report = run_eval(DATASET, mode="offline_mock")
    assert report["mode"] == "offline_mock"
    assert report["aggregates"]["n_cases"] == 13
    assert set(report["aggregates"]) >= {
        "n_cases",
        "routing_accuracy",
        "status_accuracy",
        "approval_gate_compliance",
        "issue_code_recall",
    }
    by_id = {item["case_id"]: item for item in report["cases"]}
    assert by_id["informational-vacation"]["passed"] is True
    assert by_id["valid-approval-draft"]["observed"]["status"] == "pending_approval"
    assert not by_id["valid-approval-draft"]["observed"]["execution"]
    assert by_id["human-approval-simulated-execute"]["observed"]["execution"]["simulated"] is True
    assert by_id["human-rejection"]["observed"]["approval_decision"] == "rejected"
    assert not by_id["human-rejection"]["observed"]["execution"]
    assert by_id["unauthorized-cross-tenant"]["observed"]["error"] == "document_not_found"
    assert report["passed"] is True
    assert by_id["ticket-prepare-pending"]["observed"]["status"] == "pending_approval"
    assert by_id["ticket-prepare-pending"]["observed"]["action_type"] == "create_ticket"
    assert not by_id["ticket-prepare-pending"]["observed"]["execution"]
    assert by_id["ticket-human-approval"]["observed"]["execution"]["created"] is True
    assert not by_id["ticket-human-rejection"]["observed"]["execution"]
    assert by_id["ticket-unauthorized-execution"]["observed"]["status"] == "pending_approval"
    assert not by_id["ticket-unauthorized-execution"]["observed"]["execution"]
    assert by_id["ticket-disabled-integration"]["observed"]["execution"]["created"] is False
    assert report["failures_by_scenario"] == {}
