from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_mcp_eval import DATASET, run_eval


def test_offline_mcp_eval_uses_mcp_path_and_rejects_identity() -> None:
    report = run_eval(DATASET)
    assert report["mode"] == "offline_inprocess_mcp"
    assert "execute_approved_action" not in report["aggregates"]["exposed_tools"]
    by_id = {item["case_id"]: item for item in report["cases"]}
    assert by_id["search-success"]["passed"] is True
    assert by_id["reject-identity-fields"]["passed"] is True
    assert by_id["cross-tenant-acl"]["passed"] is True
    assert by_id["unknown-execute-tool"]["passed"] is True
    assert report["passed"] is True
