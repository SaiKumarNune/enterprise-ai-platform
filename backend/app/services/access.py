from app.models.user import APPROVER_ROLES, User
from app.models.workflow import AgentWorkflow


def can_read_workflow(user: User, workflow: AgentWorkflow) -> bool:
    if user.role in APPROVER_ROLES:
        return True
    if workflow.owner_id is None:
        return False
    return workflow.owner_id == user.id
