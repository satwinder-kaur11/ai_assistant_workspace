import logging
from sqlalchemy.orm import Session
from app.db.models import ActionLog, ActionStatus, Task, TaskSource

logger = logging.getLogger(__name__)

def get_pending_actions(db: Session, tenant_id: int, user_id: int):
    """Returns all pending actions for a user."""
    return db.query(ActionLog).filter(
        ActionLog.tenant_id == tenant_id,
        ActionLog.user_id == user_id,
        ActionLog.status == ActionStatus.pending_approval
    ).all()

def approve_action(db: Session, action_log_id: int) -> bool:
    """Approves an action and executes its side effects."""
    action_log = db.query(ActionLog).filter(ActionLog.id == action_log_id).first()
    if not action_log or action_log.status != ActionStatus.pending_approval:
        return False
        
    try:
        if action_log.action_type == "create_tasks":
            tasks_data = action_log.payload_json.get("tasks", [])
            for t_data in tasks_data:
                task = Task(
                    tenant_id=action_log.tenant_id,
                    user_id=action_log.user_id,
                    title=t_data.get("title", "Untitled"),
                    description=t_data.get("description", ""),
                    priority=t_data.get("priority", "Medium"),
                    owner=t_data.get("owner", "Unassigned"),
                    status="pending",
                    source=TaskSource.ai_generated
                )
                db.add(task)
                
        elif action_log.action_type == "draft_email":
            # Real execution would send the email here.
            logger.info(f"Email approved and 'sent': {action_log.payload_json}")
            
        action_log.status = ActionStatus.executed
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error approving action {action_log_id}: {e}")
        db.rollback()
        return False

def reject_action(db: Session, action_log_id: int) -> bool:
    """Rejects a pending action."""
    action_log = db.query(ActionLog).filter(ActionLog.id == action_log_id).first()
    if not action_log or action_log.status != ActionStatus.pending_approval:
        return False
        
    action_log.status = ActionStatus.rejected
    db.commit()
    return True
