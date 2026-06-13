"""
Observability: wraps every LangGraph node to capture latency, token estimates,
input/output snapshots, and persist them to the AgentTrace table.
"""
import time
import logging
import json
from functools import wraps
from typing import Dict, Any

from app.db.models import AgentTrace
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _safe_serialize(obj: Any) -> Any:
    """Convert state values to JSON-serialisable types."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    return str(obj)


def trace_node(node_func):
    """
    Decorator factory: wraps a LangGraph node function.
    Logs node name, latency_ms, estimated token usage, and I/O snapshot to DB.
    """
    @wraps(node_func)
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        output_state: Dict[str, Any] = {}
        raised_exc = None

        try:
            output_state = node_func(state)
        except Exception as exc:
            raised_exc = exc
            output_state = {"error": str(exc)}
        finally:
            latency_ms = (time.perf_counter() - start) * 1000

            # Estimate tokens: ~4 chars per token
            try:
                combined = json.dumps(_safe_serialize(state)) + json.dumps(_safe_serialize(output_state))
                estimated_tokens = len(combined) // 4
            except Exception:
                estimated_tokens = 0

            # Persist trace
            db = SessionLocal()
            try:
                trace = AgentTrace(
                    tenant_id=state.get("tenant_id"),
                    conversation_id=state.get("conversation_id"),
                    message_id=state.get("message_id"),
                    node_name=node_func.__name__,
                    input_json=_safe_serialize(
                        {k: v for k, v in state.items()
                         if k in ("intent", "current_query", "rag_confidence", "validation_passed")}
                    ),
                    output_json=_safe_serialize(
                        {k: v for k, v in output_state.items()
                         if k in ("intent", "final_response", "rag_confidence",
                                  "validation_passed", "error_message")}
                    ),
                    latency_ms=round(latency_ms, 2),
                    tokens_used=estimated_tokens,
                )
                db.add(trace)
                db.commit()
            except Exception as db_exc:
                logger.error(f"Failed to save AgentTrace: {db_exc}")
            finally:
                db.close()

        if raised_exc:
            raise raised_exc

        return output_state

    return wrapper
