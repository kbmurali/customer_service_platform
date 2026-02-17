from typing import Optional, Dict, Any
import logging
import json
import functools
import inspect

from databases.context_graph_data_access import get_cg_data_access
from agents.security import RBACService
from security.presidio_memory_security import get_presidio_security

logger = logging.getLogger(__name__)

# Initialize RBAC service
rbac_service = RBACService()

# Initialize Presidio security
presidio_security = get_presidio_security()

# ============================================
# Permission Decorator
# ============================================

def require_permissions(resource_type: str, action: str):
    """
    Decorator that enforces dual RBAC permission checks before a tool function executes.

    Checks, in order:
      1. Tool-level permission  — rbac_service.check_tool_permission(user_role, tool_name)
      2. Resource-level permission — rbac_service.check_permission(user_role, resource_type, action)

    On any denial it tracks the event via track_tool_execution_in_cg and returns a
    JSON error string immediately, short-circuiting the wrapped function.

    The tool name is derived automatically from the decorated function's __name__,
    so it never needs to be supplied manually.

    Stacking order with @tool:
        @tool                                      # outermost  — registered last
        @require_permissions("MEMBER", "READ")     # innermost  — runs first at call time

    Args:
        resource_type: RBAC resource identifier (e.g. "MEMBER", "CLAIM").
        action:        RBAC action identifier    (e.g. "READ", "WRITE").

    Usage:
        @tool
        @require_permissions("MEMBER", "READ")
        def member_lookup(member_id: str, user_role: str, session_id: str = "default") -> str:
            ...
    """
    def decorator(func):
        # Resolve the tool name once at decoration time — never at call time.
        tool_name = func.__name__

        # Identify which positional index holds user_role / session_id so we can
        # support both positional and keyword call styles without fragility.
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        def _get(name, args, kwargs):
            """Extract a parameter by name from args or kwargs."""
            if name in kwargs:
                return kwargs[name]
            try:
                return args[param_names.index(name)]
            except (ValueError, IndexError):
                return None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_role  = _get("user_role",  args, kwargs)
            session_id = _get("session_id", args, kwargs) or "default"

            # ── Check 1: Tool-level permission ────────────────────────────────
            if not rbac_service.check_tool_permission(user_role, tool_name):
                error = f"Access denied: insufficient permissions for {tool_name} tool"
                track_tool_execution_in_cg(
                    session_id, tool_name,
                    {"user_role": user_role},
                    status="tool_permission_denied",
                    error=error,
                )
                return json.dumps({"error": error})

            # ── Check 2: Resource-level permission ────────────────────────────
            if not rbac_service.check_permission(user_role, resource_type, action):
                error = f"Access denied: insufficient permissions for {resource_type} resource"
                track_tool_execution_in_cg(
                    session_id, tool_name,
                    {"user_role": user_role, "resource_type": resource_type, "action": action},
                    status="resource_permission_denied",
                    error=error,
                )
                return json.dumps({"error": error})

            return func(*args, **kwargs)

        return wrapper
    return decorator

# Initialize Context Graph data access for tool execution tracking
cg_data_access = get_cg_data_access()

def track_tool_execution_in_cg(
    session_id: str,
    tool_name: str,
    input_data: Dict[str, Any],
    output_data: Optional[Dict[str, Any]] = None,
    status: str = "success",
    execution_time_ms: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """
    Track tool execution in Context Graph using ContextGraphDataAccess.
    
    Args:
        session_id: Session ID
        tool_name: Tool name
        input_data: Tool input
        output_data: Tool output
        status: Execution status (success, failed, permission_denied, not_found)
        execution_time_ms: Execution time in milliseconds
        error: Optional error message
    """
    try:
        cg_data_access.track_tool_execution(
            session_id=session_id,
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            status=status,
            execution_time_ms=execution_time_ms,
            error=error
        )
        logger.debug(f"Tracked tool execution in CG: {tool_name} ({status})")
    except Exception as e:
        logger.warning(f"Failed to track tool execution in CG: {e}")

def scrub_output(text: str, session_id: str) -> str:
    """
    Scrub PII/PHI from tool output using Presidio before returning.
    
    Args:
        text: Output text to scrub
        session_id: Session ID for vault namespace
    
    Returns:
        Scrubbed text
    """
    try:
        scrubbed_text, _, entities_found = presidio_security.scrub_before_storage(
            text,
            namespace=f"tool_output:{session_id}",
            ttl_hours=24
        )
        
        if entities_found:
            logger.info(f"Scrubbed {sum(entities_found.values())} PII/PHI entities from tool output")
        
        return scrubbed_text
    except Exception as e:
        logger.error(f"Presidio scrubbing failed: {e}")
        return text