from typing import Optional, Dict, Any
import logging
import json
import functools
import inspect
from datetime import datetime

from langchain_core.tools import tool

from databases.context_graph_data_access import get_cg_data_access
from databases.chroma_vector_data_access import get_chroma_data_access
from agents.security import RBACService, RateLimiter, RateLimitError
from security.approval_workflow import get_approval_workflow, CircuitBreakerError
from security.presidio_memory_security import get_presidio_security
from security.nh3_sanitization import sanitize_text
from observability.prometheus_metrics import rate_limit_checks, track_rate_limit_exceeded

logger = logging.getLogger(__name__)

# Initialize RBAC service
rbac_service = RBACService()

# Initialize Presidio security
presidio_security = get_presidio_security()

# Initialize Rate Limiter (Control 6)
rate_limiter = RateLimiter()

# Initialize Chroma vector data access for semantic search
chroma_data_access = get_chroma_data_access()

# Initialize Context Graph data access for tool execution tracking
cg_data_access = get_cg_data_access()

# ============================================
# Permission Decorators
# ============================================
def require_approvals(
    action: str,
    record_name: str,
    record_id_arg: str,
    record_field_arg: Optional[str] = None,
    changed_value_arg: Optional[str] = None,
):
    """
    Control 5: Decorator that gates a tool behind the human-in-the-loop
    approval workflow.

    The decorator automatically extracts user_id, user_role, reason, and
    the record identifier from the wrapped function's keyword arguments,
    builds the action_description and details dict, and submits the
    approval request via ApprovalWorkflow.submit_approval_request().

    If the request is not approved the tool returns the approval result
    as a JSON string and the wrapped function body is **not** executed.

    Args:
        action: Human-readable verb label, e.g. "Update" or "Deny".
        record_name: Human-readable noun, e.g. "claim" or "prior authorization".
        record_id_arg: Name of the function kwarg that holds the record ID
                       (e.g. "claim_id" or "pa_id").
        record_field_arg: Optional name of the function kwarg that holds the
                          field being changed (e.g. "field").
        changed_value_arg: Optional name of the function kwarg that holds the
                           new value (e.g. "new_status" or "new_value").
    
    Stacking order with @tool:
        @tool                                      # outermost  — registered last
        @circuit_breaker                           # second - should check breaker first
        @require_approvals(...)                    # immediately after circuit breaker
        @require_rate_limits                       # middle - should run before permissions
        @require_permissions("MEMBER", "READ")     # innermost  — runs first at call time

    Usage:
        @tool
        @circuit_breaker
        @require_approvals( action="Update", record_name="Member", record_id_arg="member_id", record_field_arg="field",  changed_value_arg="new_value" )
        @require_rate_limits
        @require_permissions("MEMBER", "UPDATE")
        def update_member_info( member_id: str, field: str, new_value: str, reason: str, ...) -> str:
            ...
    """

    def decorator(func):
        # Resolve the tool name once at decoration time — never at call time.
        tool_name = func.__name__
        
        # Identify which positional index holds user_role, user_id, session_id.
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
            # --- Extract values from the tool's kwargs ---
            user_role  = _get("user_role",  args, kwargs) or "unknown"
            user_id = _get("user_role",  args, kwargs) or ""
            session_id = _get("session_id", args, kwargs) or "default"
            reason = _get("reason",  args, kwargs) or ""

            extracted_record_id  = _get( record_id_arg,  args, kwargs) or "unknown"

            extracted_record_field = _get( record_field_arg,  args, kwargs) if record_field_arg else None
            
            changed_value = _get( changed_value_arg,  args, kwargs) if changed_value_arg else None

            # --- Build action_description dynamically ---
            if record_field_arg is None and changed_value_arg is None:
                action_description = f"{action} {record_name} {extracted_record_id}. Reason: {reason}"
            elif changed_value_arg is not None and record_field_arg is None:
                action_description = f"{action} {record_name} {extracted_record_id} to {changed_value}. Reason: {reason}"
            else:  # both supplied
                action_description = f"{action} {record_name} {extracted_record_id} {extracted_record_field} to {changed_value}. Reason: {reason}"

            # --- Build details dict dynamically ---
            if record_field_arg is None and changed_value_arg is None:
                details = {
                    record_id_arg: extracted_record_id,
                    "action": action,
                    "reason": reason,
                }
            elif changed_value_arg is not None and record_field_arg is None:
                details = {
                    record_id_arg: extracted_record_id,
                    "action": action,
                    changed_value_arg: changed_value,
                    "reason": reason,
                }
            else:  # both supplied
                details = {
                    record_id_arg: extracted_record_id,
                    "action": action,
                    record_field_arg: extracted_record_field,
                    changed_value_arg: changed_value,
                    "reason": reason,
                }

            # --- Submit approval request ---
            try:
                approval_wf = get_approval_workflow()
                
                result = approval_wf.submit_approval_request(
                    agent_id=tool_name,
                    action_type=tool_name,
                    action_description=action_description,
                    user_id=user_id,
                    user_role=user_role,
                    details=details,
                )
                
                logger.info(
                    f"Approval result for {tool_name} by {user_id}: "
                    f"approved={result['approved']}, "
                    f"pending={result['pending']}, "
                    f"request_id={result.get('request_id')}"
                )
            except CircuitBreakerError:
                result = {
                    "approved": False,
                    "pending": False,
                    "request_id": None,
                    "message": "System is halted by circuit breaker. No actions can be submitted."
                }
            except Exception as e:
                logger.error(
                    f"Approval submission failed for {tool_name}: {e}"
                )
                result = {
                    "approved": False,
                    "pending": False,
                    "request_id": None,
                    "message": f"Approval submission failed: {str(e)}",
                }

            if not result["approved"]:
                track_tool_execution_in_cg(
                    session_id, tool_name, details, status="pending_approval"
                )
                return json.dumps(result)

            # Approved - proceed to the actual tool logic
            return func(*args, **kwargs)

        return wrapper

    return decorator

def circuit_breaker(func):
    """
    Decorator that checks for circuit breaker flag before a tool function executes.

    On active circuit breaker it tracks the event via track_tool_execution_in_cg and returns a
    JSON error string.

    The tool name is derived automatically from the decorated function's __name__,
    so it never needs to be supplied manually.

    Stacking order with @tool:
        @tool                                      # outermost  — registered last
        @circuit_breaker                           # second - should check breaker first
        @require_rate_limits                       # middle - should run before permissions
        @require_permissions("MEMBER", "READ")     # innermost  — runs first at call time

    Usage:
        @tool
        @circuit_breaker
        @require_rate_limits
        @require_permissions("MEMBER", "READ")
        def member_lookup(member_id: str, user_role: str, ...) -> str:
            ...
    """
    # Resolve the tool name once at decoration time — never at call time.
    tool_name = func.__name__
    
    # Identify which positional index holds user_role, user_id, session_id.
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
        user_role  = _get("user_role",  args, kwargs) or "unknown"
        user_id = _get("user_role",  args, kwargs) or ""
        session_id = _get("session_id", args, kwargs) or "default"
        
        try:
            approval_wf = get_approval_workflow()
            
            if approval_wf.is_circuit_breaker_active():
                error = "The system is temporarily unavailable due to an emergency stop. Please contact your supervisor."
                
                # Track in Context Graph
                track_tool_execution_in_cg(
                    session_id, tool_name,
                    {"user_role": user_role, "user_id": user_id, "action": "check_circuit_breaker" },
                    status="circuit_breaker_active",
                    error=error
                )
                
                return json.dumps({"error": error})        
        except Exception as e:
            logger.warning(f"Circuit breaker check failed for tool call {tool_name} with error: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper

def require_rate_limits(func):
    """
    Decorator that enforces tool call rate limits before a tool function executes.

    On any denial it tracks the event via track_tool_execution_in_cg and returns a
    JSON error string.

    The tool name is derived automatically from the decorated function's __name__,
    so it never needs to be supplied manually.

    Stacking order with @tool:
        @tool                                      # outermost  — registered last
        @require_rate_limits                       # middle - should run before permissions
        @require_permissions("MEMBER", "READ")     # innermost  — runs first at call time

    Usage:
        @tool
        @require_rate_limits
        @require_permissions("MEMBER", "READ")
        def member_lookup(member_id: str, user_role: str, ...) -> str:
            ...
    """
    # Resolve the tool name once at decoration time — never at call time.
    tool_name = func.__name__
    
    # Identify which positional index holds user_role, user_id, session_id.
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
        user_role  = _get("user_role",  args, kwargs) or "unknown"
        user_id = _get("user_role",  args, kwargs) or ""
        session_id = _get("session_id", args, kwargs) or "default"
        
        try:
            # Look up the per-role rate limit for this tool
            limit_per_minute = rbac_service.get_tool_rate_limit(user_role, tool_name)
            
            if limit_per_minute <= 0:
                # Tool not permitted for this role (handled by RBAC check),
                # or rate limit is 0 - deny
                logger.warning(
                    f"Rate limit lookup returned 0 for {user_role}/{tool_name}"
                )
                return func(*args, **kwargs)  # Let RBAC check handle the denial
            
            # Record the check in Prometheus
            rate_limit_checks.labels(
                endpoint=tool_name,
                user_id=user_id,
                result="checked"
            ).inc()
            
            # Enforce the sliding-window rate limit
            rate_limiter.check_rate_limit(
                user_id=user_id,
                resource_type="TOOL",
                resource_name=tool_name,
                limit_per_minute=limit_per_minute
            )
            
            # If it is here, then it is within limits
            return func(*args, **kwargs)
        except RateLimitError as e:
            logger.warning(
                f"Control 6: Rate limit exceeded for user {user_id} "
                f"on tool {tool_name} (role={user_role}): {e}"
            )
            
            # Track in Prometheus
            track_rate_limit_exceeded(
                user_id=user_id,
                tool_name=tool_name,
                user_role=user_role,
                limit_type="per_minute"
            )
            
            rate_limit_checks.labels(
                endpoint=tool_name,
                user_id=user_id,
                result="exceeded"
            ).inc()
            
            # Track in Context Graph
            track_tool_execution_in_cg(
                session_id, tool_name,
                {"user_role": user_role, "user_id": user_id, "action": "require_rate_limits" },
                status="rate_limit_exceeded",
                error=str(e),
            )
            
            return json.dumps({
                "error": f"Rate limit exceeded for {tool_name}. "
                        f"Please wait before trying again.",
                "rate_limited": True,
                "tool_name": tool_name
            })
            
    return wrapper

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
        def member_lookup(member_id: str, user_role: str, ...) -> str:
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

def check_rate_limit_for_tool(
    tool_name: str,
    user_id: str,
    user_role: str,
    session_id: str = "default"
) -> Optional[str]:
    """
    Utility for Control 6: Enforce per-tool, per-role rate limiting.
    
    Looks up the rate_limit_per_minute for the given role/tool pair from
    the MySQL tool_permissions table (via RBACService) and then checks
    the sliding-window counter in the rate_limits table (via RateLimiter).
    
    Args:
        tool_name: Name of the tool being invoked
        user_id: User identifier for per-user tracking
        user_role: User's role for per-role limits
        session_id: Session ID for audit tracking
    
    Returns:
        None if within limits, or a JSON error string if rate limit exceeded.
    """
    try:
        # Look up the per-role rate limit for this tool
        limit_per_minute = rbac_service.get_tool_rate_limit(user_role, tool_name)
        
        if limit_per_minute <= 0:
            # Tool not permitted for this role (handled by RBAC check),
            # or rate limit is 0 - deny
            logger.warning(
                f"Rate limit lookup returned 0 for {user_role}/{tool_name}"
            )
            return None  # Let RBAC check handle the denial
        
        # Record the check in Prometheus
        rate_limit_checks.labels(
            endpoint=tool_name,
            user_id=user_id,
            result="checked"
        ).inc()
        
        # Enforce the sliding-window rate limit
        rate_limiter.check_rate_limit(
            user_id=user_id,
            resource_type="TOOL",
            resource_name=tool_name,
            limit_per_minute=limit_per_minute
        )
        
        # Within limits
        return None
    
    except RateLimitError as e:
        logger.warning(
            f"Control 6: Rate limit exceeded for user {user_id} "
            f"on tool {tool_name} (role={user_role}): {e}"
        )
        
        # Track in Prometheus
        track_rate_limit_exceeded(
            user_id=user_id,
            tool_name=tool_name,
            user_role=user_role,
            limit_type="per_minute"
        )
        rate_limit_checks.labels(
            endpoint=tool_name,
            user_id=user_id,
            result="exceeded"
        ).inc()
        
        # Track in Context Graph
        track_tool_execution_in_cg(session_id, tool_name, {}, status="rate_limited")
        
        return json.dumps({
            "error": f"Rate limit exceeded for {tool_name}. "
                     f"Please wait before trying again.",
            "rate_limited": True,
            "tool_name": tool_name
        })
    
    except Exception as e:
        # Fail open - log but allow the request
        logger.error(f"Rate limit check failed for {tool_name}: {e}")
        return None


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
    
# ============================================
# Semantic Search Tools (Chroma Vector DB)
# ============================================

@tool
@require_permissions("POLICY", "READ")
def search_policy_info(query: str, user_role: str, plan_type: str = "", session_id: str = "default") -> str:
    """
    Semantic search over policy documents in Chroma vector database.

    Searches the 'policies' collection which contains policy text with plan
    details, premiums, deductibles, and out-of-pocket maximums.

    Args:
        query:     Natural-language question (e.g. "What is my deductible?")
        user_role: The role of the user making the request
        plan_type: Optional filter - HMO, PPO, EPO, POS
        session_id: Session ID for audit and scrubbing

    Returns:
        JSON string with semantically matched policy documents
    """
    
    start_time = datetime.now()

    # Sanitize inputs
    query = sanitize_text(query)
    plan_type = sanitize_text(plan_type) if plan_type else ""

    try:
        results = chroma_data_access.search_policies(
            query=query,
            n_results=5,
            plan_type=plan_type,
        )

        output = json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        }, indent=2)

        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = { "user_role": user_role, "query": query }
        track_tool_execution_in_cg(session_id, "search_policy_info", inputs, status="success", execution_time_ms=execution_time)

        return scrubbed_output

    except Exception as e:
        logger.error(f"search_policy_info failed: {e}")
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = { "user_role": user_role, "query": query }
        error = str(e)
        track_tool_execution_in_cg(session_id, "search_policy_info", inputs, status="failed", execution_time_ms=execution_time, error=error)
        return json.dumps({"error": error})


@tool
@require_permissions("PA", "READ")
def search_medical_codes(
    query: str,
    user_role: str,
    code_type: str = "both",
    session_id: str = "default",
) -> str:
    """
    Semantic search over CPT procedure codes and ICD-10 diagnosis codes.

    Searches the 'procedures' and/or 'diagnoses' Chroma collections.

    Args:
        query:     Natural-language description (e.g. "knee replacement surgery")
        user_role: The role of the user making the request
        code_type: "procedure", "diagnosis", or "both" (default)
        session_id: Session ID for audit and scrubbing

    Returns:
        JSON string with matched CPT/ICD-10 codes and descriptions
    """
    start_time = datetime.now()

    query = sanitize_text(query)
    code_type = sanitize_text(code_type) if code_type else "both"

    try:
        result_data: dict = {"query": query}

        if code_type in ("procedure", "both"):
            result_data["procedures"] = chroma_data_access.search_procedures(
                query=query, n_results=5
            )

        if code_type in ("diagnosis", "both"):
            result_data["diagnoses"] = chroma_data_access.search_diagnoses(
                query=query, n_results=5
            )

        output = json.dumps(result_data, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = { "user_role": user_role, "query": query }
        track_tool_execution_in_cg(session_id, "search_medical_codes", inputs, status="success", execution_time_ms=execution_time)

        return scrubbed_output

    except Exception as e:
        logger.error(f"search_medical_codes failed: {e}")
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = { "user_role": user_role, "query": query }
        error = str(e)
        track_tool_execution_in_cg(session_id, "search_medical_codes", inputs, status="failed", execution_time_ms=execution_time, error=error)
        return json.dumps({"error": error})


@tool
@require_permissions("KB", "READ")
def search_knowledge_base(
    query: str,
    user_role: str,
    source: str = "all",
    session_id: str = "default",
) -> str:
    """
    Semantic search over FAQs, clinical guidelines, and regulations.

    Searches the 'faqs', 'clinical_guidelines', and 'regulations' Chroma
    collections to find relevant knowledge-base content.

    Args:
        query:      Natural-language question
        user_role:  The role of the user making the request
        source:     "faqs", "guidelines", "regulations", or "all" (default)
        session_id: Session ID for audit and scrubbing

    Returns:
        JSON string with matched knowledge-base documents
    """
    start_time = datetime.now()

    query = sanitize_text(query)
    source = sanitize_text(source) if source else "all"

    try:
        result_data: dict = {"query": query}

        if source in ("faqs", "all"):
            result_data["faqs"] = chroma_data_access.search_faqs(
                query=query, n_results=5
            )

        if source in ("guidelines", "all"):
            result_data["clinical_guidelines"] = chroma_data_access.search_clinical_guidelines(
                query=query, n_results=5
            )

        if source in ("regulations", "all"):
            result_data["regulations"] = chroma_data_access.search_regulations(
                query=query, n_results=3
            )

        output = json.dumps(result_data, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = { "user_role": user_role, "query": query }
        track_tool_execution_in_cg(session_id, "search_knowledge_base", inputs, status="success", execution_time_ms=execution_time)

        return scrubbed_output

    except Exception as e:
        logger.error(f"search_knowledge_base failed: {e}")
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = { "user_role": user_role, "query": query }
        error = str(e)
        track_tool_execution_in_cg(session_id, "search_medisearch_knowledge_basecal_codes", inputs, status="failed", execution_time_ms=execution_time, error=error)
        return json.dumps({"error": error})