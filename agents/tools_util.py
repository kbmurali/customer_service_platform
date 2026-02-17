from typing import Optional, Dict, Any
import logging
import json
import functools
import inspect

from langchain_core.tools import tool

from databases.context_graph_data_access import get_cg_data_access
from databases.chroma_vector_data_access import get_chroma_data_access
from agents.security import RBACService
from security.presidio_memory_security import get_presidio_security
from security.nh3_sanitization import sanitize_text

logger = logging.getLogger(__name__)

# Initialize RBAC service
rbac_service = RBACService()

# Initialize Presidio security
presidio_security = get_presidio_security()

# Initialize Chroma vector data access for semantic search
chroma_data_access = get_chroma_data_access()

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
        plan_type: Optional filter – HMO, PPO, EPO, POS
        session_id: Session ID for audit and scrubbing

    Returns:
        JSON string with semantically matched policy documents
    """
    from datetime import datetime
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
    from datetime import datetime
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
    from datetime import datetime
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