"""
PA Services MCP Tool Server (HTTP Transport)
=================================================
Exposes pa services tools as remote HTTP endpoints following the
Model Context Protocol pattern.

The MCP endpoint is at:
    POST/GET http://<host>:<port>/mcp        ← MCP Streamable HTTP transport
    GET      http://<host>:<port>/health     ← Docker Swarm health check

Architecture:
    A2A Supervisor Container          MCP Tool Container
    ┌──────────────────────┐         ┌──────────────────────┐
    │ PAServicesSuperv     │         │ pa_services          │
    │   └─ MCPToolClient   │─-HTTP──▶│   mcp_server.py      │
    │  (StreamableHTTP)    │         │   (tools inlined)    │
    └──────────────────────┘         └──────────────────────┘
"""

import json
import logging
import os
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from agents.security import rbac_service

from dotenv import load_dotenv, find_dotenv

from agents.tools_util import (
    circuit_breaker,
    validate_user_role,
    require_approvals,
    require_rate_limits,
    require_permissions,
    track_tool_execution_in_cg,
    scrub_output,
)
from databases.knowledge_graph_data_access import get_kg_data_access
from security.nh3_sanitization import sanitize_text

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastMCP server
# ─────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="pa_services_mcp",
    instructions=(
        "Prior Authorization Services MCP Tool Server. Provides pa lookup, "
        "pa requirements, and pa status tools. "
        "All tools require user_id and user_role for RBAC enforcement."
    ),
    host=os.getenv("MCP_SERVER_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_SERVER_PORT", "8001")),
    stateless_http=True,   # no Mcp-Session-Id handshake; each request is self-contained
    json_response=True,    # return application/json instead of text/event-stream
)

# ─────────────────────────────────────────────────────────────
# Health check (Docker Swarm / load-balancer liveness probe)
# ─────────────────────────────────────────────────────────────

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Health check endpoint for Docker Swarm."""
    return JSONResponse({
        "status": "healthy",
        "service": "pa_services_mcp",
        "version": "27.0.0",
        "transport": "streamable-http",
    })

# ─────────────────────────────────────────────────────────────
# Admin endpoint — utility to clear the cache
# ─────────────────────────────────────────────────────────────

@mcp.custom_route("/admin/rbac/clear-cache", methods=["POST"])
async def clear_rbac_cache(request: Request) -> JSONResponse:
    """
    Clear the RBACService in-memory permission cache.

    Call this after making direct changes to `tool_permissions` or
    `role_permissions` in MySQL so the MCP server picks them up
    immediately without a container restart.

    Reachable only over the internal Docker overlay network:
        docker exec $(docker ps -qf name=<container_name>) \
            python3 -c "import urllib.request; \
            r = urllib.request.urlopen(urllib.request.Request('http://localhost:8001/admin/rbac/clear-cache', method='POST')); \
            print(r.read().decode())"
    """
    result = rbac_service.clear_cache()
    logger.info("RBAC cache cleared via admin endpoint: %s", result)
    return JSONResponse({"status": "ok", **result})


# ─────────────────────────────────────────────────────────────
# MCP Tools
#
# Each tool applies the full decorator stack from tools.py:
#   @circuit_breaker, @validate_user_role, @require_approvals, @require_rate_limits,
#   @require_permissions
#
# Input sanitization, KG queries, PII/PHI scrubbing, and context
# graph tracking are inlined directly from the original tool bodies.
# ─────────────────────────────────────────────────────────────

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="pa", record_id_arg="pa_id")
@require_rate_limits
@require_permissions("PA", "READ")
def pa_lookup(
    pa_id: str,
    user_id: str, 
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up prior authorization information by PA ID.
    
    Uses relationships:
        (Member)-[:REQUESTED_PA]->(PriorAuthorization)
        (PriorAuthorization)-[:REQUESTED_BY]->(Provider)
    
    PA properties: paId, paNumber, procedureCode, procedureDescription,
    requestDate, status, urgency, approvalDate, expirationDate, denialReason
    
    Args:
        pa_id: The prior authorization's unique identifier
        user_id: ID of the user making the request (for rate limiting)
        user_role: The role of the user making the request
        session_id: Session ID for audit and scrubbing
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with PA information
    """
    start_time = datetime.now()
    
    # Sanitize input
    pa_id = sanitize_text(pa_id)
    
    try:
        kg_data_access = get_kg_data_access()
        pa = kg_data_access.get_prior_authorization(pa_id)
        
        if not pa:
            error = f"PA not found: {pa_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "pa_lookup", {"pa_id": pa_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})
        
        output = json.dumps(pa, indent=2)
        scrubbed_output = scrub_output(output, session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "pa_lookup", {"pa_id": pa_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        
        
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"pa_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "pa_lookup", {"pa_id": pa_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="pa", record_id_arg="procedure_code")
@require_rate_limits
@require_permissions("PA", "READ")
def pa_requirements(
    procedure_code: str, 
    policy_type: str,
    user_id: str, 
    user_role: str, 
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up prior authorization requirements for a procedure under a policy type.
    
    This queries the member's policy to determine if a procedure requires PA,
    based on the policy type (HMO, PPO, etc.) and procedure code.
    
    Args:
        procedure_code: CPT code of the procedure
        policy_type: Type of policy (HMO, PPO, EPO, POS)
        user_id: ID of the user making the request (for rate limiting)
        user_role: The role of the user making the request
        session_id: Session ID for audit and scrubbing
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with PA requirements
    """
    start_time = datetime.now()
    
    # Sanitize inputs
    procedure_code = sanitize_text(procedure_code)
    policy_type = sanitize_text(policy_type)
    
    try:
        # Query existing PAs for this procedure code to determine requirements
        # Schema: PriorAuthorization has procedureCode property
        kg_data_access = get_kg_data_access()

        results = kg_data_access.get_pa_requirements( procedure_code=procedure_code, policy_type=policy_type )
        
        if not results:
            output = json.dumps({
                "requires_pa": False,
                "procedureCode": procedure_code,
                "policyType": policy_type,
                "reason": "No PA history found for this procedure/policy combination"
            })
        else:
            output = json.dumps({
                "requires_pa": True,
                "procedureCode": procedure_code,
                "policyType": policy_type,
                "history": results
            }, indent=2)
        
        # Scrub PII/PHI from output
        scrubbed_output = scrub_output(output, session_id)
        
        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "pa_requirements", {"procedure_code": procedure_code, "policy_type": policy_type},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"pa_requirements failed: {e}")
        error = str(e)
        inputs = {"procedure_code": procedure_code, "policy_type": policy_type}
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "pa_requirements", inputs,
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="pa", record_id_arg="pa_id")
@require_rate_limits
@require_permissions("PA", "READ")
def pa_status(
    pa_id: str,
    user_id: str, 
    user_role: str, 
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Check the status of a prior authorization.
    
    PA status properties: paId, paNumber, status, requestDate,
    urgency, approvalDate, expirationDate, denialReason
    
    Args:
        pa_id: The prior authorization's unique identifier
        user_id: ID of the user making the request (for rate limiting)
        user_role: The role of the user making the request
        session_id: Session ID for audit and scrubbing
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with PA status
    """
    start_time = datetime.now()
    
    # Sanitize input
    pa_id = sanitize_text(pa_id)
    
    try:
        kg_data_access = get_kg_data_access()
        pa = kg_data_access.get_prior_authorization(pa_id)
        
        if not pa:
            error = f"PA not found: {pa_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "pa_status", {"pa_id": pa_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})
        
        # Extract status-relevant fields
        status_info = {
            "paId": pa.get("paId"),
            "paNumber": pa.get("paNumber"),
            "status": pa.get("status"),
            "urgency": pa.get("urgency"),
            "requestDate": pa.get("requestDate"),
            "approvalDate": pa.get("approvalDate"),
            "expirationDate": pa.get("expirationDate"),
            "denialReason": pa.get("denialReason")
        }
        
        output = json.dumps(status_info, indent=2)
    
        scrubbed_output = scrub_output(output, session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "pa_status", {"pa_id": pa_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
         
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"pa_status failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "pa_status", {"pa_id": pa_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})
    
# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def run_mcp_server() -> None:
    """Run the MCP tool server as a standalone HTTP service."""
    logger.info(
        "Starting PA Services MCP Server (streamable-http) on %s:%d",
        mcp.settings.host, mcp.settings.port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()