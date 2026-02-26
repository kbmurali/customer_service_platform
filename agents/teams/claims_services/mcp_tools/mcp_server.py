"""
Claims Services MCP Tool Server (HTTP Transport)
=================================================
Exposes claims services tools as remote HTTP endpoints following the
Model Context Protocol pattern.

v27: Merged tools.py into mcp_server.py. Each @mcp.tool() function now
contains the full tool implementation directly — KG data access, input
sanitization, PII/PHI scrubbing, RBAC, audit logging, and context graph
tracking — with no dependency on tools.py or LangChain @tool wrappers.

The MCP endpoint is at:
    POST/GET http://<host>:<port>/mcp        ← MCP Streamable HTTP transport
    GET      http://<host>:<port>/health     ← Docker Swarm health check

Architecture:
    A2A Supervisor Container          MCP Tool Container
    ┌──────────────────────┐         ┌──────────────────────┐
    │ ClaimsServicesSuperv │         │ claims_services       │
    │   └─ MCPToolClient   │─HTTP──▶│   mcp_server.py      │
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
    name="claims_services_mcp",
    instructions=(
        "Claims Services MCP Tool Server. Provides claims lookup, "
        "check eligibility, and coverage lookup tools. "
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
        "service": "claims_services_mcp",
        "version": "27.0.0",
        "transport": "streamable-http",
    })

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="claim", record_id_arg="claim_id")
@require_rate_limits
@require_permissions("CLAIM", "READ")
def claim_lookup(
    claim_id: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up claim information by claim ID.
    
    Uses relationships:
        (Member)-[:FILED_CLAIM]->(Claim)
        (Claim)-[:UNDER_POLICY]->(Policy)
        (Claim)-[:SERVICED_BY]->(Provider)
    
    Args:
        claim_id:     The claim's unique identifier
        user_id:      ID of the requesting user (for audit logging).
        user_role:    RBAC role of the requesting user.
        session_id:   Session ID for audit and PII scrubbing.
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with claim information, or an error payload.
    """
    start_time = datetime.now()
    
    # Sanitize input
    claim_id = sanitize_text(claim_id)
    
    try:
        # Query claim with full context via data access layer
        kg_data_access = get_kg_data_access()
        claim = kg_data_access.get_claim_with_full_context(claim_id)
        
        if not claim:
            error = f"Claim not found: {claim_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "claim_lookup", {"claim_id": claim_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})
        
        output = json.dumps(claim, indent=2)
        
        # Scrub PII/PHI from output
        scrubbed_output = scrub_output(output, session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "claim_lookup", {"claim_id": claim_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
         
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"claim_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "claim_lookup", {"claim_id": claim_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="claim", record_id_arg="claim_number")
@require_rate_limits
@require_permissions("CLAIM", "READ")
def claim_status(
    claim_number: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Check the status of a claim by claim number.
    
    Claim properties: claimId, claimNumber, serviceDate, submissionDate, status,
    totalAmount, paidAmount, denialReason, processingDate
    
    Args:
        claim_number:     The claim number (e.g., CLM-123456) (different from claim id)
        user_id:          ID of the requesting user (for audit logging).
        user_role:        RBAC role of the requesting user.
        session_id:       Session ID for audit and PII scrubbing.
        execution_id:     AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with claim status or an error payload
    """
    start_time = datetime.now()
    
    # Sanitize input
    claim_number = sanitize_text(claim_number)
    
    try:
        # Use data access layer to get claim status
        kg_data_access = get_kg_data_access()
        
        claim = kg_data_access.get_claim_status(claim_number)
        
        if not claim:
            error = f"Claim status not found: {claim_number}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "claim_status", {"claim_number": claim_number},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})
        
        output = json.dumps(claim, indent=2)
        
        # Scrub PII/PHI from output
        scrubbed_output = scrub_output(output, session_id)
        
        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "claim_status", {"claim_number": claim_number},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
         
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"claim_status failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "claim_status", {"claim_number": claim_number},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})
    
@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="claim", record_id_arg="claim_id")
@require_rate_limits
@require_permissions("CLAIM", "READ")
def claim_payment_info(
    claim_id: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Get payment information for a claim.
    
    Payment info is derived from Claim node properties:
    totalAmount, paidAmount, status, processingDate
    
    Args:
        claim_id:     The claim's unique identifier
        user_id:      ID of the requesting user (for audit logging).
        user_role:    RBAC role of the requesting user.
        session_id:   Session ID for audit and PII scrubbing.
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with claim payment information
    """
    start_time = datetime.now()
    
    # Sanitize input
    claim_id = sanitize_text(claim_id)
    
    try:
        # Get claim with full context to include payment and policy info
        kg_data_access = get_kg_data_access()
        
        claim = kg_data_access.get_claim_with_full_context(claim_id)
        
        if not claim:
            error = f"Claim not found: {claim_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "claim_payment_info", {"claim_id": claim_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})
        
        
        # Extract payment-relevant fields
        payment_info = {
            "claimId": claim.get("claimId"),
            "claimNumber": claim.get("claimNumber"),
            "status": claim.get("status"),
            "totalAmount": claim.get("totalAmount"),
            "paidAmount": claim.get("paidAmount"),
            "processingDate": claim.get("processingDate"),
            "denialReason": claim.get("denialReason"),
            "policy": claim.get("policy")
        }
        
        output = json.dumps(payment_info, indent=2)
        
        # Scrub PII/PHI from output
        scrubbed_output = scrub_output(output, session_id)
        
        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "claim_payment_info", {"claim_id": claim_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"claim_payment_info failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "claim_payment_info", {"claim_id": claim_id},
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
        "Starting Claim Services MCP Server (streamable-http) on %s:%d",
        mcp.settings.host, mcp.settings.port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()