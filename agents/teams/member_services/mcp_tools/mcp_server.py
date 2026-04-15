"""
Member Services MCP Tool Server (HTTP Transport)
=================================================
Exposes member services tools as remote HTTP endpoints following the
Model Context Protocol pattern.

The MCP endpoint is at:
    POST/GET http://<host>:<port>/mcp        ← MCP Streamable HTTP transport
    GET      http://<host>:<port>/health     ← Docker Swarm health check

Architecture:
    A2A Supervisor Container          MCP Tool Container
    ┌──────────────────────┐         ┌──────────────────────┐
    │ MemberServicesSuperv │         │ member_services       │
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
    name="member_services_mcp",
    instructions=(
        "Member Services MCP Tool Server. Provides member lookup, "
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
        "service": "member_services_mcp",
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
@require_approvals(action="Read", record_name="member", record_id_arg="member_id")
@require_rate_limits
@require_permissions("MEMBER", "READ")
def member_lookup(
    member_id: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up member information by member ID.

    Args:
        member_id:    The member's unique identifier (e.g. M123456).
        user_id:      ID of the requesting user (for audit logging).
        user_role:    RBAC role of the requesting user.
        session_id:   Session ID for audit and PII scrubbing.
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with member information, or an error payload.
    """
    start_time = datetime.now()
    member_id = sanitize_text(member_id)

    try:
        kg_data_access = get_kg_data_access()
        member = kg_data_access.get_member(member_id)

        if not member:
            error = f"Member not found: {member_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "member_lookup", {"member_id": member_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})

        output = json.dumps(member, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "member_lookup", {"member_id": member_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        return scrubbed_output

    except Exception as e:
        logger.error(f"member_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "member_lookup", {"member_id": member_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})


@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="member", record_id_arg="member_id")
@require_rate_limits
@require_permissions("MEMBER", "READ")
def check_eligibility(
    member_id: str,
    service_date: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Check member eligibility for services on a specific date.

    Args:
        member_id:    The member's unique identifier.
        service_date: Date of service in YYYY-MM-DD format.
        user_id:      ID of the requesting user (for audit logging).
        user_role:    RBAC role of the requesting user.
        session_id:   Session ID for audit and PII scrubbing.
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with eligibility information, or an error payload.
    """
    start_time = datetime.now()
    member_id    = sanitize_text(member_id)
    service_date = sanitize_text(service_date)
    
    # Validate service_date format before hitting the KG
    if not service_date:
        return json.dumps({"error": "service_date is required."})
    try:
        datetime.strptime(service_date, "%Y-%m-%d")
    except ValueError:
        return json.dumps({"error": f"Invalid service_date '{service_date}'. Expected format: YYYY-MM-DD."})

    try:
        kg_data_access = get_kg_data_access()
        eligibility = kg_data_access.check_eligibility(member_id, service_date)

        output = json.dumps(eligibility, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = {"member_id": member_id, "service_date": service_date}
        track_tool_execution_in_cg(
            session_id, "check_eligibility", inputs,
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        return scrubbed_output

    except Exception as e:
        logger.error(f"check_eligibility failed: {e}")
        error = str(e)
        inputs = {"member_id": member_id, "service_date": service_date}
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "check_eligibility", inputs,
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})


@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="member", record_id_arg="member_id")
@require_rate_limits
@require_permissions("MEMBER", "READ")
def coverage_lookup(
    member_id: str,
    procedure_code: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up coverage details derived from a member's active policy.

    Uses the graph relationship: (Member)-[:HAS_POLICY]->(Policy).

    Args:
        member_id:      The member's unique identifier.
        procedure_code: Optional CPT code to include in the response context.
        user_id:        ID of the requesting user (for audit logging).
        user_role:      RBAC role of the requesting user.
        session_id:     Session ID for audit and PII scrubbing.
        execution_id:   AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with coverage information, or an error payload.
    """
    start_time = datetime.now()
    member_id      = sanitize_text(member_id)
    procedure_code = sanitize_text(procedure_code)

    try:
        kg_data_access = get_kg_data_access()
        coverage = kg_data_access.get_member_coverage(member_id)

        if not coverage:
            error = "No active policy found for member"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "coverage_lookup", {"member_id": member_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"covered": False, "reason": error})

        if procedure_code:
            coverage["requestedProcedureCode"] = procedure_code

        output = json.dumps(coverage, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "coverage_lookup", {"member_id": member_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        return scrubbed_output

    except Exception as e:
        logger.error(f"coverage_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "coverage_lookup", {"member_id": member_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Update", record_name="member", record_id_arg="member_id")
@require_rate_limits
@require_permissions("MEMBER", "UPDATE")
def update_member_info(
    member_id: str,
    field: str,
    new_value: str,
    reason: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Update a member's information field. HIGH-IMPACT: requires human approval.

    Updatable fields: phone, email, address_street, address_city, address_state,
    address_zip, enrollmentDate, status

    Args:
        member_id: The member's unique identifier
        field: The field to update
        new_value: The new value
        reason: Justification for the change
        user_id: ID of the CSR making the change
        user_role: Role of the CSR
        session_id: Session ID for audit
        execution_id:   AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with result or approval-pending message
    """
    start_time = datetime.now()

    ALLOWED_FIELDS = {"phone", "email", "address_street", "address_city", "address_state", "address_zip", "enrollmentDate", "status"}

    member_id = sanitize_text(member_id)
    field = sanitize_text(field)
    new_value = sanitize_text(new_value)
    reason = sanitize_text(reason)

    try:
        kg_data_access = get_kg_data_access()
        success = kg_data_access.update_member_field(member_id, field, new_value)

        if not success:
            error = f"Member not found or update failed: {member_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "update_member_info", {"member_id": member_id, "field": field, "new_value": new_value, "reason": reason},
                status="update_failed", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})

        result = {"member_id": member_id, "field": field, "updated": True}
        scrubbed = scrub_output(json.dumps(result), session_id)
        
        # Track successful execution in Context Graph
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "update_member_info", {"member_id": member_id, "field": field, "new_value": new_value, "reason": reason},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )

        return scrubbed

    except Exception as e:
        logger.error(f"update_member_info failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "update_member_info", {"member_id": member_id, "field": field, "new_value": new_value, "reason": reason},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})


@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="member_policy", record_id_arg="member_id")
@require_rate_limits
@require_permissions("MEMBER", "READ")
def member_policy_lookup(
    member_id: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up member information together with their associated insurance policy.

    Uses the graph relationship: (Member)-[:HAS_POLICY]->(Policy).
    Returns member demographics and full policy details (policyId, policyNumber,
    policyType, planName, planType, effectiveDate, expirationDate, status,
    premium, deductible, outOfPocketMax).

    Args:
        member_id:    The member's unique identifier (e.g. M123456).
        user_id:      ID of the requesting user (for audit logging).
        user_role:    RBAC role of the requesting user.
        session_id:   Session ID for audit and PII scrubbing.
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with member and policy information, or an error payload.
    """
    start_time = datetime.now()
    member_id = sanitize_text(member_id)

    try:
        kg_data_access = get_kg_data_access()
        member_with_policy = kg_data_access.get_member_with_policy(member_id)

        if not member_with_policy:
            error = f"Member not found: {member_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "member_policy_lookup", {"member_id": member_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})

        output = json.dumps(member_with_policy, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "member_policy_lookup", {"member_id": member_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        return scrubbed_output

    except Exception as e:
        logger.error(f"member_policy_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "member_policy_lookup", {"member_id": member_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})


@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="treatment", record_id_arg="member_id")
@require_rate_limits
@require_permissions("MEMBER", "READ")
def treatment_history(
    member_id: str,
    user_id: str,
    user_role: str,
    session_id: str,
    treatment_type: str = "",
    procedure_code: str = "",
    execution_id: str = "",
) -> str:
    """
    Retrieve treatment history for a member.

    Used by the PA recommendation decision agent to verify whether
    conservative treatment requirements (e.g. physical therapy sessions,
    medication trials) have been met before recommending approval of a
    surgical prior authorization.

    Uses the graph relationships:
        (Member)-[:HAS_TREATMENT]->(TreatmentRecord)-[:PERFORMED_BY]->(Provider)

    Args:
        member_id:      The member's unique identifier.
        user_id:        ID of the requesting user (for audit logging).
        user_role:      RBAC role of the requesting user.
        session_id:     Session ID for audit and PII scrubbing.
        treatment_type: Optional filter — PHYSICAL_THERAPY, MEDICATION_TRIAL,
                        INJECTION, IMAGING, WEIGHT_COUNSELING, etc.
        procedure_code: Optional CPT code filter.
        execution_id:   AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with treatment records and provider info, or an error payload.
    """
    start_time = datetime.now()
    member_id = sanitize_text(member_id)
    treatment_type = sanitize_text(treatment_type) if treatment_type else ""
    procedure_code = sanitize_text(procedure_code) if procedure_code else ""

    try:
        kg_data_access = get_kg_data_access()
        treatments = kg_data_access.get_member_treatment_history(
            member_id=member_id,
            treatment_type=treatment_type or None,
            procedure_code=procedure_code or None,
        )

        output = json.dumps({
            "member_id": member_id,
            "treatment_type_filter": treatment_type or "all",
            "count": len(treatments),
            "treatments": treatments,
        }, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        inputs = {"member_id": member_id}
        if treatment_type:
            inputs["treatment_type"] = treatment_type
        if procedure_code:
            inputs["procedure_code"] = procedure_code
        track_tool_execution_in_cg(
            session_id, "treatment_history", inputs,
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        return scrubbed_output

    except Exception as e:
        logger.error(f"treatment_history failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "treatment_history", {"member_id": member_id},
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
        "Starting Member Services MCP Server (streamable-http) on %s:%d",
        mcp.settings.host, mcp.settings.port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()