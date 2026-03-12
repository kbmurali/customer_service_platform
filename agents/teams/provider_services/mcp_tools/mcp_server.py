"""
Provider Services MCP Tool Server (HTTP Transport)
=================================================
Exposes provider services tools as remote HTTP endpoints following the
Model Context Protocol pattern.

The MCP endpoint is at:
    POST/GET http://<host>:<port>/mcp        ← MCP Streamable HTTP transport
    GET      http://<host>:<port>/health     ← Docker Swarm health check

Architecture:
    A2A Supervisor Container          MCP Tool Container
    ┌──────────────────────┐         ┌──────────────────────┐
    │ProviderServicesSuperv│         │ provider_services    │
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
    name="provider_services_mcp",
    instructions=(
        "Provider Services MCP Tool Server. Provides provider lookup, "
        "provider network check, and provider search by specialty tools. "
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
        "service": "provider_services_mcp",
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
@require_approvals(action="Read", record_name="provider", record_id_arg="provider_id")
@require_rate_limits
@require_permissions("PROVIDER", "READ")
def provider_lookup(
    provider_id: str,
    user_id: str, 
    user_role: str, 
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Look up provider information by provider ID.
    
    Provider properties (from schema):
        providerId, npi, providerType, specialty, phone,
        street, city, state, zipCode,
        organizationName (if ORGANIZATION), firstName/lastName (if INDIVIDUAL)
    
    Args:
        provider_id: The provider's unique identifier
        user_id: ID of the user making the request (for rate limiting)
        user_role: The role of the user making the request
        session_id: Session ID for audit and scrubbing
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with provider information
    """
    start_time = datetime.now()
    
    # Sanitize input
    provider_id = sanitize_text(provider_id)
    
    try:
        kg_data_access = get_kg_data_access()
        provider = kg_data_access.get_provider(provider_id)
        
        if not provider:
            error = f"Provider not found: {provider_id}"
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            track_tool_execution_in_cg(
                session_id, "provider_lookup", {"provider_id": provider_id},
                status="not_found", execution_time_ms=execution_time, error=error,
                execution_id=execution_id or None,
            )
            return json.dumps({"error": error})
        
        output = json.dumps(provider, indent=2)
        
        scrubbed_output = scrub_output(output, session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "provider_lookup", {"provider_id": provider_id},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
         
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"provider_lookup failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "provider_lookup", {"provider_id": provider_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})
    
@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="provider", record_id_arg="provider_id")
@require_rate_limits
@require_permissions("PROVIDER", "READ")
def provider_network_check(
    provider_id: str, 
    policy_id: str,
    user_id: str, 
    user_role: str, 
    session_id: str,
    execution_id: str = "",
    ) -> str:
    """
    Check if a provider serviced claims under a specific policy.
    
    Since the schema does not have a separate Network node, network status
    is inferred from existing claim relationships:
        (Claim)-[:SERVICED_BY]->(Provider)
        (Claim)-[:UNDER_POLICY]->(Policy)
    
    If a provider has serviced claims under the given policy, they are
    considered in-network for that policy.
    
    Args:
        provider_id: The provider's unique identifier
        policy_id: The policy's unique identifier
        user_id: ID of the user making the request (for rate limiting)
        user_role: The role of the user making the request
        session_id: Session ID for audit and scrubbing
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with provider network status
    """
    start_time = datetime.now()
    
    # Sanitize inputs
    provider_id = sanitize_text(provider_id)
    policy_id = sanitize_text(policy_id)
    
    try:
        kg_data_access = get_kg_data_access()
    
        results = kg_data_access.find_serviced_claims_by_provider_under_a_policy( provider_id=provider_id, policy_id=policy_id )
        
        if not results or results[0].get("claimCount", 0) == 0:
            output = json.dumps({
                "has_history": False,
                "reason": "No claims found for this provider under this policy"
            })
        else:
            data = results[0]
            output = json.dumps({
                "has_history": True,
                "provider": data.get("provider"),
                "policy": data.get("policy"),
                "claimCount": data.get("claimCount")
            }, indent=2)
        
        scrubbed_output = scrub_output(output, session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "provider_network_check", {"provider_id": provider_id, "policy_id": policy_id },
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"provider_network_check failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "provider_network_check", {"provider_id": provider_id, "policy_id": policy_id},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="provider", record_id_arg="specialty")
@require_rate_limits
@require_permissions("PROVIDER", "READ")
def provider_search_by_specialty(
    specialty: str, 
    zip_code: str,
    user_id: str,
    user_role: str, 
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Search for providers by specialty and location.
    
    Provider properties (from schema):
        providerId, npi, providerType, specialty, phone,
        street, city, state, zipCode,
        organizationName (if ORGANIZATION), firstName/lastName (if INDIVIDUAL)
    
    Args:
        specialty: Provider specialty (e.g., "Cardiology")
        zip_code: ZIP code for location search
        user_id: ID of the user making the request (for rate limiting)
        user_role: The role of the user making the request
        session_id: Session ID for audit and scrubbing
        execution_id: AgentExecution.executionId for CG CALLED_TOOL link.
    
    Returns:
        JSON string with provider search results
    """
    start_time = datetime.now()
    
    # Sanitize inputs
    specialty = sanitize_text(specialty)
    zip_code = sanitize_text(zip_code)
    
    try:
        kg_data_access = get_kg_data_access()
        
        providers = kg_data_access.search_providers(
            specialty=specialty,
            zip_code=zip_code,
            limit=10
        )
        
        output = json.dumps({
            "count": len(providers),
            "providers": providers
        }, indent=2)
        
        scrubbed_output = scrub_output(output, session_id)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "provider_search_by_specialty", {"specialty": specialty, "zip_code": zip_code },
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )
        
        return scrubbed_output
    
    except Exception as e:
        logger.error(f"provider_search_by_specialty failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "provider_search_by_specialty", {"specialty": specialty, "zip_code": zip_code },
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
        "Starting Provider Services MCP Server (streamable-http) on %s:%d",
        mcp.settings.host, mcp.settings.port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()