"""
Search Services MCP Tool Server (HTTP Transport)
=================================================
Exposes search services tools as remote HTTP endpoints following the
Model Context Protocol pattern.

The MCP endpoint is at:
    POST/GET http://<host>:<port>/mcp        ← MCP Streamable HTTP transport
    GET      http://<host>:<port>/health     ← Docker Swarm health check

Architecture:
    A2A Supervisor Container          MCP Tool Container
    ┌──────────────────────┐         ┌──────────────────────┐
    │ SearchServicesSuperv │         │  search_services     │
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
from databases.chroma_vector_data_access import get_chroma_data_access
from security.nh3_sanitization import sanitize_text

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastMCP server
# ─────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="search_services_mcp",
    instructions=(
        "Search Services MCP Tool Server. Provides search_knowledge_base, "
        "search_medical_codes, and search_policy_info tools. "
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
        "service": "search_services_mcp",
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
@require_approvals(action="Read", record_name="policy", record_id_arg="query")
@require_rate_limits
@require_permissions("POLICY", "READ")
def search_knowledge_base(
    query: str,
    source: str,
    user_id: str,
    user_role: str,
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Semantic search over FAQs, clinical guidelines, and regulations.

    Searches the 'faqs', 'clinical_guidelines', and 'regulations' Chroma
    collections to find relevant knowledge-base content.

    Args:
        query:          Natural-language question
        source:         "faqs", "guidelines", "regulations", or "all"
        user_id:        ID of the user making the request (for rate limiting)
        user_role:      The role of the user making the request
        session_id:     Session ID for audit and scrubbing
        execution_id:   AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with matched knowledge-base documents
    """
    start_time = datetime.now()

    query = sanitize_text(query)
    source = sanitize_text(source) if source else "all"

    try:
        chroma_data_access = get_chroma_data_access()
        
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
                query=query, n_results=5
            )

        output = json.dumps(result_data, indent=2)
        scrubbed_output = scrub_output(output, session_id)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "search_knowledge_base", {"query": query, "source": source},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )

        return scrubbed_output

    except Exception as e:
        logger.error(f"search_knowledge_base failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "search_knowledge_base", {"query": query, "source": source},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})
    
@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="policy", record_id_arg="query")
@require_rate_limits
@require_permissions("POLICY", "READ")
def search_medical_codes(
    query: str,
    code_type: str,
    user_id: str,
    user_role: str,
    session_id: str = "default",
    execution_id: str = "",
) -> str:
    """
    Semantic search over CPT procedure codes and ICD-10 diagnosis codes.

    Searches the 'procedures' and/or 'diagnoses' Chroma collections.

    Args:
        query:          Natural-language description (e.g. "knee replacement surgery")
        code_type:      "procedure", "diagnosis", or "both" (default)
        user_id:        ID of the user making the request (for rate limiting)
        user_role:      The role of the user making the request
        session_id:     Session ID for audit and scrubbing
        execution_id:   AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with matched CPT/ICD-10 codes and descriptions
    """
    start_time = datetime.now()

    query = sanitize_text(query)
    code_type = sanitize_text(code_type) if code_type else "both"

    try:
        chroma_data_access = get_chroma_data_access()
        
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
        track_tool_execution_in_cg(
            session_id, "search_medical_codes", {"query": query, "code_type": code_type},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )

        return scrubbed_output

    except Exception as e:
        logger.error(f"search_medical_codes failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "search_medical_codes", {"query": query, "code_type": code_type},
            status="failed", execution_time_ms=execution_time, error=error,
            execution_id=execution_id or None,
        )
        return json.dumps({"error": error})

@mcp.tool()
@circuit_breaker
@validate_user_role
@require_approvals(action="Read", record_name="policy", record_id_arg="query")
@require_rate_limits
@require_permissions("POLICY", "READ")
def search_policy_info(
    query: str,
    plan_type: str, 
    user_id: str, 
    user_role: str, 
    session_id: str,
    execution_id: str = "",
) -> str:
    """
    Semantic search over policy documents in Chroma vector database.

    Searches the 'policies' collection which contains policy text with plan
    details, premiums, deductibles, and out-of-pocket maximums.

    Args:
        query:          Natural-language question (e.g. "What is my deductible?")
        plan_type:      HMO, PPO, EPO, POS
        user_id:        ID of the user making the request (for rate limiting)
        user_role:      The role of the user making the request
        session_id:     Session ID for audit and scrubbing
        execution_id:   AgentExecution.executionId for CG CALLED_TOOL link.

    Returns:
        JSON string with semantically matched policy documents
    """
    start_time = datetime.now()

    # Sanitize inputs
    query = sanitize_text(query)
    plan_type = sanitize_text(plan_type) if plan_type else ""

    try:
        chroma_data_access = get_chroma_data_access()
        
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
        track_tool_execution_in_cg(
            session_id, "search_policy_info", {"query": query, "plan_type": plan_type},
            status="success", execution_time_ms=execution_time,
            execution_id=execution_id or None,
        )

        return scrubbed_output

    except Exception as e:
        logger.error(f"search_policy_info failed: {e}")
        error = str(e)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        track_tool_execution_in_cg(
            session_id, "search_policy_info", {"query": query, "plan_type": plan_type},
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
        "Starting Search Services MCP Server (streamable-http) on %s:%d",
        mcp.settings.host, mcp.settings.port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()