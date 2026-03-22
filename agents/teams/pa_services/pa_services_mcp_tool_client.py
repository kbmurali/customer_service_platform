"""
PA Services MCP Tool Client (v26)
===================================
Connects to the Prior Authorization Services FastMCP server and exposes
its tools as LangChain Tool objects.

Server default: http://pa-services-mcp-tools:8001
Env override:   MCP_PA_SERVICES_HTTP_URL

Tools:
    pa_lookup      — look up prior authorization information by PA ID
    pa_status       — check the status of a prior authorization
    pa_requirements — look up PA requirements for a procedure/policy
    approve_prior_auth — Approve a prior authorization request. HIGH-IMPACT: requires human approval
    deny_prior_auth — Deny a prior authorization request. HIGH-IMPACT: requires human approval
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api-gateway:8443/mcp/pa"


class PAServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the PA Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_PA_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("pa_services", base_url)
        
        self.register_server("pa_services")

        self.register_tool("pa_lookup")
        self.register_tool("pa_status")
        self.register_tool("pa_requirements")
        self.register_tool("approve_prior_auth")
        self.register_tool("deny_prior_auth")
        self.register_tool("member_prior_authorizations")
        
        logger.info(
            "PAServicesMCPToolClient ready (%s) — tools: %s",
            base_url, self._registered_names,
        )