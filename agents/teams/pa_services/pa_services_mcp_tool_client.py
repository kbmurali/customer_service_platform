"""
PA Services MCP Tool Client (v26)
===================================
Connects to the Prior Authorization Services FastMCP server and exposes
its tools as LangChain Tool objects.

Server default: http://mcp-pa:8003
Env override:   MCP_PA_SERVICES_HTTP_URL

Tools:
    pa_lookup_tool       — look up prior authorization information by PA ID
    pa_status_tool       — check the status of a prior authorization
    pa_requirements_tool — look up PA requirements for a procedure/policy
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://mcp-pa:8003"


class PAServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the PA Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_PA_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("pa_services", base_url)

        self.register_tool(
            name="pa_lookup_tool",
            description="Look up prior authorization information by PA ID. Input: JSON with pa_id.",
        )
        self.register_tool(
            name="pa_status_tool",
            description="Check the status of a prior authorization. Input: JSON with pa_id.",
        )
        self.register_tool(
            name="pa_requirements_tool",
            description="Look up PA requirements for a procedure/policy. Input: JSON with procedure_code and policy_type.",
        )

        logger.info(
            "PAServicesMCPToolClient created (%s) with %d tools",
            base_url, len(self._tools),
        )