"""
Provider Services MCP Tool Client (v26)
=========================================
Connects to the Provider Services FastMCP server and exposes its tools
as LangChain Tool objects.

Server default: http://mcp-provider:8004
Env override:   MCP_PROVIDER_SERVICES_HTTP_URL

Tools:
    provider_search_tool — search for providers by specialty and location
    provider_lookup_tool — look up provider information by provider ID
    network_check_tool   — check if a provider has claim history under a policy
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://mcp-provider:8004"


class ProviderServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Provider Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_PROVIDER_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("provider_services", base_url)

        self.register_tool(
            name="provider_search_tool",
            description="Search for providers by specialty and location. Input: JSON with specialty and zip_code.",
        )
        self.register_tool(
            name="provider_lookup_tool",
            description="Look up provider information by provider ID. Input: JSON with provider_id.",
        )
        self.register_tool(
            name="network_check_tool",
            description="Check if a provider has claim history under a policy. Input: JSON with provider_id and policy_id.",
        )

        logger.info(
            "ProviderServicesMCPToolClient created (%s) with %d tools",
            base_url, len(self._tools),
        )