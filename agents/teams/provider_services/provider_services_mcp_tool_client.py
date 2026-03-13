"""
Provider Services MCP Tool Client
=========================================
Connects to the Provider Services FastMCP server and exposes its tools
as LangChain Tool objects.

Server default: http://provider-services-mcp-tools:8001
Env override:   MCP_PROVIDER_SERVICES_HTTP_URL

Tools:
    provider_lookup — look up provider information by provider ID
    provider_search_by_specialty — search for providers by specialty and location (zip code)
    provider_network_check  — check if a provider has claim history under a policy
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api-gateway:8443/mcp/provider"


class ProviderServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Provider Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_PROVIDER_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("provider_services", base_url)
        
        self.register_server("provider_services")

        self.register_tool("provider_lookup")
        self.register_tool("provider_search_by_specialty")
        self.register_tool("provider_network_check")

        logger.info(
            "ProviderServicesMCPToolClient ready (%s) — tools: %s",
            base_url, self._registered_names,
        )