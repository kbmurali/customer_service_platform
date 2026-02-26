"""
Claim Services MCP Tool Client
=================================
Connects to the Claim Services FastMCP server and exposes its tools
as LangChain Tool objects.

Server default: http://mcp-claim:8002
Env override:   MCP_CLAIM_SERVICES_HTTP_URL

Tools registered:
    claim_lookup       — look up claim information by claim ID
    claim_status       — check the status of a claim by claim number
    claim_payment_info — get payment information for a claim
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api-gateway:8443/claims-services"


class ClaimServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Claim Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_CLAIM_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("claim_services", base_url)

        self.register_server("claims_services")

        self.register_tool("claim_lookup")
        self.register_tool("claim_status")
        self.register_tool("claim_payment_info")

        logger.info(
            "ClaimsServicesMCPToolClient ready (%s) — tools: %s",
            base_url, self._registered_names,
        )