"""
Claim Services MCP Tool Client
=================================
Connects to the Claim Services FastMCP server and exposes its tools
as LangChain Tool objects.

Server default: http://mcp-claim:8002
Env override:   MCP_CLAIM_SERVICES_HTTP_URL

Tools:
    claim_lookup_tool       — look up claim information by claim ID
    claim_status_tool       — check the status of a claim by claim number
    claim_payment_info_tool — get payment information for a claim
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://mcp-claim:8002"


class ClaimServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Claim Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_CLAIM_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("claim_services", base_url)

        self.register_tool(
            name="claim_lookup_tool",
            description="Look up claim information by claim ID. Input: JSON with claim_id.",
        )
        self.register_tool(
            name="claim_status_tool",
            description="Check the status of a claim by claim number. Input: JSON with claim_number.",
        )
        self.register_tool(
            name="claim_payment_info_tool",
            description="Get payment information for a claim. Input: JSON with claim_id.",
        )

        logger.info(
            "ClaimServicesMCPToolClient created (%s) with %d tools",
            base_url, len(self._tools),
        )