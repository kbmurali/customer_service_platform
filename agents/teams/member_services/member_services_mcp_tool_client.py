"""
Member Services MCP Tool Client
================================
Connects to the Member Services FastMCP server and exposes its tools
as sync LangChain tools via langchain_mcp_adapters.

Server default: http://mcp-member:8001
Env override:   MCP_MEMBER_SERVICES_HTTP_URL

Tools registered:
    member_lookup      — look up member information by member ID
    check_eligibility  — check member eligibility for a service date
    coverage_lookup    — get coverage details from the member's active policy
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api-gateway:8443/member-services"


class MemberServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Member Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_MEMBER_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("member_services", base_url)

        self.register_server("member_services")

        self.register_tool("member_lookup")
        self.register_tool("check_eligibility")
        self.register_tool("coverage_lookup")

        logger.info(
            "MemberServicesMCPToolClient ready (%s) — tools: %s",
            base_url, self._registered_names,
        )