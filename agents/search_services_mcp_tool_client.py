"""
Search Services MCP Tool Client (v26)
=======================================
Connects to the Search Services FastMCP server and exposes its tools
as LangChain Tool objects.

Previously, search tools (search_medical_codes, search_knowledge_base,
search_policy_info) were duplicated across the claim, PA, and provider
service clients. They are now consolidated here as a dedicated search
service client with a single server.

Server default: http://mcp-search:8005
Env override:   MCP_SEARCH_SERVICES_HTTP_URL

Tools:
    search_medical_codes_tool  — search for CPT/ICD-10 codes
    search_knowledge_base_tool — search FAQs, clinical guidelines, and regulations
    search_policy_info_tool    — search policy documents using semantic search
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://mcp-search:8005"


class SearchServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Search Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_SEARCH_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("search_services", base_url)

        self.register_tool(
            name="search_medical_codes_tool",
            description="Search for CPT/ICD-10 codes. Input: JSON with query and optional code_type.",
        )
        self.register_tool(
            name="search_knowledge_base_tool",
            description="Search FAQs, clinical guidelines, and regulations. Input: JSON with query.",
        )
        self.register_tool(
            name="search_policy_info_tool",
            description="Search policy documents using semantic similarity search. Input: JSON with query.",
        )

        logger.info(
            "SearchServicesMCPToolClient created (%s) with %d tools",
            base_url, len(self._tools),
        )