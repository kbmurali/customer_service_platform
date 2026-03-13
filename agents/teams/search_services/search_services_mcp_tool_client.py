"""
Search Services MCP Tool Client
=======================================
Connects to the Search Services FastMCP server and exposes its tools
as LangChain Tool objects.

Server default: http://search-services-mcp-tools:8001
Env override:   MCP_SEARCH_SERVICES_HTTP_URL

Tools:
    search_medical_codes  — search for CPT/ICD-10 codes
    search_knowledge_base — search FAQs, clinical guidelines, and regulations
    search_policy_info    — search policy documents using semantic search
"""

import logging
import os

from agents.core.mcp_tool_client_base import MCPToolClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api-gateway:8443/mcp/search"


class SearchServicesMCPToolClient(MCPToolClient):
    """MCP tool client for the Search Services FastMCP server."""

    def __init__(self, base_url: str = None):
        base_url = base_url or os.getenv("MCP_SEARCH_SERVICES_HTTP_URL", _DEFAULT_URL)
        super().__init__("search_services", base_url)
        
        self.register_server("search_services")

        self.register_tool("search_knowledge_base")
        self.register_tool("search_medical_codes")
        self.register_tool("search_policy_info")

        logger.info(
            "SearchServicesMCPToolClient ready (%s) — tools: %s",
            base_url, self._registered_names,
        )