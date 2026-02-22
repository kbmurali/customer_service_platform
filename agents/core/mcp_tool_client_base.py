"""
MCP Tool Client — Base
=======================
Shared infrastructure for all team-specific MCP tool clients.

Uses langchain_mcp_adapters.MultiServerMCPClient for tool discovery and
async→sync wrapping. No manual JSON-RPC construction needed.

user_id and user_role are passed as tool arguments at call time.

Usage:
    client = MemberServicesMCPToolClient()
    tools = client.get_langchain_tools()
    tool  = client.get_tool("member_lookup")
"""

import asyncio
import logging
import os
import ssl
from typing import Any, Dict, List

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool, StructuredTool

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

MCP_HTTP_TIMEOUT = float(os.getenv("MCP_HTTP_TIMEOUT", "30"))

# ── mTLS cert paths (injected by Docker Swarm secrets) ───────────────────────
# Secrets are auto-mounted at /run/secrets/<NAME> by Swarm.
# Override via env vars for local development (e.g. point to local cert files).
MCP_CLIENT_CERT = os.getenv("MCP_CLIENT_CERT", "/run/secrets/MCP_CLIENT_CERT")
MCP_CLIENT_KEY  = os.getenv("MCP_CLIENT_KEY",  "/run/secrets/MCP_CLIENT_KEY")
MCP_CA_CERT     = os.getenv("MCP_CA_CERT",     "/run/secrets/MCP_CA_CERT")


# ─────────────────────────────────────────────────────────────
# Sync wrapper
# ─────────────────────────────────────────────────────────────

def _make_sync_tool(async_tool: BaseTool) -> BaseTool:
    """
    Convert an async MCP tool to a sync StructuredTool.
    user_id and user_role are passed through as regular kwargs at call time.
    """
    async_fn = async_tool.coroutine

    def sync_fn(**kwargs):
        return asyncio.run(async_fn(**kwargs))

    return StructuredTool.from_function(
        sync_fn,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )


# ─────────────────────────────────────────────────────────────
# MCPToolClient (base)
# ─────────────────────────────────────────────────────────────

class MCPToolClient:
    """
    Base client for a remote FastMCP tool server.

    Subclasses call register_server() then register_tool() in __init__.
    """

    def __init__(self, team_name: str, base_url: str):
        self.team_name = team_name
        self.base_url = base_url.rstrip("/")
        self._server_name: str = ""
        self._async_tool_map: Dict[str, BaseTool] = {}
        self._registered_names: List[str] = []

    # ── Server + tool registration ────────────────────────────

    def register_server(self, server_name: str) -> None:
        """
        Discover and cache all tools from the MCP server.
        Called once in subclass __init__ before register_tool().
        """
        self._server_name = server_name
        # Build httpx_client_factory with mTLS if cert files exist
        if all(os.path.exists(p) for p in [MCP_CLIENT_CERT, MCP_CLIENT_KEY, MCP_CA_CERT]):
            ssl_context = ssl.create_default_context(cafile=MCP_CA_CERT)
            ssl_context.load_cert_chain(MCP_CLIENT_CERT, MCP_CLIENT_KEY)
            def httpx_client_factory(**kwargs):
                return httpx.AsyncClient(verify=ssl_context, **kwargs)
            logger.info("mTLS enabled for %s", server_name)
        else:
            httpx_client_factory = None
            logger.warning(
                "mTLS cert files not found — connecting without mTLS. "
                "Expected: %s, %s, %s", MCP_CLIENT_CERT, MCP_CLIENT_KEY, MCP_CA_CERT
            )

        config = {
            server_name: {
                "transport": "streamable_http",
                "url": f"{self.base_url}/mcp",
                "headers": {"Accept": "application/json, text/event-stream"},
                # Factory function injects mTLS ssl_context at transport level
                "httpx_client_factory": httpx_client_factory,
            }
        }

        async def _fetch():
            client = MultiServerMCPClient(config)
            return await client.get_tools(server_name=server_name)

        try:
            tools = asyncio.run(_fetch())
            self._async_tool_map = {t.name: t for t in tools}
            logger.info(
                "Connected to %s — discovered tools: %s",
                server_name, list(self._async_tool_map.keys()),
            )
        except Exception as e:
            logger.error("Failed to connect to %s (%s/mcp): %s", server_name, self.base_url, e)
            raise

    def register_tool(self, name: str, description: str = "") -> None:
        """
        Register a tool by name for use via get_tool() / get_langchain_tools().
        Validates the tool exists on the server.
        """
        if name not in self._async_tool_map:
            raise RuntimeError(
                f"Tool '{name}' not found on {self._server_name}. "
                f"Available: {list(self._async_tool_map.keys())}"
            )
        if name not in self._registered_names:
            self._registered_names.append(name)
        logger.debug("Registered tool: %s from %s", name, self._server_name)

    # ── Tool access ───────────────────────────────────────────

    def get_tool(self, name: str) -> BaseTool:
        """Return a single sync tool by name."""
        async_tool = self._async_tool_map.get(name)
        if async_tool is None:
            raise RuntimeError(
                f"Tool '{name}' not found on {self._server_name}. "
                f"Available: {list(self._async_tool_map.keys())}"
            )
        return _make_sync_tool(async_tool)

    def get_langchain_tools(self) -> List[BaseTool]:
        """Return all registered tools as sync LangChain tools."""
        return [_make_sync_tool(self._async_tool_map[name]) for name in self._registered_names]

    # ── Health check ──────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the remote MCP server via GET /health."""
        try:
            if all(os.path.exists(p) for p in [MCP_CLIENT_CERT, MCP_CLIENT_KEY, MCP_CA_CERT]):
                import ssl as _ssl
                ssl_ctx = _ssl.create_default_context(cafile=MCP_CA_CERT)
                ssl_ctx.load_cert_chain(MCP_CLIENT_CERT, MCP_CLIENT_KEY)
                verify: Any = ssl_ctx
            else:
                verify = True
            with httpx.Client(timeout=5, verify=verify) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "service": self.team_name}