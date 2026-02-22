"""
RemoteMCPNode - LangGraph-compatible node wrapper for remote MCP agents.

Transparently handles:
  1. Secure envelope construction (AES-256-GCM + HMAC-SHA256)
  2. HTTP transport to the remote MCP agent (with optional mTLS)
  3. Secure envelope verification on the response
  4. Prometheus metrics for encryption, signing, and replay events

Usage in central_supervisor.py:
    node = RemoteMCPNode(
        agent_name="member_services_team",
        base_url="https://mcp-member:8443",
    )
    workflow.add_node("member_services_team", node)
"""

import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from agents.core.state import SupervisorState
from security.message_encryption import (
    SecureMessageBus,
    SecurityError,
    get_secure_message_bus,
)
from security.schemas.tool_schemas import build_schema_registry

logger = logging.getLogger(__name__)

# Supervisor identity used in envelope from_agent field
SUPERVISOR_AGENT_ID = "central_supervisor"


class RemoteMCPNode:
    """
    A LangGraph node that delegates execution to a remote MCP agent
    over an encrypted HTTP channel.

    The node is callable with ``(state: SupervisorState) -> SupervisorState``
    so it can be added directly to a LangGraph ``StateGraph``.
    """

    def __init__(
        self,
        agent_name: str,
        base_url: str,
        secure_bus: Optional[SecureMessageBus] = None,
        timeout_seconds: int = 30,
        verify_tls: bool = True,
        client_cert: Optional[tuple] = None,
    ):
        """
        Args:
            agent_name: Logical name of the remote agent (e.g. "member_services_team").
            base_url: Base URL of the remote MCP agent (e.g. "https://mcp-member:8443").
            secure_bus: SecureMessageBus instance.  Uses global singleton if None.
            timeout_seconds: HTTP request timeout.
            verify_tls: Whether to verify the server's TLS certificate.
            client_cert: Optional (cert, key) tuple for mTLS client authentication.
        """
        self.agent_name = agent_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.verify_tls = verify_tls
        self.client_cert = client_cert

        # Initialise secure bus with schema registry
        if secure_bus is not None:
            self.secure_bus = secure_bus
        else:
            registry = build_schema_registry()
            self.secure_bus = get_secure_message_bus(schema_registry=registry)

    # ── LangGraph node interface ──────────────────────────────────────────

    def __call__(self, state: SupervisorState) -> SupervisorState:
        """
        Execute the remote MCP agent call with encrypted communication.

        Steps:
            1. Extract query and context from LangGraph state
            2. Build secure envelope (encrypt + sign)
            3. POST to remote MCP agent
            4. Verify + decrypt response envelope
            5. Merge results back into SupervisorState
        """
        start = time.time()
        user_id = state.get("user_id", "unknown")
        user_role = state.get("user_role", "CSR_TIER1")
        session_id = state.get("session_id", "default")

        # Extract the user query from messages
        messages = state.get("messages", [])
        if messages:
            query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        else:
            query = ""

        # Build the payload for the remote agent
        payload = {
            "query": query,
            "user_id": user_id,
            "user_role": user_role,
            "session_id": session_id,
        }

        # Add plan context if available
        if state.get("plan"):
            payload["plan"] = state["plan"]

        try:
            # ── Step 1: Wrap in secure envelope ──
            tool_name = f"{self.agent_name}_invoke"
            envelope = self.secure_bus.wrap_message(
                from_agent=SUPERVISOR_AGENT_ID,
                to_agent=self.agent_name,
                tool_name=tool_name,
                payload=payload,
            )

            # Track encryption metric
            self._track_metric("encrypt", self.agent_name)

            # ── Step 2: Send to remote MCP agent ──
            response = self._send_request(envelope)

            # ── Step 3: Unwrap response ──
            result_payload = self.secure_bus.unwrap_response(response)
            self._track_metric("decrypt", self.agent_name)

            # ── Step 4: Merge into state ──
            elapsed_ms = (time.time() - start) * 1000
            execution_path = state.get("execution_path", [])
            execution_path.append(
                f"central_supervisor -> {self.agent_name} (remote MCP, {elapsed_ms:.0f}ms)"
            )

            return_state: Dict[str, Any] = {
                "messages": result_payload.get("messages", state.get("messages", [])),
                "execution_path": execution_path,
                "tool_results": result_payload.get("tool_results", {}),
            }

            # Preserve error handling fields if present in response
            for field in ("error", "error_count", "error_history", "retry_count"):
                if field in result_payload:
                    return_state[field] = result_payload[field]

            return return_state

        except SecurityError as exc:
            logger.error(
                "Security error communicating with %s: %s",
                self.agent_name, exc,
            )
            self._track_metric("security_failure", self.agent_name)
            return self._error_state(state, f"Security error: {exc}")

        except requests.RequestException as exc:
            logger.error(
                "HTTP error communicating with %s: %s",
                self.agent_name, exc,
            )
            self._track_metric("transport_failure", self.agent_name)
            return self._error_state(state, f"Transport error: {exc}")

        except Exception as exc:
            logger.error(
                "Unexpected error communicating with %s: %s",
                self.agent_name, exc,
            )
            return self._error_state(state, f"Unexpected error: {exc}")

    # ── HTTP transport ────────────────────────────────────────────────────

    def _send_request(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST the encrypted envelope to the remote MCP agent and return
        the response envelope.
        """
        url = f"{self.base_url}/mcp/invoke"
        headers = {"Content-Type": "application/json"}

        kwargs: Dict[str, Any] = {
            "url": url,
            "headers": headers,
            "json": envelope,
            "timeout": self.timeout,
            "verify": self.verify_tls,
        }
        if self.client_cert:
            kwargs["cert"] = self.client_cert

        resp = requests.post(**kwargs)
        resp.raise_for_status()

        return resp.json()

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _error_state(state: SupervisorState, error_msg: str) -> Dict[str, Any]:
        """Build an error state to return to the graph."""
        execution_path = state.get("execution_path", [])
        execution_path.append(f"ERROR: {error_msg}")
        return {
            "messages": state.get("messages", []),
            "execution_path": execution_path,
            "error": error_msg,
            "error_count": state.get("error_count", 0) + 1,
        }

    @staticmethod
    def _track_metric(event: str, agent_name: str) -> None:
        """Safely increment Prometheus metrics (fail-silent)."""
        try:
            from observability.prometheus_metrics import (
                track_mcp_encryption_event,
            )
            track_mcp_encryption_event(event=event, agent_name=agent_name)
        except Exception:
            pass  # Metrics are best-effort
