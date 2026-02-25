"""
A2A Client Node
================
Replaces RemoteMCPNode for inter-supervisor (agent-to-agent) communication.

Key differences from RemoteMCPNode:
    - Uses A2A protocol task lifecycle (submitted → working → completed/failed)
      instead of a single synchronous POST to /mcp/invoke.
    - Sends tasks via POST /a2a/tasks/send with A2A Message/Part structure.
    - Supports task state polling for long-running operations.
    - Preserves SecureMessageBus encryption as a transport-layer enhancement
      on top of A2A's JSON-RPC messages.
    - Tracks execution in Context Graph (Neo4j) for observability.

MCP continues to be used for worker-tool integration within each team.
A2A with mTLS is used exclusively for inter-supervisor communication.

A2A Task Lifecycle:
    submitted  → The task has been received by the remote agent
    working    → The remote agent is actively processing the task
    input-required → The remote agent needs additional information
    completed  → The task finished successfully with artifacts
    failed     → The task encountered an unrecoverable error
    canceled   → The task was canceled by the client
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional
from functools import lru_cache

import os
import ssl
import httpx

from langchain_core.messages import AIMessage

from security.message_encryption import get_secure_message_bus            
from agents.core.state import SupervisorState
from agents.core.context_compressor import get_cross_agent_compressor
from config.settings import get_settings
from databases.context_graph_data_access import get_cg_data_access
from observability.prometheus_metrics import track_mcp_encryption_event
from observability.langfuse_integration import get_langfuse_tracer

logger = logging.getLogger(__name__)
settings = get_settings()

# ── mTLS cert paths ───────────────────────────────────────────────────────────
# Docker Swarm secrets are auto-mounted at /run/secrets/<NAME>.
# Override via env vars for local development.
_MCP_CLIENT_CERT = os.getenv("MCP_CLIENT_CERT", "/run/secrets/MCP_CLIENT_CERT").strip()
_MCP_CLIENT_KEY  = os.getenv("MCP_CLIENT_KEY",  "/run/secrets/MCP_CLIENT_KEY").strip()
_MCP_CA_CERT     = os.getenv("MCP_CA_CERT",     "/run/secrets/MCP_CA_CERT").strip()


def _build_ssl_context() -> ssl.SSLContext | None:
    """
    Build an SSL context with mTLS certs if all cert files exist.
    Returns None if cert files are not found (falls back to default TLS).
    """
    if all(os.path.exists(p) for p in [_MCP_CLIENT_CERT, _MCP_CLIENT_KEY, _MCP_CA_CERT]):
        ctx = ssl.create_default_context(cafile=_MCP_CA_CERT)
        ctx.load_cert_chain(_MCP_CLIENT_CERT, _MCP_CLIENT_KEY)
        logger.info("A2AClientNode: mTLS enabled")
        return ctx
    logger.warning(
        "A2AClientNode: mTLS cert files not found — falling back to default TLS. "
        "Expected: %s, %s, %s", _MCP_CLIENT_CERT, _MCP_CLIENT_KEY, _MCP_CA_CERT
    )
    return None


# ---------------------------------------------------------------------------
# A2A Task State Constants
# ---------------------------------------------------------------------------

class A2ATaskState:
    """A2A protocol task states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# ---------------------------------------------------------------------------
# A2A Client Node
# ---------------------------------------------------------------------------

class A2AClientNode:
    """
    LangGraph node that delegates work to a remote team supervisor via A2A.

    This node is used to communicate with remote team supervisors 
    (member_services, claim_services) that are deployed as separate services.

    Communication flow:
        1. Build an A2A Task with Message containing the query and context
        2. Encrypt the payload using SecureMessageBus (transport security)
        3. POST to the remote agent's /a2a/tasks/send endpoint
        4. Receive the A2A Task response with state and artifacts
        5. Decrypt and extract the result into SupervisorState

    The remote agent's URL is discovered from its Agent Card (fetched at
    startup or on-demand) rather than hardcoded environment variables.

    Args:
        agent_name:     Logical name (e.g., "member_services_team")
        agent_url:      A2A endpoint URL from Agent Card (e.g., "https://host:8443/a2a")
        shared_secret:  Shared secret for SecureMessageBus HMAC/AES-GCM encryption
    """

    def __init__(
        self,
        agent_name: str,
        agent_url: str,
        schema_registry: dict,
        shared_secret: Optional[str] = None,
        from_agent_name: str = "central_supervisor"
    ):
        self.agent_name = agent_name
        self.agent_url = agent_url
        self.schema_registry = schema_registry
        self.shared_secret = shared_secret
        self.from_agent_name = from_agent_name

        # Lazy-initialized singletons
        self._secure_bus = None
        self._cg_data_access = None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------
    @lru_cache
    def _get_secure_bus(self):
        """Return (and cache) the SecureMessageBus singleton."""
        if self._secure_bus is None:
            self._secure_bus = get_secure_message_bus(
                schema_registry=self.schema_registry
            )
        return self._secure_bus

    @lru_cache
    def _get_cg(self):
        """Return (and cache) the Context Graph data access singleton."""
        if self._cg_data_access is None:
            self._cg_data_access = get_cg_data_access()
            
        return self._cg_data_access

    # ------------------------------------------------------------------
    # Observability helpers (best-effort, never raise)
    # ------------------------------------------------------------------
    def _track_prometheus(self, event: str) -> None:
        """Track A2A communication events in Prometheus."""
        try:
            track_mcp_encryption_event(event=event, agent_name=self.agent_name)
        except Exception:
            pass

    def _trace_langfuse(
        self,
        session_id: str,
        user_id: str,
        status: str,
        elapsed_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Trace A2A client request in Langfuse."""
        try:
            tracer = get_langfuse_tracer()
            tracer.trace_agent_execution(
                name=f"a2a_client_{self.agent_name}",
                agent_type="a2a_client",
                input_data={"agent": self.agent_name, "url": self.agent_url},
                output_data={
                    "status": status,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                metadata={"error": error} if error else {},
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            pass

    def _track_cg(
        self,
        session_id: str,
        status: str,
        elapsed_ms: float,
        task_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Track A2A client request in Context Graph."""
        try:
            cg = self._get_cg()
            cg.track_agent_execution(
                session_id=session_id,
                agent_name=f"a2a_client_{self.agent_name}",
                agent_type="a2a_client",
                status=status,
                metadata={
                    "elapsed_ms": round(elapsed_ms, 1),
                    "a2a_task_id": task_id,
                    "error": error,
                },
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # A2A Task Construction
    # ------------------------------------------------------------------
    def _build_a2a_task(
        self,
        query: str,
        user_id: str,
        user_role: str,
        session_id: str,
        plan: Optional[Dict[str, Any]] = None,
        central_step_id: str = "",
    ) -> Dict[str, Any]:
        """
        Build an A2A Task envelope with Message and Parts.

        Parts emitted:
            1. {"type": "text",  "text": "<query>"}
            2. {"type": "data",  "data": {user_id, user_role, session_id}}
            3. {"type": "data",  "data": {"plan": ...}}         (optional)
            4. {"type": "data",  "data": {"central_step_id": ...}} (when delegating)

        central_step_id is the Step.stepId from the central supervisor's
        plan that is delegating this work.  The receiving team supervisor
        stores it to create:
            (CentralStep)-[:DELEGATED_TO]->(TeamPlan)
        """
        task_id = str(uuid.uuid4())

        parts = [
            {"type": "text", "text": query},
            {
                "type": "data",
                "data": {
                    "user_id":    user_id,
                    "user_role":  user_role,
                    "session_id": session_id,
                },
            },
        ]

        if plan:
            parts.append({"type": "data", "data": {"plan": plan}})

        if central_step_id:
            parts.append({"type": "data", "data": {"central_step_id": central_step_id}})

        return {
            "id": task_id,
            "message": {
                "role": "user",
                "parts": parts,
            },
        }

    # ------------------------------------------------------------------
    # A2A Response Parsing
    # ------------------------------------------------------------------
    def _parse_a2a_response(
        self, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse an A2A Task response into a flat result dict.

        The remote agent returns:
            {
                "id": "<task_id>",
                "state": "completed" | "failed" | ...,
                "artifacts": [
                    {
                        "parts": [
                            {"type": "text", "text": "<answer>"},
                            {"type": "data", "data": {<structured_result>}}
                        ]
                    }
                ],
                "history": [...]  // optional state transition history
            }

        This method extracts the text and data parts from artifacts into
        a flat dict compatible with SupervisorState.
        """
        task_state = response_data.get("state", A2ATaskState.FAILED)
        task_id = response_data.get("id", "unknown")

        if task_state == A2ATaskState.FAILED:
            error_msg = "Remote agent task failed"
            # Try to extract error message from artifacts
            for artifact in response_data.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if part.get("type") == "text":
                        error_msg = part["text"]
                        break
            return {
                "error": error_msg,
                "a2a_task_id": task_id,
                "a2a_task_state": task_state,
            }

        # Extract results from artifacts
        messages = []
        tool_results = {}
        execution_path = []
        error_fields = {}

        for artifact in response_data.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    messages.append(part["text"])
                elif part.get("type") == "data":
                    data = part.get("data", {})
                    if "tool_results" in data:
                        tool_results.update(data["tool_results"])
                    if "execution_path" in data:
                        execution_path.extend(data["execution_path"])
                    # Propagate error fields if present
                    for field in ("error", "error_count", "error_history", "retry_count"):
                        if field in data:
                            error_fields[field] = data[field]

        result = {
            "messages": messages,
            "tool_results": tool_results,
            "execution_path": execution_path,
            "a2a_task_id": task_id,
            "a2a_task_state": task_state,
        }
        
        result.update(error_fields)
        
        return result

    # ------------------------------------------------------------------
    # Main Invocation (LangGraph node __call__)
    # ------------------------------------------------------------------
    def __call__(self, state: SupervisorState) -> SupervisorState:
        """
        LangGraph node entry point.

        Called by a LangGraph graph when routing to this remote
        team. Sends an A2A task to the remote agent and returns the result
        merged into SupervisorState.

        Steps:
            1. Extract query and context from state
            2. Build A2A Task with Message/Parts
            3. Encrypt via SecureMessageBus
            4. POST to remote agent's /a2a/tasks/send
            5. Decrypt response
            6. Parse A2A Task response into SupervisorState fields
            7. Track in Prometheus, Langfuse, and Context Graph
        """
        start = time.time()

        # ── Extract from state ──
        messages = state.get("messages", [])
        
        query = ""
        
        if messages:
            last_msg = messages[-1]
            query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        user_id = state.get("user_id", "unknown")
        user_role = state.get("user_role", "unknown")
        session_id = state.get("session_id", "default")
        plan = state.get("plan")

        # ── Step 0: Cross-agent context compression ──
        # Compress accumulated state before sending to the remote agent
        # to prevent token budget exhaustion in downstream LLM calls.
        compressed_plan = plan
        try:
            cross_compressor = get_cross_agent_compressor()
            compressed_ctx = cross_compressor.compress_delegation_context(
                plan=plan,
                conversation_summary=str(query),  # query carries the latest context
                prior_results=state.get("tool_results"),
            )
            compressed_plan = compressed_ctx.get("plan", plan)
            logger.debug(
                "Cross-agent compression applied for %s delegation",
                self.agent_name,
            )
        except Exception as exc:
            logger.warning(
                "Cross-agent compression failed (non-fatal): %s", exc
            )
            # Proceed with uncompressed plan

        # ── Step 1: Build A2A Task ──
        # Derive the Step.stepId for the current goal so the team supervisor
        # can link its TeamPlan back to this central step in the CG:
        #   (CentralStep)-[:DELEGATED_TO]->(TeamPlan)
        central_step_id = ""
        try:
            _idx     = state.get("current_goal_index", 0)
            _plan    = state.get("plan") or {}
            _goals   = _plan.get("goals", [])
            _steps   = _plan.get("steps", [])
            _gid     = _goals[_idx].get("id", "") if _idx < len(_goals) else ""
            central_step_id = next(
                (s.get("step_id", "") for s in _steps if s.get("goal_id") == _gid), ""
            )
        except Exception:
            pass  # Non-fatal — traceability degrades gracefully

        a2a_task = self._build_a2a_task(
            query=str(query),
            user_id=user_id,
            user_role=user_role,
            session_id=session_id,
            plan=compressed_plan,
            central_step_id=central_step_id,
        )
        task_id = a2a_task["id"]

        logger.info(
            "A2A task %s -> %s: sending to %s",
            task_id,
            self.agent_name,
            self.agent_url,
        )

        try:
            # ── Step 2: Encrypt payload ──
            secure_bus = self._get_secure_bus()
            encrypted_envelope = secure_bus.wrap_message(
                from_agent=self.from_agent_name,
                to_agent=self.agent_name,
                tool_name=f"{self.agent_name}_a2a_task",
                payload=a2a_task,
            )
            self._track_prometheus("encrypt")

            # ── Step 3: Send A2A task ──
            # Build the full endpoint URL: {agent_url}/tasks/send
            tasks_url = f"{self.agent_url}/tasks/send"

            ssl_context = _build_ssl_context()
            with httpx.Client(
                verify=ssl_context if ssl_context is not None else settings.MCP_VERIFY_TLS,
                timeout=settings.AGENT_TIMEOUT_SECONDS,
            ) as client:
                response = client.post(tasks_url, json=encrypted_envelope)
                response.raise_for_status()
                response_envelope = response.json()

            # ── Step 4: Decrypt response ──
            response_payload = secure_bus.unwrap_message(response_envelope)
            self._track_prometheus("decrypt")

            # ── Step 5: Parse A2A response ──
            result = self._parse_a2a_response(response_payload)

            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                "A2A task %s completed in %.0f ms (state=%s)",
                task_id,
                elapsed_ms,
                result.get("a2a_task_state", "unknown"),
            )

            # ── Step 6: Observability ──
            self._trace_langfuse(
                session_id=session_id,
                user_id=user_id,
                status="success",
                elapsed_ms=elapsed_ms,
            )
            self._track_cg(
                session_id=session_id,
                status="success",
                elapsed_ms=elapsed_ms,
                task_id=task_id,
            )

            # ── CG: link (a2a_client)-[:CALLED_AGENT]->(a2a_server) ────────
            # Matched via shared a2a_task_id. Works generically across all
            # teams — member_services, claims, provider, etc.
            try:
                cg = self._get_cg()
                cg.link_a2a_client_to_server(
                    session_id=session_id,
                    a2a_task_id=task_id,
                )
            except Exception as e:
                logger.warning("Failed to link a2a_client to server (non-fatal): %s", e)

            # ── Step 7: Build return state ──
            response_messages = result.get("messages", [])
            combined_text = "\n".join(response_messages) if response_messages else ""

            return_state: SupervisorState = {
                "messages": [AIMessage(content=combined_text)] if combined_text else [],
                "tool_results": result.get("tool_results", {}),
                "execution_path": state.get("execution_path", [])
                + [f"a2a_{self.agent_name}"]
                + result.get("execution_path", []),
            }

            # Propagate error fields
            if result.get("error"):
                return_state["error"] = result["error"]
                
            for field in ("error_count", "error_history", "retry_count"):
                if field in result:
                    return_state[field] = result[field]

            return return_state

        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.time() - start) * 1000
            error_msg = f"A2A HTTP error from {self.agent_name}: {exc.response.status_code}"
            logger.error("%s — %s", error_msg, exc)

            self._track_prometheus("transport_failure")
            
            self._trace_langfuse(
                session_id=session_id,
                user_id=user_id,
                status="http_error",
                elapsed_ms=elapsed_ms,
                error=error_msg,
            )
            self._track_cg(
                session_id=session_id,
                status="http_error",
                elapsed_ms=elapsed_ms,
                task_id=task_id,
                error=error_msg,
            )

            return {
                "messages": [AIMessage(content=f"Error communicating with {self.agent_name}: {error_msg}")],
                "error": error_msg,
                "execution_path": state.get("execution_path", [])
                + [f"a2a_{self.agent_name}_failed"],
            }

        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000
            error_msg = f"A2A error from {self.agent_name}: {exc}"
            logger.error(error_msg, exc_info=True)

            self._track_prometheus("transport_failure")
            self._trace_langfuse(
                session_id=session_id,
                user_id=user_id,
                status="error",
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
            self._track_cg(
                session_id=session_id,
                status="error",
                elapsed_ms=elapsed_ms,
                task_id=task_id,
                error=str(exc),
            )

            return {
                "messages": [AIMessage(content=f"Error communicating with {self.agent_name}: {exc}")],
                "error": str(exc),
                "execution_path": state.get("execution_path", [])
                + [f"a2a_{self.agent_name}_failed"],
            }