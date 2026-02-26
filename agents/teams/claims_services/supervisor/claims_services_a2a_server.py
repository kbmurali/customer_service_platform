"""
Claims Services Remote Agent Server (A2A Protocol)
=========================================================
Receives A2A task envelopes from A2AClientNode (central supervisor),
decrypts, executes the claims services supervisor's LangGraph subgraph,
and returns an A2A task response with artifacts.

Also serves the Agent Card at /.well-known/agent.json for A2A discovery.

Endpoint contract (matches A2AClientNode):
    GET  /.well-known/agent.json  — A2A Agent Card for discovery
    POST /a2a/tasks/send          — A2A task: encrypted envelope in, encrypted response out
    GET  /health                  — liveness probe
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage

from agents.core.state import SupervisorState

from security.message_encryption import get_secure_message_bus, SecurityError
from agents.teams.claims_services.supervisor.claims_services_supervisor import get_claims_services_graph
from agents.teams.claims_services.supervisor.tool_schemas import build_schema_registry
from agents.teams.claims_services.claims_services_a2a_agent_card import build_claims_services_agent_card
from observability.prometheus_metrics import track_mcp_encryption_event
from observability.langfuse_integration import get_langfuse_tracer
from databases.context_graph_data_access import get_cg_data_access

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: initialise heavy singletons before accepting requests
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: eagerly initialise the graph and secure bus so that
    asyncio.run() inside ClaimServicesMCPToolClient.__init__ runs
    BEFORE uvicorn's event loop is active.

    This avoids:
        RuntimeError: asyncio.run() cannot be called from a running event loop
    """
    logger.info("A2A server startup: initialising graph and secure bus...")
    _get_secure_bus()   # warms SecureMessageBus singleton
    _get_graph()        # triggers ClaimsServicesSupervisor.__init__ → workers → MCP client
    _get_agent_card()   # warms agent card singleton
    logger.info("A2A server startup complete — ready to accept requests.")
    yield
    # Nothing to clean up on shutdown


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Claims Services A2A Agent",
    version="1.0.0",
    description="Remote A2A agent for claims services with encrypted communication",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_NAME      = "claims_services_supervisor_agent"
SUPERVISOR_NAME = "claims_services_supervisor"

# ---------------------------------------------------------------------------
# Lazy singletons (initialised on first request to keep startup fast)
# ---------------------------------------------------------------------------
_secure_bus = None
_graph: Optional[CompiledStateGraph] = None
_agent_card = None


def _get_secure_bus():
    """Return (and cache) the SecureMessageBus singleton."""
    global _secure_bus

    if _secure_bus is None:
        _secure_bus = get_secure_message_bus(schema_registry=build_schema_registry())

    return _secure_bus


def _get_graph() -> CompiledStateGraph:
    """Return (and cache) the compiled claims services LangGraph."""
    global _graph

    if _graph is None:
        _graph = get_claims_services_graph()

    return _graph


def _get_agent_card():
    """Return (and cache) the Agent Card for this service."""
    global _agent_card

    if _agent_card is None:
        base_url = os.getenv("A2A_CLAIMS_SERVICES_URL", "https://api-gateway:8443/a2a/claims")
        _agent_card = build_claims_services_agent_card(base_url)

    return _agent_card


# ---------------------------------------------------------------------------
# Observability helpers (best-effort, never raises)
# ---------------------------------------------------------------------------
def _track(event: str) -> None:
    try:
        track_mcp_encryption_event(event=event, agent_name=AGENT_NAME)
    except Exception:
        pass


def _trace_langfuse(
    session_id: str,
    user_id: str,
    status: str,
    elapsed_ms: float,
    error: str | None = None,
) -> None:
    """Trace server-side request processing in Langfuse (best-effort)."""
    try:
        tracer = get_langfuse_tracer()
        tracer.trace_agent_execution(
            name=f"{AGENT_NAME}_server",
            agent_type="a2a_server",
            input_data={"agent": AGENT_NAME},
            output_data={"status": status, "elapsed_ms": round(elapsed_ms, 1)},
            metadata={"error": error} if error else {},
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        pass


def _track_cg(
    session_id: str,
    status: str,
    elapsed_ms: float,
    task_id: str | None = None,
    error: str | None = None,
) -> None:
    """Track server-side request processing in Context Graph (best-effort)."""
    try:
        cg = get_cg_data_access()
        cg.track_agent_execution(
            session_id=session_id,
            agent_name=AGENT_NAME,
            agent_type="a2a_server",
            status=status,
            metadata={
                "elapsed_ms": round(elapsed_ms, 1),
                "a2a_task_id": task_id,
                "error": error,
            },
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# A2A Helper: Extract query and metadata from A2A Task message parts
# ---------------------------------------------------------------------------
def _extract_from_a2a_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract query, user metadata, and plan from A2A Task message parts.

    A2A Task message structure:
        {
            "id": "<task_id>",
            "message": {
                "role": "user",
                "parts": [
                    {"type": "text", "text": "<query>"},
                    {"type": "data", "data": {"user_id": ..., "session_id": ...}},
                    {"type": "data", "data": {"plan": {...}}}  // optional
                ]
            }
        }
    """
    message = task.get("message", {})
    parts   = message.get("parts", [])

    query           = ""
    user_id         = "unknown"
    user_role       = "unknown"
    session_id      = "default"
    plan            = None
    central_step_id = ""

    for part in parts:
        part_type = part.get("type", "")
        if part_type == "text":
            query = part.get("text", "")
        elif part_type == "data":
            data = part.get("data", {})
            if "user_id"         in data: user_id         = data["user_id"]
            if "user_role"       in data: user_role       = data["user_role"]
            if "session_id"      in data: session_id      = data["session_id"]
            if "plan"            in data: plan            = data["plan"]
            if "central_step_id" in data: central_step_id = data["central_step_id"]

    return {
        "query":           query,
        "user_id":         user_id,
        "user_role":       user_role,
        "session_id":      session_id,
        "plan":            plan,
        "central_step_id": central_step_id,
    }


# ---------------------------------------------------------------------------
# A2A Helper: Build A2A Task response with artifacts
# ---------------------------------------------------------------------------
def _build_a2a_response(
    task_id: str,
    state: str,
    messages: list,
    tool_results: dict,
    execution_path: list,
    error_fields: dict | None = None,
) -> Dict[str, Any]:
    """
    Build an A2A Task response with artifacts.

    A2A response structure:
        {
            "id": "<task_id>",
            "state": "completed" | "failed",
            "artifacts": [
                {
                    "parts": [
                        {"type": "text", "text": "<combined_answer>"},
                        {"type": "data", "data": {"tool_results": ..., "execution_path": ...}}
                    ]
                }
            ]
        }
    """
    combined_text = "\n".join(messages) if messages else ""

    data_part: Dict[str, Any] = {
        "tool_results":  tool_results,
        "execution_path": execution_path,
    }

    if error_fields:
        data_part.update(error_fields)

    return {
        "id":    task_id,
        "state": state,
        "artifacts": [
            {
                "parts": [
                    {"type": "text", "text": combined_text},
                    {"type": "data", "data": data_part},
                ]
            }
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "agent": AGENT_NAME, "protocol": "a2a"}


@app.get("/.well-known/agent.json")
async def agent_card():
    """
    A2A Agent Card discovery endpoint.

    Returns the Agent Card JSON document describing this agent's
    capabilities, skills, and authentication requirements.
    The CentralSupervisor fetches this to discover available skills.
    """
    card = _get_agent_card()
    return JSONResponse(content=card.to_dict())


@app.post("/a2a/tasks/send")
async def a2a_tasks_send(request: Request):
    """
    Claims Services A2A task endpoint: receive an encrypted A2A task envelope,
    decrypt → execute LangGraph subgraph → encrypt response.

    Implements the A2A task lifecycle:
        submitted → working → completed (or failed)

    The SecureMessageBus encryption is preserved as a transport-layer
    enhancement on top of A2A's JSON-RPC messages.
    """
    start    = time.time()
    envelope: Dict[str, Any] = await request.json()

    secure_bus = _get_secure_bus()
    graph      = _get_graph()

    try:
        # ── Step 1: Unwrap (verify HMAC, check replay, AES-GCM decrypt) ──
        a2a_task = secure_bus.unwrap_message(envelope)
        _track("decrypt")

        task_id = a2a_task.get("id", str(uuid.uuid4()))

        logger.info(
            "Claims Services A2A task %s received from %s",
            task_id,
            envelope.get("from_agent", "?"),
        )

        # ── Step 2: Extract query and metadata from A2A Task parts ──
        extracted = _extract_from_a2a_task(a2a_task)

        logger.info(
            "Claims Services A2A task %s: user=%s session=%s query_len=%d",
            task_id,
            extracted["user_id"],
            extracted["session_id"],
            len(extracted["query"]),
        )

        # ── Step 3: Build LangGraph state from extracted data ──
        state: SupervisorState = {
            "messages":        [HumanMessage(content=extracted["query"])],
            "user_id":         extracted["user_id"],
            "user_role":       extracted["user_role"],
            "session_id":      extracted["session_id"],
            "execution_path":  [],
            "tool_results":    {},
            # CG traceability fields — tell the team supervisor what it is
            # and which central step delegated work here via A2A.
            "plan_type":       "team",
            "team_name":       "claims_services",
            "central_step_id": extracted["central_step_id"],
        }

        if extracted["plan"]:
            state["plan"] = extracted["plan"]

        # ── Step 4: Execute the claims services subgraph ──
        result = graph.invoke(state)

        # ── Step 5: Build A2A response with artifacts ──
        response_messages = [
            m.content if hasattr(m, "content") else str(m)
            for m in result.get("messages", [])
        ]

        error_fields = {}
        for field in ("error", "error_count", "error_history", "retry_count"):
            if field in result:
                error_fields[field] = result[field]

        a2a_state    = "failed" if result.get("error") else "completed"
        a2a_response = _build_a2a_response(
            task_id=task_id,
            state=a2a_state,
            messages=response_messages,
            tool_results=result.get("tool_results", {}),
            execution_path=result.get("execution_path", []),
            error_fields=error_fields if error_fields else None,
        )

        # ── Step 6: Wrap response in encrypted envelope ──
        tool_name = envelope.get("tool_name", f"{AGENT_NAME}_a2a_task")
        response_envelope = secure_bus.wrap_response(
            from_agent=AGENT_NAME,
            to_agent=envelope["from_agent"],
            tool_name=tool_name,
            payload=a2a_response,
        )
        _track("encrypt")

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Claims Services A2A task %s completed in %.0f ms (state=%s)",
            task_id,
            elapsed_ms,
            a2a_state,
        )

        # ── Langfuse + CG: trace successful execution ──
        _trace_langfuse(
            session_id=extracted["session_id"],
            user_id=extracted["user_id"],
            status="success",
            elapsed_ms=elapsed_ms,
        )
        _track_cg(
            session_id=extracted["session_id"],
            status="success",
            elapsed_ms=elapsed_ms,
            task_id=task_id,
        )

        # ── CG: link (a2a_server)-[:HAS_PLAN]->(Plan) ───────────────────────
        # Connects the A2A transport layer to the team execution graph,
        # enabling full end-to-end traceability from central supervisor
        # down to every tool call across every team.
        try:
            _plan_id = result.get("plan_id", "")
            if _plan_id and task_id:
                cg = get_cg_data_access()
                cg.link_a2a_server_to_plan(
                    session_id=extracted["session_id"],
                    a2a_task_id=task_id,
                    plan_id=_plan_id,
                )
        except Exception as e:
            logger.warning("Failed to link a2a_server to plan (non-fatal): %s", e)

        return JSONResponse(content=response_envelope)

    except Exception as exc:
        elapsed_ms   = (time.time() - start) * 1000
        session_id   = "default"
        user_id      = "unknown"
        task_id_err  = "unknown"
        try:
            task_id_err   = a2a_task.get("id", "unknown")       # type: ignore[union-attr]
            extracted_err = _extract_from_a2a_task(a2a_task)    # type: ignore[union-attr]
            session_id    = extracted_err.get("session_id", "default")
            user_id       = extracted_err.get("user_id", "unknown")
        except Exception:
            pass

        if isinstance(exc, SecurityError):
            logger.error("Security error on Claims Services A2A task %s: %s", task_id_err, exc)
            _track("security_failure")
            _trace_langfuse(session_id=session_id, user_id=user_id, status="security_error", elapsed_ms=elapsed_ms, error=str(exc))
            _track_cg(session_id=session_id, status="security_error", elapsed_ms=elapsed_ms, task_id=task_id_err, error=str(exc))
            raise HTTPException(status_code=403, detail=str(exc))

        logger.error("Execution error on Claims Services A2A task %s: %s", task_id_err, exc, exc_info=True)
        _track("transport_failure")
        _trace_langfuse(session_id=session_id, user_id=user_id, status="execution_error", elapsed_ms=elapsed_ms, error=str(exc))
        _track_cg(session_id=session_id, status="execution_error", elapsed_ms=elapsed_ms, task_id=task_id_err, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Backward compatibility: keep /mcp/invoke as deprecated alias
# ---------------------------------------------------------------------------
@app.post("/mcp/invoke")
async def invoke_legacy(request: Request):
    """
    Legacy endpoint for backward compatibility with RemoteMCPNode clients.
    Delegates to the Claims Services A2A task handler after wrapping the payload.

    Deprecated: Use POST /a2a/tasks/send instead.
    """
    logger.warning("Legacy /mcp/invoke called — migrate to /a2a/tasks/send")
    return await a2a_tasks_send(request)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("A2A_AGENT_PORT", os.getenv("MCP_AGENT_PORT", "8443"))),
    )
