"""
FastAPI for CSIP Agentic Access
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import sys
import os
import uuid
import warnings
import redis as redis_lib

# Suppress FutureWarning AND UserWarning from torch/transformers/huggingface_hub/spacy
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="spacy")      # [W095] model version mismatch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")       # _pytree deprecation

# Ensure the project root is on sys.path before any project imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings, Settings
from agents.security import auth_service, audit_logger, create_access_token, decode_access_token, AuthenticationError, rbac_service
from agents.request_processor import process_user_request, ProcessResult
from observability.metrics_persister import start_background_pusher
from security.guardrails_output_validation import get_output_validator
from security.dlp_scanner import get_dlp_scanner
from security.nemo_guardrails_integration import get_nemo_filter
from security.presidio_memory_security import get_presidio_security

logger = logging.getLogger(__name__)

# ── Configure root logger from LOG_LEVEL env var ──────────────────────────────
# Must be done before any other module emits log records.
# The compose file sets LOG_LEVEL=INFO; without this call the root logger
# defaults to WARNING and all INFO records from central_supervisor,
# a2a_client_node, request_processor etc. are silently discarded.
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
    force=True,   # Override any handler already attached by uvicorn/gunicorn
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)   # suppress /health noise
# Suppress Presidio config-missing WARNINGs and per-recognizer INFO (fires per init × N workers)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
# Suppress NeMo action_dispatcher "Added X to actions" INFO spam (30+ lines per worker)
logging.getLogger("nemoguardrails.actions.action_dispatcher").setLevel(logging.WARNING)
# Suppress noisy HuggingFace/transformers deprecation warnings during model load
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger.info("Logging configured at level: %s", os.getenv("LOG_LEVEL", "INFO"))

# ── Authoritative tool → service mapping ──────────────────────────────────
# Used by /api/stats/tool-usage, /api/admin/tool-permissions, and
# /api/admin/tool-catalog as a reliable fallback when A2A agent card
# builders fail (import errors, missing dependencies in API container).
TOOL_SERVICE_MAP = {
    "member_lookup": "Member Services",
    "check_eligibility": "Member Services",
    "coverage_lookup": "Member Services",
    "update_member_info": "Member Services",
    "member_policy_lookup": "Member Services",
    "claim_lookup": "Claims Services",
    "claim_status": "Claims Services",
    "claim_payment_info": "Claims Services",
    "member_claims": "Claims Services",
    "update_claim_status": "Claims Services",
    "pa_lookup": "PA Services",
    "pa_status": "PA Services",
    "pa_requirements": "PA Services",
    "member_prior_authorizations": "PA Services",
    "approve_prior_auth": "PA Services",
    "deny_prior_auth": "PA Services",
    "provider_lookup": "Provider Services",
    "provider_network_check": "Provider Services",
    "provider_search_by_specialty": "Provider Services",
    "search_policy_info": "Search Services",
    "search_medical_codes": "Search Services",
    "search_knowledge_base": "Search Services",
}
TOOL_DESCRIPTION_MAP = {
    "member_lookup": "Look up member demographics by member ID",
    "check_eligibility": "Check member eligibility and active coverage status",
    "coverage_lookup": "Retrieve detailed coverage info (deductibles, copays, benefits)",
    "update_member_info": "Update member contact or demographic information",
    "member_policy_lookup": "Look up member with their associated insurance policy",
    "claim_lookup": "Look up full claim details by claim ID",
    "claim_status": "Check claim processing status by claim number",
    "claim_payment_info": "Retrieve payment amounts and dates for a claim",
    "member_claims": "List all claims filed by a specific member",
    "update_claim_status": "Update claim status (requires HITL approval)",
    "pa_lookup": "Look up prior authorization details by PA ID",
    "pa_status": "Check PA processing status by PA ID",
    "pa_requirements": "Determine if a procedure requires PA under a policy type",
    "member_prior_authorizations": "List all PAs for a specific member",
    "approve_prior_auth": "Approve a prior authorization (requires HITL approval)",
    "deny_prior_auth": "Deny a prior authorization (requires HITL approval)",
    "provider_lookup": "Look up provider details by provider ID",
    "provider_network_check": "Check if a provider is in-network for a plan",
    "provider_search_by_specialty": "Search providers by specialty and location",
    "search_policy_info": "Semantic search over policy documents",
    "search_medical_codes": "Look up CPT/ICD codes and descriptions",
    "search_knowledge_base": "Search clinical guidelines and knowledge base",
}

settings: Settings = get_settings()

# Redis DB used by MetricsPersister for the central /metrics aggregation point
_METRICS_REDIS_DB = int(os.getenv("REDIS_METRICS_DB", "3"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: launch the metrics background push loop so this process's
    Prometheus counters are pushed to Redis DB 3 every 30 seconds.
    All six CSIP services (this API + 5 A2A servers) push to the same
    Redis DB; the /metrics endpoint below reads from it and serves the
    aggregated result to Prometheus as a single scrape target.
    """
    start_background_pusher()

    # ── Eager singleton initialisation ────────────────────────────────
    # Warm up security components at startup so the first request pays
    # no initialisation cost. Order matches the request processing pipeline:
    # NeMo (Control 1) → output validator / hub validators (Control 7)
    # → DLP scanner (Control 8). Presidio and the approval workflow are
    # initialised lazily on first use inside their own singletons.
    logger.info("Warming up NeMo Guardrails...")
    get_nemo_filter()
    logger.info("Warming up Presidio memory security...")
    get_presidio_security()
    logger.info("Warming up Guardrails output validator (hub + Presidio)...")
    get_output_validator()
    logger.info("Warming up DLP scanner...")
    get_dlp_scanner()

    logger.info("Agentic Access API lifespan startup complete.")
    yield
    # Daemon thread exits automatically with the process — no teardown needed.


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hierarchical Agentic AI for Health Insurance Customer Service Intelligence Platform",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Security
security = HTTPBearer()

# ============================================
# Request/Response Models
# ============================================

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str
    member_id: Optional[str] = None
    prior_session_id: Optional[str] = None   # For cross-session conversation continuity


class FeedbackRequest(BaseModel):
    session_id: str
    rating: str                              # 'correct', 'incorrect', or 'partial'
    correction: Optional[str] = None         # Free-text correction from CSR
    trace_id: Optional[str] = None           # LangFuse trace ID if available


class FeedbackResponse(BaseModel):
    feedback_id: Optional[str]
    status: str


class QueryResponse(BaseModel):
    session_id: str
    response: str
    execution_path: List[str]
    # Keyed by worker name, e.g. {"pa_lookup": {...}, "member_lookup": {...}}
    tool_results: Dict[str, Any]
    error_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]

# ============================================
# Authentication Dependency
# ============================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return payload
    
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# ============================================
# API Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "services": {
            "neo4j_kg": "connected",
            "neo4j_cg": "connected",
            "mysql": "connected",
            "chroma": "connected"
        }
    }


@app.get("/api/system/status")
async def system_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Lightweight system status for any authenticated user.
    Returns circuit breaker state so the top-bar badge reflects
    the real state regardless of role.
    """
    try:
        from security.approval_workflow import get_approval_workflow
        wf = get_approval_workflow()
        halted = wf.is_circuit_breaker_active()
        return {"circuit_breaker_active": halted}
    except Exception:
        return {"circuit_breaker_active": False}


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """
    Aggregate Prometheus metrics endpoint — single scrape target for the
    entire CSIP platform.

    Every CSIP service (this API + all 5 A2A servers) runs a
    MetricsPersister background thread that pushes its local Prometheus
    registry to Redis DB 3 every 30 seconds.  This endpoint reads all
    metrics:* keys from that shared Redis DB and returns them in Prometheus
    text exposition format so a single Prometheus scrape job pointed at
    this endpoint captures metrics from all six processes.

    Prometheus scrape config:
        scrape_configs:
          - job_name: 'csip'
            static_configs:
              - targets: ['agentic-access-api:8000']
            metrics_path: '/metrics'
            scrape_interval: 30s
    """
    try:
        r = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=_METRICS_REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        keys = r.keys("metrics:*")

        lines = []
        for key in sorted(keys):
            value = r.get(key)
            if value is None:
                continue

            # Key format: metrics:{metric_name}
            #         or: metrics:{metric_name}:{label_k=v,label_k=v}
            # Strip the leading "metrics:" prefix.
            remainder = key[len("metrics:"):]
            colon_pos = remainder.find(":")

            if colon_pos == -1:
                # No labels — skip internal aggregation shortcut keys that
                # don't follow the label format (e.g. metrics:requests_blocked)
                lines.append(f"{remainder} {value}")
            else:
                metric_name = remainder[:colon_pos]
                label_body  = remainder[colon_pos + 1:]
                # Reconstruct Prometheus label syntax: {k="v",k="v"}
                label_pairs = label_body.split(",")
                label_str = ",".join(
                    f'{p.split("=")[0]}="{p.split("=", 1)[1]}"'
                    for p in label_pairs
                    if "=" in p
                )
                lines.append(f'{metric_name}{{{label_str}}} {value}')

        output = "\n".join(lines) + "\n"
        return Response(
            content=output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    except Exception as exc:
        logger.error("Failed to serve /metrics: %s", exc)
        return Response(
            content="# CSIP metrics temporarily unavailable\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=500,
        )

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """User login"""
    try:
        # Authenticate user
        user = auth_service.authenticate_user(request.username, request.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user["user_id"], "role": user["role"]}
        )
        
        # Log successful login
        audit_logger.log_action(
            user_id=user["user_id"],
            action="LOGIN",
            resource_type="AUTH",
            resource_id=user["user_id"],
            status="SUCCESS"
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user
        }
    
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/api/agent/query", response_model=QueryResponse)
async def agent_query(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Submit query to agent with integrated security controls"""
    session_id = str(uuid.uuid4())
    user_id: str = str(current_user["sub"])

    # Reject the request immediately if the JWT has no role claim.
    # str(None) would silently produce the string "none", which would
    # pass through to RBAC and fail with a misleading unauthorised error.
    raw_role = current_user.get("role")
    if not raw_role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing the required role claim.",
        )
    user_role: str = str(raw_role)

    try:
        # Process request through security-integrated entry point.
        # process_user_request returns a ProcessResult carrying the validated
        # response string plus execution metadata from the graph.
        result = process_user_request(
            user_input=request.query,
            user_id=user_id,
            user_role=user_role,
            session_id=session_id,
            member_id=request.member_id,
            prior_session_id=request.prior_session_id,
        )

        logger.info("Query processed successfully for session %s", session_id)

        return {
            "session_id":     session_id,
            "response":       result.response,
            "execution_path": result.execution_path,
            "tool_results":   result.tool_results,
            "error_count":    result.error_count,
        }
    
    except Exception as e:
        logger.error(f"Agent query failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )



@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Accept quality feedback from a CSR on a completed session.

    Stores the rating and optional correction in MySQL and links it to
    the LangFuse trace for that session so evaluation metrics can be
    computed.
    """
    user_id: str = str(current_user.get("sub", ""))
    valid_ratings = {"correct", "incorrect", "partial"}
    if request.rating not in valid_ratings:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"rating must be one of {sorted(valid_ratings)}",
        )
    try:
        from databases.feedback_data_access import get_feedback_data_access
        feedback_da = get_feedback_data_access()
        feedback_id = feedback_da.store_feedback(
            session_id=request.session_id,
            rating=request.rating,
            user_id=user_id,
            trace_id=request.trace_id or "",
            correction=request.correction,
        )
        # Also score the LangFuse trace if a trace_id was provided
        if request.trace_id:
            try:
                from observability.langfuse_integration import get_langfuse_tracer
                tracer = get_langfuse_tracer()
                if tracer.enabled and tracer.langfuse:
                    score_map = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
                    tracer.langfuse.score(
                        trace_id=request.trace_id,
                        name="csip_csr_rating",
                        value=score_map.get(request.rating, 0.5),
                        comment=request.correction,
                    )
            except Exception as lf_exc:
                logger.warning("feedback: LangFuse scoring failed (non-fatal): %s", lf_exc)

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="stored" if feedback_id else "failed",
        )
    except Exception as exc:
        logger.error("submit_feedback failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store feedback",
        )


# ---------------------------------------------------------------------------
# Feedback Learning Endpoints
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    session_id: str
    classification_type: str   # planning, routing, tool, synthesis, security, data_quality, retrieval
    notes: Optional[str] = None


class ClassifyResponse(BaseModel):
    classification_id: Optional[str]
    status: str


@app.post("/api/feedback/classify", response_model=ClassifyResponse)
async def classify_feedback(
    request: ClassifyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Classify the root cause of a low-rated session.
    Requires CSR_SUPERVISOR role.
    """
    user_id: str = str(current_user.get("sub", ""))
    user_role: str = str(current_user.get("role", ""))

    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Classification requires CSR_SUPERVISOR role",
        )

    valid_types = {
        "planning", "routing", "tool", "synthesis",
        "security", "data_quality", "retrieval",
    }
    if request.classification_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"classification_type must be one of {sorted(valid_types)}",
        )

    try:
        from databases.feedback_data_access import get_feedback_data_access
        feedback_da = get_feedback_data_access()
        cid = feedback_da.store_classification(
            session_id=request.session_id,
            classified_by=user_id,
            classification_type=request.classification_type,
            notes=request.notes or "",
        )

        # Increment Prometheus counter so the Feedback Dashboard funnel is accurate
        try:
            from observability.prometheus_metrics import feedback_classifications_total
            feedback_classifications_total.labels(
                classification_type=request.classification_type
            ).inc()
        except Exception:
            pass  # Non-fatal — metric may not be available in all environments

        return ClassifyResponse(
            classification_id=cid,
            status="stored" if cid else "failed",
        )
    except Exception as exc:
        logger.error("classify_feedback failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store classification",
        )


@app.get("/api/feedback/patterns")
async def get_feedback_patterns(
    limit: int = 20,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return the latest feedback pattern analysis reports.
    Requires CSR_SUPERVISOR role.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Pattern reports require CSR_SUPERVISOR role",
        )
    try:
        from databases.feedback_data_access import get_feedback_data_access
        return get_feedback_data_access().get_latest_pattern_reports(limit=limit)
    except Exception as exc:
        logger.error("get_feedback_patterns failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pattern reports",
        )


@app.get("/api/feedback/improvements")
async def get_improvements(
    limit: int = 20,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return prompt change log entries with before/after metrics.
    Requires CSR_SUPERVISOR role.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Improvement log requires CSR_SUPERVISOR role",
        )
    try:
        from databases.feedback_data_access import get_feedback_data_access
        return get_feedback_data_access().get_prompt_changes(limit=limit)
    except Exception as exc:
        logger.error("get_improvements failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve improvement log",
        )


@app.delete("/api/experience/{session_id}")
async def delete_experience(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Remove an experience from the Chroma store.
    Used by supervisors to remove experiences later found to be incorrect.
    Requires CSR_SUPERVISOR role.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Experience deletion requires CSR_SUPERVISOR role",
        )
    try:
        from databases.chroma_experience_store import get_experience_store
        success = get_experience_store().remove_experience(session_id)
        return {"session_id": session_id, "deleted": success}
    except Exception as exc:
        logger.error("delete_experience failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete experience",
        )


# ============================================
# Context Graph Explorer Endpoints
# ============================================

@app.get("/api/cg/session/{session_id}")
async def get_cg_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Retrieve the Context Graph tree for a session, including the
    full follow-up chain and execution tree.

    Returns:
        - session: Session node properties
        - follow_up_chain: Ordered list of sessions in the conversation chain
        - tree: Hierarchical execution tree (Session → Plan → Goal → Step → AgentExecution → ToolExecution)
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()

        # 1. Get session info
        session = cg.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )

        # 2. Get follow-up chain — find root then walk forward
        # get_session() doesn't return rootSessionId/chainDepth, so query directly
        root_query = """
        MATCH (s:Session {sessionId: $sessionId})
        RETURN coalesce(s.rootSessionId, s.sessionId) AS rootId
        """
        root_result = cg.execute_custom_query(root_query, {"sessionId": session_id})
        root_id = root_result[0]["rootId"] if root_result else session_id

        chain_query = """
        MATCH path = (root:Session {sessionId: $rootId})-[:HAS_FOLLOW_UP*0..20]->(s:Session)
        WITH s, length(path) AS depth
        OPTIONAL MATCH (s)-[:HAS_EXECUTION]->(:AgentExecution)-[:HAS_PLAN|CALLED_AGENT*0..2]->
                       (:AgentExecution {agentType: 'supervisor'})-[:HAS_PLAN]->(p:Plan)
        RETURN s {
            .sessionId, .userId, .startTime, .status,
            .chainDepth, .rootSessionId, .metadata,
            .conversationMessages
        } AS session,
        depth,
        p.status AS planStatus
        ORDER BY depth ASC
        """
        chain_result = cg.execute_custom_query(chain_query, {"rootId": root_id})
        follow_up_chain = []
        for row in chain_result:
            sess_data = row.get("session", {})
            sess_data["depth"] = row.get("depth", 0)
            sess_data["planStatus"] = row.get("planStatus")
            # Extract query text from conversationMessages (first human message)
            query_text = ""
            conv_raw = sess_data.pop("conversationMessages", None)
            if conv_raw:
                try:
                    import json as _json
                    conv = _json.loads(conv_raw) if isinstance(conv_raw, str) else conv_raw
                    for _m in conv:
                        if _m.get("type") == "human":
                            query_text = (_m.get("data") or {}).get("content", "")[:200]
                            break
                except Exception:
                    pass
            if not query_text:
                meta = sess_data.get("metadata")
                if isinstance(meta, str):
                    try:
                        meta = _json.loads(meta)
                    except Exception:
                        meta = {}
                query_text = (meta or {}).get("query", "")
            sess_data["query"] = query_text
            follow_up_chain.append(sess_data)

        # 3. Build execution tree for the requested session
        #
        # Graph structure (from actual CG data):
        #   Session → HAS_EXECUTION → AgentExecution(central_supervisor_planner)
        #     → HAS_PLAN → Plan(central) → HAS_GOAL → Goal → HAS_STEP → Step
        #       → EXECUTED_BY → AgentExecution(team_routing, e.g. claims_services_team)
        #         → CALLED_AGENT → AgentExecution(a2a_client)
        #           → CALLED_AGENT → AgentExecution(a2a_server)
        #             → HAS_PLAN → Plan(team) → Goal → Step
        #               → EXECUTED_BY → AgentExecution(worker)
        #                 → CALLED_TOOL → ToolExecution
        #                   → HAD_ERROR → ToolError (optional)
        #                 → HAD_ERROR → AgentError (optional)
        #
        # IMPORTANT: A2A chain starts from cStepExec (the step execution),
        # NOT from topExec (the central planner).
        tree_query = """
        MATCH (s:Session {sessionId: $sessionId})
        OPTIONAL MATCH (s)-[:HAS_EXECUTION]->(topExec:AgentExecution)
        OPTIONAL MATCH (topExec)-[:HAS_PLAN]->(centralPlan:Plan)
        OPTIONAL MATCH (centralPlan)-[:HAS_GOAL]->(cGoal:Goal)
        OPTIONAL MATCH (cGoal)-[:HAS_STEP]->(cStep:Step)
        OPTIONAL MATCH (cStep)-[:EXECUTED_BY]->(cStepExec:AgentExecution)

        WITH s, topExec, centralPlan, cGoal, cStep, cStepExec

        OPTIONAL MATCH (cStepExec)-[:CALLED_AGENT]->(a2aClient:AgentExecution {agentType: 'a2a_client'})
        OPTIONAL MATCH (a2aClient)-[:CALLED_AGENT]->(a2aServer:AgentExecution {agentType: 'a2a_server'})
        OPTIONAL MATCH (a2aServer)-[:HAS_PLAN]->(teamPlan:Plan)
        OPTIONAL MATCH (teamPlan)-[:HAS_GOAL]->(tGoal:Goal)
        OPTIONAL MATCH (tGoal)-[:HAS_STEP]->(tStep:Step)
        OPTIONAL MATCH (tStep)-[:EXECUTED_BY]->(tStepExec:AgentExecution)
        OPTIONAL MATCH (tStepExec)-[:CALLED_TOOL]->(tTool:ToolExecution)
        OPTIONAL MATCH (tTool)-[:HAD_ERROR]->(tToolErr:ToolError)
        OPTIONAL MATCH (tStepExec)-[:HAD_ERROR]->(tAgentErr:AgentError)

        RETURN
            s { .sessionId, .userId, .startTime, .endTime, .status } AS session,
            collect(DISTINCT topExec {
                .executionId, .agentName, .agentType, .status, .startTime, .endTime, .duration
            }) AS topExecutions,
            collect(DISTINCT centralPlan {
                .planId, .planType, .teamName, .status, .totalGoals
            }) AS centralPlans,
            collect(DISTINCT cGoal {
                .goalId, .description, .priority, .status
            }) AS centralGoals,
            collect(DISTINCT cStep {
                .stepId, .action, .worker, .status, goalId: cGoal.goalId
            }) AS centralSteps,
            collect(DISTINCT cStepExec {
                .executionId, .agentName, .agentType, .status, .duration,
                stepId: cStep.stepId, teamName: cStepExec.agentName
            }) AS centralStepExecs,
            collect(DISTINCT a2aClient {
                .executionId, .agentName, .agentType, .status, .duration,
                parentStepExecId: cStepExec.executionId
            }) AS a2aClients,
            collect(DISTINCT a2aServer {
                .executionId, .agentName, .agentType, .status, .duration,
                clientExecId: a2aClient.executionId
            }) AS a2aServers,
            collect(DISTINCT teamPlan {
                .planId, .planType, .teamName, .status, .totalGoals,
                serverExecId: a2aServer.executionId
            }) AS teamPlans,
            collect(DISTINCT tGoal {
                .goalId, .description, .priority, .status, planId: teamPlan.planId
            }) AS teamGoals,
            collect(DISTINCT tStep {
                .stepId, .action, .worker, .status, goalId: tGoal.goalId
            }) AS teamSteps,
            collect(DISTINCT tStepExec {
                .executionId, .agentName, .agentType, .status, .duration, stepId: tStep.stepId
            }) AS teamStepExecs,
            collect(DISTINCT tTool {
                .toolName, .status, .executionTime, .inputData, .outputSummary,
                .toolExecutionId, execId: tStepExec.executionId
            }) AS teamTools,
            collect(DISTINCT tToolErr {
                .errorId, .errorType, .message, .timestamp,
                toolExecId: tTool.toolExecutionId
            }) AS teamToolErrors,
            collect(DISTINCT tAgentErr {
                .errorId, .errorType, .message, .timestamp,
                agentExecId: tStepExec.executionId
            }) AS teamAgentErrors
        """
        tree_result = cg.execute_custom_query(tree_query, {"sessionId": session_id})

        # 4. Assemble hierarchical tree from flat query results
        tree = {"type": "Session", "id": session_id, "status": session.get("status", "unknown"), "props": session, "children": []}
        if tree_result:
            row = tree_result[0]

            # Helper: filter out None/empty dicts from collect()
            def clean(lst):
                return [x for x in (lst or []) if x and any(v is not None for v in x.values())]

            top_execs = clean(row.get("topExecutions"))
            central_plans = clean(row.get("centralPlans"))
            central_goals = clean(row.get("centralGoals"))
            central_steps = clean(row.get("centralSteps"))
            central_step_execs = clean(row.get("centralStepExecs"))
            a2a_clients = clean(row.get("a2aClients"))
            a2a_servers = clean(row.get("a2aServers"))
            team_plans = clean(row.get("teamPlans"))
            team_goals = clean(row.get("teamGoals"))
            team_steps = clean(row.get("teamSteps"))
            team_step_execs = clean(row.get("teamStepExecs"))
            team_tools = clean(row.get("teamTools"))
            team_tool_errors = clean(row.get("teamToolErrors"))
            team_agent_errors = clean(row.get("teamAgentErrors"))

            # Build central plan subtree with full A2A → team → worker → tool chain
            for plan in central_plans:
                plan_node = {"type": "Plan", "id": plan.get("planId", ""), "status": plan.get("status", ""), "props": plan, "children": []}
                for goal in [g for g in central_goals if g.get("goalId")]:
                    goal_node = {"type": "Goal", "id": goal.get("goalId", ""), "status": goal.get("status", ""), "props": goal, "children": []}
                    for step in [s for s in central_steps if s.get("goalId") == goal.get("goalId")]:
                        step_node = {"type": "Step", "id": step.get("stepId", ""), "status": step.get("status", ""), "props": step, "children": []}

                        # Central step exec (e.g. claims_services_team routing)
                        for ex in [e for e in central_step_execs if e.get("stepId") == step.get("stepId")]:
                            exec_node = {"type": "AgentExecution", "id": ex.get("executionId", ""), "status": ex.get("status", ""), "props": ex, "children": []}

                            # A2A chain: step exec → a2a_client → a2a_server → team plan
                            for server in a2a_servers:
                                server_node = {"type": "AgentExecution", "id": server.get("executionId", ""), "status": server.get("status", ""), "props": {**server, "agentType": "a2a_server"}, "children": []}

                                for tplan in [p for p in team_plans if p.get("serverExecId") == server.get("executionId")]:
                                    tplan_node = {"type": "Plan", "id": tplan.get("planId", ""), "status": tplan.get("status", ""), "props": tplan, "children": []}
                                    for tgoal in [g for g in team_goals if g.get("planId") == tplan.get("planId")]:
                                        tgoal_node = {"type": "Goal", "id": tgoal.get("goalId", ""), "status": tgoal.get("status", ""), "props": tgoal, "children": []}
                                        for tstep in [s for s in team_steps if s.get("goalId") == tgoal.get("goalId")]:
                                            tstep_node = {"type": "Step", "id": tstep.get("stepId", ""), "status": tstep.get("status", ""), "props": tstep, "children": []}
                                            for tex in [e for e in team_step_execs if e.get("stepId") == tstep.get("stepId")]:
                                                texec_node = {"type": "AgentExecution", "id": tex.get("executionId", ""), "status": tex.get("status", ""), "props": tex, "children": []}

                                                # Agent errors on the worker execution
                                                for aerr in [e for e in team_agent_errors if e.get("agentExecId") == tex.get("executionId")]:
                                                    texec_node["children"].append({"type": "AgentError", "id": aerr.get("errorId", ""), "status": "error", "props": aerr, "children": []})

                                                # Tool executions under this worker
                                                for ttool in [t for t in team_tools if t.get("execId") == tex.get("executionId")]:
                                                    tool_node = {"type": "ToolExecution", "id": ttool.get("toolName", ""), "status": ttool.get("status", ""), "props": ttool, "children": []}

                                                    # Tool errors
                                                    for terr in [e for e in team_tool_errors if e.get("toolExecId") == ttool.get("toolExecutionId")]:
                                                        tool_node["children"].append({"type": "ToolError", "id": terr.get("errorId", ""), "status": "error", "props": terr, "children": []})

                                                    texec_node["children"].append(tool_node)

                                                tstep_node["children"].append(texec_node)
                                            tgoal_node["children"].append(tstep_node)
                                        tplan_node["children"].append(tgoal_node)
                                    server_node["children"].append(tplan_node)

                                if server_node["children"]:
                                    exec_node["children"].append(server_node)

                            step_node["children"].append(exec_node)
                        goal_node["children"].append(step_node)
                    plan_node["children"].append(goal_node)
                tree["children"].append(plan_node)

            # If no plans found, attach raw executions
            if not tree["children"] and top_execs:
                for ex in top_execs:
                    tree["children"].append({"type": "AgentExecution", "id": ex.get("executionId", ""), "status": ex.get("status", ""), "props": ex, "children": []})

        return {
            "session": session,
            "follow_up_chain": follow_up_chain,
            "tree": tree,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_cg_session failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve Context Graph data",
        )


@app.get("/api/stats/agentic-health")
async def get_agentic_health(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return all Tier 1 and Tier 2 agentic performance metrics.

    Reads Prometheus gauges set by the evaluation pipeline (every 30s).
    Falls back to Redis cache if Prometheus gauges are unavailable.
    Available to CSR_TIER2 and above.
    """
    user_role = str(current_user.get("role", ""))
    if user_role not in ("CSR_TIER2", "CSR_SUPERVISOR"):
        raise HTTPException(status_code=403, detail="Requires CSR_TIER2 or CSR_SUPERVISOR")

    metrics = {}
    try:
        # Try Prometheus gauges first (set by evaluation_pipeline every 30s)
        try:
            from observability.evaluation_pipeline import (
                planning_routing_accuracy, positive_feedback_rate,
                avg_plan_goals, avg_plan_steps,
                estimated_tokens_per_query, llm_calls_per_query,
                tool_success_rate, agent_error_rate,
                avg_agent_latency, avg_e2e_latency,
            )
            metrics = {
                # Tier 1: LLM & Cost
                "estimated_tokens_per_query": round(estimated_tokens_per_query._value.get(), 1),
                "llm_calls_per_query":        round(llm_calls_per_query._value.get(), 2),
                "tool_success_rate":          round(tool_success_rate._value.get(), 4),
                # Tier 1: Quality
                "planning_routing_accuracy":  round(planning_routing_accuracy._value.get(), 4),
                "positive_feedback_rate":     round(positive_feedback_rate._value.get(), 4),
                # Tier 2: Agent Health
                "agent_error_rate":           round(agent_error_rate._value.get(), 4),
                "avg_agent_latency_seconds":  round(avg_agent_latency._value.get(), 3),
                "avg_e2e_latency_seconds":    round(avg_e2e_latency._value.get(), 2),
                # Tier 2: Drift
                "avg_plan_goals":             round(avg_plan_goals._value.get(), 2),
                "avg_plan_steps":             round(avg_plan_steps._value.get(), 2),
            }
        except Exception:
            pass

        # Fallback: read from Redis if gauges are 0
        if all(v == 0 for v in metrics.values()) or not metrics:
            try:
                import redis as _redis
                r = _redis.Redis(
                    host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                    decode_responses=True, socket_connect_timeout=2,
                )
                keys_map = {
                    "estimated_tokens_per_query":  "metrics:estimated_tokens_per_query",
                    "llm_calls_per_query":         "metrics:llm_calls_per_query",
                    "tool_success_rate":           "metrics:tool_success_rate",
                    "planning_routing_accuracy":   "metrics:planning_routing_accuracy",
                    "positive_feedback_rate":      "metrics:positive_feedback_rate",
                    "agent_error_rate":            "metrics:agent_error_rate",
                    "avg_agent_latency_seconds":   "metrics:avg_agent_latency_seconds",
                    "avg_e2e_latency_seconds":     "metrics:avg_e2e_latency_seconds",
                    "avg_plan_goals":              "metrics:avg_plan_goals",
                    "avg_plan_steps":              "metrics:avg_plan_steps",
                }
                for k, redis_key in keys_map.items():
                    val = r.get(redis_key)
                    if val is not None:
                        metrics[k] = float(val)
            except Exception:
                pass

        return {"metrics": metrics}
    except Exception as exc:
        logger.error("get_agentic_health failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve agentic health metrics")


@app.get("/api/stats/team-invocations")
async def get_team_invocations(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return agent invocation counts by team from the Context Graph.

    The Prometheus ``agent_invocations_total`` counter is defined but
    never incremented — agent execution tracking goes through the CG.
    This endpoint queries AgentExecution nodes grouped by agentName
    for supervisor-level (team routing) executions.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (ae:AgentExecution)
            WHERE ae.agentType = 'supervisor'
              AND ae.agentName ENDS WITH '_team'
            RETURN ae.agentName AS team, count(ae) AS invocations
            ORDER BY invocations DESC
            """
        )
        teams = [
            {
                "name": r["team"].replace("_services_team", "").replace("_team", "").replace("_", " ").title(),
                "count": int(r["invocations"]),
            }
            for r in result
        ]
        return {"teams": teams}
    except Exception as exc:
        logger.error("get_team_invocations failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve team invocation data",
        )


@app.get("/api/stats/rate-limit-utilization")
async def get_rate_limit_utilization(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return per-user rate limit utilization from MySQL.

    The Prometheus ``rate_limit_checks_total`` counter increments in A2A
    containers which aren't scraped reliably. The actual rate limit data
    lives in MySQL's ``rate_limits`` table (sliding 1-minute windows).

    Requires CSR_SUPERVISOR role.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires CSR_SUPERVISOR")
    try:
        from databases.connections import get_mysql
        db = get_mysql()

        # Aggregate current-window usage per user (last 1 minute)
        rows = db.execute_query(
            """
            SELECT user_id,
                   SUM(request_count) AS total_requests,
                   MAX(limit_per_window) AS configured_limit
            FROM rate_limits
            WHERE window_end > DATE_SUB(NOW(), INTERVAL 1 MINUTE)
            GROUP BY user_id
            ORDER BY total_requests DESC
            LIMIT 20
            """
        )
        users = []
        for r in rows:
            uid = r.get("user_id", "unknown")
            current = int(r.get("total_requests") or 0)
            limit = int(r.get("configured_limit") or 120)
            pct = round(min(current / max(limit, 1) * 100, 100), 1)
            users.append({"user": uid, "current": current, "limit": limit, "pct": pct})

        return {"users": users}
    except Exception as exc:
        logger.error("get_rate_limit_utilization failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rate limit utilization",
        )


@app.get("/api/stats/tool-usage")
async def get_tool_usage(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return per-tool invocation counts from the Context Graph, enriched
    with service team metadata from A2A agent cards.

    Queries ToolExecution nodes grouped by toolName. Each result includes
    the tool name, invocation count, and the service team it belongs to
    (derived from the tool catalog).

    Available to any authenticated role.
    """
    try:
        from databases.context_graph_data_access import get_cg_data_access
        cg = get_cg_data_access()
        result = cg.conn.execute_query(
            """
            MATCH (te:ToolExecution)
            WHERE te.toolName IS NOT NULL
            RETURN te.toolName AS tool, count(te) AS invocations
            ORDER BY invocations DESC
            """
        )

        # ── Tool → Service mapping ─────────────────────────────────
        # Start with the module-level authoritative map; optionally
        # overlay live A2A agent card data if available.
        tool_svc = dict(TOOL_SERVICE_MAP)

        # Optional: override with live A2A agent card data if available
        try:
            from agents.core.a2a_agent_card import get_agent_card_registry
            registry = get_agent_card_registry()
            cards = registry.get_all_cards()
            if not cards:
                from agents.teams.claims_services.claims_services_a2a_agent_card import build_claims_services_agent_card
                from agents.teams.member_services.member_services_a2a_agent_card import build_member_services_agent_card
                from agents.teams.pa_services.pa_services_a2a_agent_card import build_pa_services_agent_card
                from agents.teams.provider_services.provider_services_a2a_agent_card import build_provider_services_agent_card
                from agents.teams.search_services.search_services_a2a_agent_card import build_search_services_agent_card
                for name, builder in [
                    ("claims_services_team", build_claims_services_agent_card),
                    ("member_services_team", build_member_services_agent_card),
                    ("pa_services_team", build_pa_services_agent_card),
                    ("provider_services_team", build_provider_services_agent_card),
                    ("search_services_team", build_search_services_agent_card),
                ]:
                    try:
                        card = builder()
                        registry.register_card(name, card)
                    except Exception:
                        pass
                cards = registry.get_all_cards()

            service_labels = {
                "claims_services_team": "Claims Services",
                "member_services_team": "Member Services",
                "pa_services_team": "PA Services",
                "provider_services_team": "Provider Services",
                "search_services_team": "Search Services",
            }
            for agent_name, card in cards.items():
                service = service_labels.get(agent_name, agent_name)
                for skill in card.skills:
                    tool_svc[skill.id] = service
        except Exception as cat_exc:
            logger.warning("A2A card enrichment unavailable, using built-in map: %s", cat_exc)

        tools = [
            {
                "tool": r["tool"],
                "count": int(r["invocations"]),
                "service": tool_svc.get(r["tool"], "Other"),
            }
            for r in result
        ]
        return {"tools": tools}
    except Exception as exc:
        logger.error("get_tool_usage failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tool usage data",
        )


# ============================================
# Admin Endpoints — Tool Permissions & Rate Limits
# ============================================

class CircuitBreakerRequest(BaseModel):
    action: str      # "activate" or "deactivate"
    reason: str = ""


@app.get("/api/admin/circuit-breaker")
async def get_circuit_breaker_status(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return current circuit breaker state from Redis."""
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires CSR_SUPERVISOR")
    try:
        from security.approval_workflow import get_approval_workflow
        wf = get_approval_workflow()
        active = wf.is_circuit_breaker_active()
        reason = wf.redis_client.get("circuit_breaker:reason") or ""
        activated_by = wf.redis_client.get("circuit_breaker:activated_by") or ""
        timestamp = wf.redis_client.get("circuit_breaker:timestamp") or ""
        return {
            "active": active,
            "reason": reason if isinstance(reason, str) else reason.decode(),
            "activated_by": activated_by if isinstance(activated_by, str) else activated_by.decode(),
            "timestamp": timestamp if isinstance(timestamp, str) else timestamp.decode(),
        }
    except Exception as exc:
        logger.error("get_circuit_breaker_status failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read circuit breaker state")


@app.post("/api/admin/circuit-breaker")
async def toggle_circuit_breaker(
    request: CircuitBreakerRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Activate or deactivate the circuit breaker. Requires CSR_SUPERVISOR.

    Activate: sets Redis flag → all agent queries immediately rejected at
    request_processor.py outermost boundary. Logs ACTIVATED event to MySQL.

    Deactivate: clears Redis flag → agent processing resumes. Logs
    DEACTIVATED event to MySQL.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires CSR_SUPERVISOR")
    if request.action not in ("activate", "deactivate"):
        raise HTTPException(status_code=400, detail="action must be 'activate' or 'deactivate'")
    try:
        from security.approval_workflow import get_approval_workflow
        wf = get_approval_workflow()
        reviewer = current_user.get("sub", "unknown")

        if request.action == "activate":
            wf.activate_circuit_breaker(
                reason=request.reason or "Activated via supervisor control pane",
                activated_by=reviewer,
            )
        else:
            wf.deactivate_circuit_breaker(
                deactivated_by=reviewer,
                rationale=request.reason or "Deactivated via supervisor control pane",
            )

        logger.info("Circuit breaker %sd by %s: %s", request.action, reviewer, request.reason)
        return {"status": request.action + "d", "by": reviewer}
    except Exception as exc:
        logger.error("toggle_circuit_breaker failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to toggle circuit breaker")


@app.get("/api/admin/approval-queue")
async def get_approval_queue(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return pending approval requests and summary counts from MySQL.
    Requires CSR_SUPERVISOR role.

    The Prometheus counters approval_requests_total / approval_responses_total
    are unreliable because approvals execute in A2A server containers which
    Prometheus does not scrape. MySQL is the source of truth.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Approval queue requires CSR_SUPERVISOR role",
        )
    try:
        from security.approval_workflow import get_approval_workflow
        from databases.connections import get_mysql
        wf = get_approval_workflow()
        db = get_mysql()

        # Get pending requests
        pending = wf.get_pending_approvals(limit=50)

        # Get summary counts across all statuses
        counts_result = db.execute_query(
            """
            SELECT status, COUNT(*) as cnt
            FROM approval_requests
            GROUP BY status
            """
        )
        counts = {row["status"]: row["cnt"] for row in counts_result}

        return {
            "pending": [
                {
                    "request_id":   r.get("request_id"),
                    "tool_name":    r.get("tool_name"),
                    "impact_level": r.get("impact_level"),
                    "requested_by": r.get("requested_by"),
                    "requested_at": str(r.get("requested_at", "")),
                    "expires_at":   str(r.get("expires_at", "")),
                    "parameters":   r.get("parameters"),
                    "status":       r.get("status"),
                }
                for r in pending
            ],
            "counts": {
                "total":    sum(counts.values()),
                "pending":  counts.get("PENDING", 0),
                "approved": counts.get("APPROVED", 0),
                "denied":   counts.get("DENIED", 0),
                "expired":  counts.get("EXPIRED", 0),
            },
        }
    except Exception as exc:
        logger.error("get_approval_queue failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve approval queue",
        )


class ApprovalResolveRequest(BaseModel):
    request_id: str
    decision: str          # "approve" or "deny"
    rationale: str = ""


@app.post("/api/admin/approval-resolve")
async def resolve_approval(
    request: ApprovalResolveRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Mark a pending approval request as APPROVED or DENIED in MySQL.
    Requires CSR_SUPERVISOR role.

    For APPROVE: the caller (webapp) first sends the action through the
    full agent pipeline via /api/agent/query, then calls this endpoint
    to close the MySQL record so it no longer appears in the queue.

    For DENY: this is the only call needed — updates MySQL and publishes
    to Redis so the waiting A2A coroutine receives the denial.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Approval resolution requires CSR_SUPERVISOR role",
        )

    if request.decision not in ("approve", "deny"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="decision must be 'approve' or 'deny'",
        )

    try:
        from security.approval_workflow import get_approval_workflow
        wf = get_approval_workflow()
        reviewer_id = current_user.get("sub", "unknown")

        if request.decision == "approve":
            wf.approve_request(
                request.request_id, reviewer_id,
                request.rationale or "Approved via supervisor control pane",
            )
        else:
            wf.deny_request(
                request.request_id, reviewer_id,
                request.rationale or "Denied via supervisor control pane",
            )

        logger.info(
            "Approval %sd: request=%s by=%s rationale=%s",
            request.decision, request.request_id, reviewer_id, request.rationale,
        )
        return {
            "status": request.decision + "d",
            "request_id": request.request_id,
            "reviewed_by": reviewer_id,
        }
    except Exception as exc:
        logger.error("resolve_approval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve approval request",
        )


@app.get("/api/admin/tool-catalog")
async def get_tool_catalog(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Return tool metadata (service group, description, tags) from A2A agent cards.
    Requires CSR_SUPERVISOR role.

    Returns a dict mapping tool_name → {service, description, tags}.
    The source of truth is the AgentCardRegistry, populated from the
    build_*_agent_card() functions in each team's agent card module.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tool catalog requires CSR_SUPERVISOR role",
        )
    try:
        # Start with authoritative built-in catalog
        catalog = {}
        for tool_name, svc in TOOL_SERVICE_MAP.items():
            catalog[tool_name] = {
                "name": tool_name,
                "service": svc,
                "description": TOOL_DESCRIPTION_MAP.get(tool_name, ""),
                "tags": [],
            }

        # Overlay with live A2A agent card data if available
        try:
            from agents.core.a2a_agent_card import get_agent_card_registry

            registry = get_agent_card_registry()
            cards = registry.get_all_cards()

            # If registry is empty (no queries sent yet), populate it
            # by building cards directly from each team's module.
            if not cards:
                from agents.teams.claims_services.claims_services_a2a_agent_card import build_claims_services_agent_card
                from agents.teams.member_services.member_services_a2a_agent_card import build_member_services_agent_card
                from agents.teams.pa_services.pa_services_a2a_agent_card import build_pa_services_agent_card
                from agents.teams.provider_services.provider_services_a2a_agent_card import build_provider_services_agent_card
                from agents.teams.search_services.search_services_a2a_agent_card import build_search_services_agent_card

                for name, builder in [
                    ("claims_services_team", build_claims_services_agent_card),
                    ("member_services_team", build_member_services_agent_card),
                    ("pa_services_team", build_pa_services_agent_card),
                    ("provider_services_team", build_provider_services_agent_card),
                    ("search_services_team", build_search_services_agent_card),
                ]:
                    try:
                        card = builder()
                        registry.register_card(name, card)
                    except Exception:
                        pass
                cards = registry.get_all_cards()

            # Map agent_name → friendly service label
            service_labels = {
                "claims_services_team":   "Claims Services",
                "member_services_team":   "Member Services",
                "pa_services_team":       "PA Services",
                "provider_services_team": "Provider Services",
                "search_services_team":   "Search Services",
            }

            for agent_name, card in cards.items():
                service = service_labels.get(agent_name, agent_name)
                for skill in card.skills:
                    catalog[skill.id] = {
                        "name": skill.name,
                        "service": service,
                        "description": skill.description,
                        "tags": skill.tags,
                    }
        except Exception as cat_exc:
            logger.warning("A2A card enrichment unavailable for tool-catalog, using built-in map: %s", cat_exc)

        return {"catalog": catalog}

    except Exception as exc:
        logger.error("get_tool_catalog failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build tool catalog",
        )


@app.get("/api/admin/tool-permissions")
async def get_tool_permissions(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    List all tool permissions across all roles, enriched with service group
    and description from A2A agent cards.
    Requires CSR_SUPERVISOR role.

    Each permission row includes: tool_name, is_allowed, rate_limit_per_minute,
    service (e.g. "Member Services"), and description (from the agent card skill).
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tool permission management requires CSR_SUPERVISOR role",
        )
    try:
        # ── Build catalog: start with authoritative map, overlay A2A cards ──
        catalog = {}
        for tool_name, svc in TOOL_SERVICE_MAP.items():
            catalog[tool_name] = {
                "service": svc,
                "description": TOOL_DESCRIPTION_MAP.get(tool_name, ""),
            }
        try:
            from agents.core.a2a_agent_card import get_agent_card_registry
            registry = get_agent_card_registry()
            cards = registry.get_all_cards()
            if not cards:
                from agents.teams.claims_services.claims_services_a2a_agent_card import build_claims_services_agent_card
                from agents.teams.member_services.member_services_a2a_agent_card import build_member_services_agent_card
                from agents.teams.pa_services.pa_services_a2a_agent_card import build_pa_services_agent_card
                from agents.teams.provider_services.provider_services_a2a_agent_card import build_provider_services_agent_card
                from agents.teams.search_services.search_services_a2a_agent_card import build_search_services_agent_card
                for name, builder in [
                    ("claims_services_team", build_claims_services_agent_card),
                    ("member_services_team", build_member_services_agent_card),
                    ("pa_services_team", build_pa_services_agent_card),
                    ("provider_services_team", build_provider_services_agent_card),
                    ("search_services_team", build_search_services_agent_card),
                ]:
                    try:
                        card = builder()
                        registry.register_card(name, card)
                    except Exception:
                        pass
                cards = registry.get_all_cards()

            service_labels = {
                "claims_services_team": "Claims Services",
                "member_services_team": "Member Services",
                "pa_services_team": "PA Services",
                "provider_services_team": "Provider Services",
                "search_services_team": "Search Services",
            }
            for agent_name, card in cards.items():
                service = service_labels.get(agent_name, agent_name)
                for skill in card.skills:
                    catalog[skill.id] = {
                        "service": service,
                        "description": skill.description,
                    }
        except Exception as cat_exc:
            logger.warning("A2A card enrichment unavailable for permissions, using built-in map: %s", cat_exc)

        # ── Fetch permissions from MySQL ────────────────────────────────
        from databases.connections import get_mysql
        db = get_mysql()
        rows = db.execute_query(
            """
            SELECT tool_permission_id, role, tool_name, is_allowed, rate_limit_per_minute
            FROM tool_permissions
            ORDER BY role, tool_name
            """
        )
        # Group by role, enriched with catalog metadata
        by_role = {}
        for row in rows:
            r = row["role"]
            if r not in by_role:
                by_role[r] = []
            tool_name = row["tool_name"]
            cat = catalog.get(tool_name, {})
            by_role[r].append({
                "id": row["tool_permission_id"],
                "tool_name": tool_name,
                "is_allowed": bool(row["is_allowed"]),
                "rate_limit_per_minute": row["rate_limit_per_minute"],
                "service": cat.get("service", "Other"),
                "description": cat.get("description", ""),
            })
        return {"permissions": by_role}
    except Exception as exc:
        logger.error("get_tool_permissions failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tool permissions",
        )


class ToolPermissionUpdate(BaseModel):
    role: str
    tool_name: str
    is_allowed: Optional[bool] = None
    rate_limit_per_minute: Optional[int] = None


@app.put("/api/admin/tool-permissions")
async def update_tool_permission(
    request: ToolPermissionUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Update a tool permission (toggle enabled or change rate limit).
    Requires CSR_SUPERVISOR role.
    Clears the RBAC cache after update so MCP servers pick up changes.
    """
    user_role: str = str(current_user.get("role", ""))
    if user_role != "CSR_SUPERVISOR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tool permission management requires CSR_SUPERVISOR role",
        )
    try:
        from databases.connections import get_mysql
        db = get_mysql()

        # Build dynamic SET clause based on which fields were provided
        set_parts = []
        params = []
        if request.is_allowed is not None:
            set_parts.append("is_allowed = %s")
            params.append(request.is_allowed)
        if request.rate_limit_per_minute is not None:
            if request.rate_limit_per_minute < 1 or request.rate_limit_per_minute > 1000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="rate_limit_per_minute must be between 1 and 1000",
                )
            set_parts.append("rate_limit_per_minute = %s")
            params.append(request.rate_limit_per_minute)

        if not set_parts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of is_allowed or rate_limit_per_minute must be provided",
            )

        params.extend([request.role, request.tool_name])
        query = f"UPDATE tool_permissions SET {', '.join(set_parts)} WHERE role = %s AND tool_name = %s"
        affected = db.execute_update(query, tuple(params))

        if affected == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No permission found for role={request.role}, tool={request.tool_name}",
            )

        # Clear RBAC cache so changes take effect immediately
        cache_result = rbac_service.clear_cache()
        logger.info(
            "Tool permission updated: role=%s tool=%s by=%s | cache cleared: %s",
            request.role, request.tool_name, current_user.get("sub"), cache_result,
        )

        return {
            "status": "updated",
            "role": request.role,
            "tool_name": request.tool_name,
            "is_allowed": request.is_allowed,
            "rate_limit_per_minute": request.rate_limit_per_minute,
            "cache_cleared": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("update_tool_permission failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tool permission",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)