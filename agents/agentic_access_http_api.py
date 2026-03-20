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
import redis as redis_lib

# Ensure the project root is on sys.path before any project imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings, Settings
from agents.security import auth_service, audit_logger, create_access_token, decode_access_token, AuthenticationError
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
logger.info("Logging configured at level: %s", os.getenv("LOG_LEVEL", "INFO"))

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)