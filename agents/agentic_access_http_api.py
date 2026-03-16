"""
FastAPI for CSIP Agentic Access
"""
from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import sys
import os
import uuid

# Ensure the project root is on sys.path before any project imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings, Settings
from agents.security import auth_service, audit_logger, create_access_token, decode_access_token, AuthenticationError
from agents.request_processor import process_user_request, ProcessResult

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

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hierarchical Agentic AI for Health Insurance Customer Service Intelligence Platform"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)