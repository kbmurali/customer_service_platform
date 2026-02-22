"""
Human-in-the-Loop Approval Workflow & Circuit Breaker

Implements security controls for high-impact actions:
1. Action classification (auto vs. requires approval)
2. Approval queue with Redis
3. Circuit breaker / kill switch
4. Rationale recording for audit

Security Control #5: Human-in-the-Loop & Kill Switch

Database usage:
- MySQL: Durable storage for approval requests and circuit breaker events
- Redis: Ephemeral queue, status polling, circuit breaker active flag, alerts
"""

import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import redis

from databases.connections import get_mysql
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ImpactLevel(Enum):
    """Impact level of agent actions"""
    LOW = "LOW"          # Auto-execute
    MEDIUM = "MEDIUM"    # Log and execute
    HIGH = "HIGH"        # Requires approval
    CRITICAL = "CRITICAL"  # Requires supervisor approval


class ApprovalStatus(Enum):
    """Status of approval request"""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"


@dataclass
class AgentAction:
    """Agent action requiring approval"""
    action_id: str
    agent_id: str
    tool_name: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None


@dataclass
class ApprovalRequest:
    """Approval request details"""
    request_id: str
    action: AgentAction
    impact_level: ImpactLevel
    requested_by: str
    requested_at: datetime
    expires_at: datetime
    status: ApprovalStatus
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_rationale: Optional[str] = None


@dataclass
class ApprovalResult:
    """Result of approval request"""
    approved: bool
    rationale: str
    reviewed_by: str
    reviewed_at: datetime


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is active"""
    pass


class ApprovalDeniedError(Exception):
    """Raised when action is denied"""
    pass


class ApprovalWorkflow:
    """
    Human-in-the-loop approval workflow with circuit breaker.

    Features:
    - Automatic impact classification
    - Redis-backed approval queue for fast polling
    - MySQL-backed durable storage for audit trail
    - Emergency circuit breaker / kill switch
    - Complete audit trail in MySQL
    """

    # Tools classified by impact level for approval routing
    # LOW: Auto-execute silently (read-only lookups and searches)
    LOW_IMPACT_TOOLS = [
        "pa_requirements",
        "provider_search",
        "network_check",
        "search_policy_info",
        "search_medical_codes",
        "search_knowledge_base",
    ]

    # MEDIUM: Log and auto-execute (reserved for future moderate-risk tools)
    MEDIUM_IMPACT_TOOLS = [
        "member_lookup",
        "check_eligibility",
        "coverage_lookup",
        "claim_lookup",
        "claim_status",
        "claim_payment_info",
        "pa_lookup",
        "pa_status",
        "provider_lookup",
    ]

    # HIGH: Requires human approval before execution (write/mutate tools)
    HIGH_IMPACT_TOOLS = [
        "update_claim_status",
        "approve_prior_auth",
        "deny_prior_auth",
        "update_member_info",
    ]

    # CRITICAL: Requires supervisor approval (reserved for future bulk operations)
    CRITICAL_IMPACT_TOOLS = []

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
    ):
        """Initialize approval workflow with MySQL and Redis."""
        self.mysql = get_mysql()

        resolved_host = settings.REDIS_HOST
        resolved_port = settings.REDIS_PORT
        
        self.redis_client = redis.Redis(
            host=resolved_host,
            port=resolved_port,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10
        )

        self._ensure_tables()
        logger.info("Approval Workflow initialized (MySQL + Redis)")

    # ------------------------------------------------------------------
    # Table initialisation
    # ------------------------------------------------------------------

    def _ensure_tables(self):
        """
        Verify that the approval_requests and circuit_breaker_events
        tables exist.  The tables are created by mysql_schema.sql at
        deployment time; this method only logs a warning if they are
        missing so the application can still start.
        """
        try:
            self.mysql.execute_query(
                "SELECT 1 FROM approval_requests LIMIT 1"
            )
            self.mysql.execute_query(
                "SELECT 1 FROM circuit_breaker_events LIMIT 1"
            )
            logger.info("Approval workflow tables verified")
        except Exception as e:
            logger.warning(
                f"Approval workflow tables may not exist yet: {e}. "
                "Run databases/mysql_schema.sql to create them."
            )

    # ------------------------------------------------------------------
    # Synchronous entry point (used by tools)
    # ------------------------------------------------------------------

    def submit_approval_request(
        self,
        agent_id: str,
        action_type: str,
        action_description: str,
        user_id: str,
        user_role: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous entry point for tool-level approval requests.

        Classifies impact from the tool name (action_type) and either
        auto-approves (LOW/MEDIUM) or submits for human review
        (HIGH/CRITICAL).

        Args:
            agent_id: Agent or tool requesting approval
            action_type: The tool / action name (used to derive impact level)
            action_description: Human-readable description
            user_id: User initiating the action
            user_role: User's role
            details: Additional context for the reviewer

        Returns:
            Dict with keys: approved (bool), pending (bool),
            request_id (str|None), message (str)

        Raises:
            CircuitBreakerError: If circuit breaker is active
        """
        # 1. Circuit breaker gate
        if self._is_circuit_breaker_active():
            reason = self.redis_client.get("circuit_breaker:reason") or "Unknown"
            raise CircuitBreakerError(
                f"System halted by emergency stop: {reason}"
            )

        # 2. Classify impact (tool list takes precedence over caller hint)
        impact = self._classify_impact(action_type)

        # 3. Build AgentAction
        action = AgentAction(
            action_id=uuid.uuid4().hex,
            agent_id=agent_id,
            tool_name=action_type,
            parameters=details or {},
            context={"description": action_description, "user_role": user_role},
        )

        # 4. LOW impact → auto-approve silently
        if impact == ImpactLevel.LOW:
            logger.info(f"Auto-approved LOW-impact action: {action_type}")
            return {
                "approved": True,
                "pending": False,
                "request_id": None,
                "message": "Action auto-approved (low impact).",
            }

        # 5. MEDIUM impact → log and auto-approve
        if impact == ImpactLevel.MEDIUM:
            self._log_action_execution(action, user_id, auto_approved=True)
            logger.info(f"Auto-approved MEDIUM-impact action: {action_type}")
            return {
                "approved": True,
                "pending": False,
                "request_id": None,
                "message": "Action auto-approved (medium impact, logged).",
            }

        # 6. HIGH / CRITICAL → submit for human review
        request = self._create_approval_request(action, impact, user_id)
        logger.info(
            f"Submitted {impact.value}-impact approval request "
            f"{request.request_id} for {action_type}"
        )
        return {
            "approved": False,
            "pending": True,
            "request_id": request.request_id,
            "message": (
                f"Action requires {'supervisor ' if impact == ImpactLevel.CRITICAL else ''}"
                f"approval. Request ID: {request.request_id}"
            ),
        }

    # ------------------------------------------------------------------
    # Async entry point (full wait-for-approval flow)
    # ------------------------------------------------------------------

    async def execute_with_approval(
        self,
        action: AgentAction,
        user_id: str,
    ) -> Any:
        """
        Execute action with approval workflow.

        Args:
            action: Agent action to execute
            user_id: User ID for audit

        Returns:
            Action result if approved

        Raises:
            CircuitBreakerError: If circuit breaker is active
            ApprovalDeniedError: If action is denied
        """
        # 1. Check circuit breaker
        if self._is_circuit_breaker_active():
            reason = self.redis_client.get("circuit_breaker:reason") or "Unknown"
            raise CircuitBreakerError(
                f"System halted by emergency stop: {reason}"
            )

        # 2. Classify impact
        impact = self._classify_impact(action.tool_name)

        # 3. LOW impact - execute immediately
        if impact == ImpactLevel.LOW:
            logger.info(f"Auto-executing low-impact action: {action.tool_name}")
            return await self._execute_action(action)

        # 4. MEDIUM impact - log and execute
        if impact == ImpactLevel.MEDIUM:
            logger.info(
                f"Executing medium-impact action with logging: {action.tool_name}"
            )
            self._log_action_execution(action, user_id, auto_approved=True)
            return await self._execute_action(action)

        # 5. HIGH / CRITICAL impact - request approval
        logger.info(
            f"Requesting approval for {impact.value}-impact action: "
            f"{action.tool_name}"
        )
        approval_request = self._create_approval_request(action, impact, user_id)

        # 6. Wait for approval (with timeout)
        timeout = 300 if impact == ImpactLevel.HIGH else 600  # 5 or 10 min
        result = await self._wait_for_approval(
            approval_request.request_id, timeout
        )

        if result.approved:
            logger.info(f"Action approved: {action.tool_name}")
            return await self._execute_action(action)
        else:
            logger.warning(
                f"Action denied: {action.tool_name} - {result.rationale}"
            )
            raise ApprovalDeniedError(result.rationale)

    # ------------------------------------------------------------------
    # Impact classification
    # ------------------------------------------------------------------

    def _classify_impact(self, tool_name: str) -> ImpactLevel:
        """Classify impact level of tool based on explicit tool lists."""
        if tool_name in self.CRITICAL_IMPACT_TOOLS:
            return ImpactLevel.CRITICAL
        elif tool_name in self.HIGH_IMPACT_TOOLS:
            return ImpactLevel.HIGH
        elif tool_name in self.MEDIUM_IMPACT_TOOLS:
            return ImpactLevel.MEDIUM
        elif tool_name in self.LOW_IMPACT_TOOLS:
            return ImpactLevel.LOW
        else:
            # Unknown tools default to MEDIUM (log and auto-approve)
            logger.warning(
                f"Tool '{tool_name}' not found in any impact class; "
                "defaulting to MEDIUM"
            )
            return ImpactLevel.MEDIUM

    # ------------------------------------------------------------------
    # Approval request lifecycle
    # ------------------------------------------------------------------

    def _create_approval_request(
        self,
        action: AgentAction,
        impact: ImpactLevel,
        user_id: str,
    ) -> ApprovalRequest:
        """Create approval request and add to queue."""
        request_id = f"approval_{action.action_id}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=30)

        request = ApprovalRequest(
            request_id=request_id,
            action=action,
            impact_level=impact,
            requested_by=user_id,
            requested_at=now,
            expires_at=expires_at,
            status=ApprovalStatus.PENDING,
        )

        # Store in MySQL (durable)
        self.mysql.execute_update(
            """
            INSERT INTO approval_requests (
                request_id, action_id, agent_id, tool_name, parameters,
                context, impact_level, requested_by, requested_at,
                expires_at, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                request.request_id,
                action.action_id,
                action.agent_id,
                action.tool_name,
                json.dumps(action.parameters),
                json.dumps(action.context) if action.context else None,
                impact.value,
                user_id,
                request.requested_at,
                request.expires_at,
                ApprovalStatus.PENDING.value,
            ),
        )

        # Add to Redis queue for fast access
        self.redis_client.lpush("approval_queue:pending", request_id)
        self.redis_client.setex(
            f"approval_request:{request_id}",
            1800,  # 30 minutes TTL
            json.dumps(
                {
                    "request_id": request_id,
                    "action_id": action.action_id,
                    "agent_id": action.agent_id,
                    "tool_name": action.tool_name,
                    "impact_level": impact.value,
                    "requested_by": user_id,
                }
            ),
        )

        logger.info(f"Created approval request: {request_id}")
        return request

    async def _wait_for_approval(
        self, request_id: str, timeout: int
    ) -> ApprovalResult:
        """Wait for approval decision by polling Redis."""
        start_time = datetime.now(timezone.utc)

        while (datetime.now(timezone.utc) - start_time).total_seconds() < timeout:
            status_key = f"approval_status:{request_id}"
            status_json = self.redis_client.get(status_key)

            if status_json:
                status = json.loads(status_json)
                return ApprovalResult(
                    approved=status["approved"],
                    rationale=status["rationale"],
                    reviewed_by=status["reviewed_by"],
                    reviewed_at=datetime.fromisoformat(status["reviewed_at"]),
                )

            await asyncio.sleep(1)

        # Timeout - mark as expired
        self._expire_approval_request(request_id)
        raise ApprovalDeniedError("Approval request expired")

    # ------------------------------------------------------------------
    # Reviewer actions
    # ------------------------------------------------------------------

    def approve_request(
        self, request_id: str, reviewer_id: str, rationale: str
    ):
        """Approve an approval request."""
        self._update_approval_status(
            request_id, ApprovalStatus.APPROVED, reviewer_id, rationale
        )

        # Notify waiting coroutine via Redis
        self.redis_client.setex(
            f"approval_status:{request_id}",
            300,
            json.dumps(
                {
                    "approved": True,
                    "rationale": rationale,
                    "reviewed_by": reviewer_id,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        )
        logger.info(f"Approval request approved: {request_id} by {reviewer_id}")

    def deny_request(
        self, request_id: str, reviewer_id: str, rationale: str
    ):
        """Deny an approval request."""
        self._update_approval_status(
            request_id, ApprovalStatus.DENIED, reviewer_id, rationale
        )

        # Notify waiting coroutine via Redis
        self.redis_client.setex(
            f"approval_status:{request_id}",
            300,
            json.dumps(
                {
                    "approved": False,
                    "rationale": rationale,
                    "reviewed_by": reviewer_id,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        )
        logger.info(f"Approval request denied: {request_id} by {reviewer_id}")

    # ------------------------------------------------------------------
    # Internal helpers - MySQL persistence
    # ------------------------------------------------------------------

    def _update_approval_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        reviewer_id: str,
        rationale: str,
    ):
        """Update approval status in MySQL."""
        self.mysql.execute_update(
            """
            UPDATE approval_requests
            SET status = %s,
                reviewed_by = %s,
                reviewed_at = %s,
                review_rationale = %s
            WHERE request_id = %s
            """,
            (
                status.value,
                reviewer_id,
                datetime.now(timezone.utc),
                rationale,
                request_id,
            ),
        )

    def _expire_approval_request(self, request_id: str):
        """Mark approval request as expired in MySQL."""
        self.mysql.execute_update(
            """
            UPDATE approval_requests
            SET status = %s
            WHERE request_id = %s
            """,
            (ApprovalStatus.EXPIRED.value, request_id),
        )

    async def _execute_action(self, action: AgentAction) -> Any:
        """Execute the actual action (placeholder - integrate with agent system)."""
        logger.info(
            f"Executing action: {action.tool_name} "
            f"with params {action.parameters}"
        )
        return {"status": "success", "message": f"Executed {action.tool_name}"}

    def _log_action_execution(
        self, action: AgentAction, user_id: str, auto_approved: bool
    ):
        """Log action execution for audit (MEDIUM impact auto-approved)."""
        self.mysql.execute_update(
            """
            INSERT INTO approval_requests (
                request_id, action_id, agent_id, tool_name, parameters,
                context, impact_level, requested_by, requested_at,
                expires_at, status, reviewed_by, reviewed_at,
                review_rationale
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                f"auto_{action.action_id}_{uuid.uuid4().hex[:8]}",
                action.action_id,
                action.agent_id,
                action.tool_name,
                json.dumps(action.parameters),
                json.dumps(action.context) if action.context else None,
                ImpactLevel.MEDIUM.value,
                user_id,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
                ApprovalStatus.APPROVED.value,
                "SYSTEM",
                datetime.now(timezone.utc),
                "Auto-approved based on impact classification",
            ),
        )

    # ------------------------------------------------------------------
    # Circuit Breaker / Kill Switch
    # ------------------------------------------------------------------

    def activate_circuit_breaker(self, reason: str, activated_by: str):
        """
        Emergency stop - halt all agent actions.

        Args:
            reason: Reason for activation
            activated_by: User who activated it
        """
        # Set Redis flags (fast reads on every request)
        self.redis_client.set("circuit_breaker:active", "1")
        self.redis_client.set("circuit_breaker:reason", reason)
        self.redis_client.set("circuit_breaker:activated_by", activated_by)
        self.redis_client.set(
            "circuit_breaker:timestamp", datetime.now(timezone.utc).isoformat()
        )

        # Log to MySQL (durable audit trail)
        self.mysql.execute_update(
            """
            INSERT INTO circuit_breaker_events
                (event_id, event_type, reason, triggered_by,
                 triggered_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                "ACTIVATED",
                reason,
                activated_by,
                datetime.now(timezone.utc),
                json.dumps({"source": "manual"}),
            ),
        )

        logger.critical(
            f"CIRCUIT BREAKER ACTIVATED by {activated_by}: {reason}"
        )

        # Push alert for dashboard
        self._send_alert("CIRCUIT BREAKER ACTIVATED", reason, activated_by)

    def deactivate_circuit_breaker(self, deactivated_by: str, rationale: str):
        """
        Resume operations after review.

        Args:
            deactivated_by: User who deactivated it
            rationale: Reason for deactivation
        """
        # Clear Redis flags
        self.redis_client.delete("circuit_breaker:active")
        self.redis_client.delete("circuit_breaker:reason")
        self.redis_client.delete("circuit_breaker:activated_by")
        self.redis_client.delete("circuit_breaker:timestamp")

        # Log to MySQL
        self.mysql.execute_update(
            """
            INSERT INTO circuit_breaker_events
                (event_id, event_type, reason, triggered_by,
                 triggered_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                "DEACTIVATED",
                rationale,
                deactivated_by,
                datetime.now(timezone.utc),
                json.dumps({"source": "manual"}),
            ),
        )

        logger.warning(
            f"Circuit breaker deactivated by {deactivated_by}: {rationale}"
        )

        self._send_alert(
            "CIRCUIT BREAKER DEACTIVATED", rationale, deactivated_by
        )

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active (sub-ms Redis read)."""
        return self.redis_client.get("circuit_breaker:active") == "1"

    def is_circuit_breaker_active(self) -> bool:
        """Public API: Check if circuit breaker is active."""
        return self._is_circuit_breaker_active()

    def _send_alert(self, title: str, message: str, triggered_by: str):
        """Send alert to administrators via Redis list for dashboard."""
        alert = {
            "title": title,
            "message": message,
            "triggered_by": triggered_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.redis_client.lpush("security:alerts", json.dumps(alert))
        self.redis_client.ltrim("security:alerts", 0, 99)  # Keep last 100
        logger.critical(f"ALERT: {title} - {message}")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_pending_approvals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending approval requests from MySQL."""
        return self.mysql.execute_query(
            """
            SELECT request_id, action_id, agent_id, tool_name,
                   parameters, context, impact_level, requested_by,
                   requested_at, expires_at, status
            FROM approval_requests
            WHERE status = %s
            ORDER BY requested_at DESC
            LIMIT %s
            """,
            (ApprovalStatus.PENDING.value, limit),
        )

    def get_circuit_breaker_history(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get circuit breaker event history from MySQL."""
        return self.mysql.execute_query(
            """
            SELECT event_id, event_type, reason, triggered_by,
                   triggered_at, metadata
            FROM circuit_breaker_events
            ORDER BY triggered_at DESC
            LIMIT %s
            """,
            (limit,),
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_approval_workflow: Optional[ApprovalWorkflow] = None


def get_approval_workflow() -> ApprovalWorkflow:
    """Get global approval workflow instance."""
    global _approval_workflow
    if _approval_workflow is None:
        _approval_workflow = ApprovalWorkflow()
    return _approval_workflow
