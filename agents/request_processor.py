"""
Request Processor
=================
Main entry point for all user interactions with the CSIP agent system.

Applies ten security controls in a defence-in-depth sequence before and
after invoking the CentralSupervisor graph:

    Pre-agent controls
    ──────────────────
    0. Circuit breaker     — reject immediately if emergency stop is active
    1. Input validation    — NeMo Guardrails: jailbreak / off-topic detection
    2. Authorization       — RBAC: user role must have AGENT:QUERY permission
    3. Rate limiting       — per-user global request rate (120 req/min)
    4. Input sanitization  — nh3: strip all HTML for XSS prevention
    5. Memory security     — Presidio: vault PII/PHI before it enters the graph

    Agent execution
    ───────────────
    6. CentralSupervisor   — plans and delegates to remote team supervisors
                             (tool authorization, human-in-the-loop, tool-level
                             rate limiting, and tool execution all happen inside
                             the graph via MCP decorators and team supervisors)

    Post-agent controls
    ───────────────────
    7. Output validation   — Guardrails AI: validate and sanitize agent response
    8. DLP post-scan       — defence-in-depth scan on sanitized output
    9. Audit logging       — log completed request with execution metadata
"""

import logging
import time
from typing import Any, Dict, List, Optional

import nh3
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from agents.central_supervisor import get_central_supervisor_graph
from agents.core.context_graph import get_context_graph_manager
from agents.core.state import SupervisorState
from agents.security import rbac_service, audit_logger, get_rate_limiter, RateLimitError
from security.approval_workflow import get_approval_workflow, CircuitBreakerError
from security.dlp_scanner import get_dlp_scanner
from security.guardrails_output_validation import get_output_validator
from security.nemo_guardrails_integration import get_nemo_filter
from security.presidio_memory_security import get_presidio_security
from observability.prometheus_metrics import (
    input_sanitization_operations,
    input_validation_latency,
    memory_security_latency,
    output_validation_latency,
    request_processing_latency,
    requests_blocked,
    successful_resolutions,
    track_audit_log,
    track_authorization_denial,
    track_input_validation,
    track_memory_security,
    track_output_validation,
    track_rate_limit_exceeded,
    user_queries,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field

@dataclass
class ProcessResult:
    """
    Structured result returned by process_user_request.

    Carries the validated response text alongside execution metadata so
    callers (e.g. the HTTP API layer) can surface path and tool results
    without bypassing the security pipeline.
    """
    response:       str
    execution_path: List[str]            = field(default_factory=list)
    tool_results:   Dict[str, Any]       = field(default_factory=dict)
    error_count:    int                  = 0

# ---------------------------------------------------------------------------
# Canned responses — defined once, referenced by name
# ---------------------------------------------------------------------------
_RESP_CIRCUIT_BREAKER = (
    "The system is temporarily unavailable due to an emergency stop. "
    "Please contact your supervisor."
)
_RESP_RATE_LIMITED = (
    "You have exceeded the maximum request rate. "
    "Please wait a moment before trying again."
)
_RESP_UNAUTHORIZED   = "You do not have permission to query the agent system."
_RESP_OUTPUT_BLOCKED = (
    "I apologize, but I cannot provide that information due to security policies."
)
_RESP_GENERIC_ERROR  = (
    "I apologize, but an error occurred while processing your request. "
    "Please try again later."
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def process_user_request(
    user_input: str,
    user_id: str,
    user_role: str,
    session_id: str,
    member_id: Optional[str] = None,
    prior_session_id: Optional[str] = None,
) -> ProcessResult:
    """
    Process a user request through the full security and agent pipeline.

    Args:
        user_input:  The user's natural-language query.
        session_id:  Unique session identifier (UUID string).
        user_id:     Authenticated user identifier (email or username).
        user_role:   Casbin role string (e.g. "csr", "csr_supervisor", "member").
        member_id:   Health-plan member ID when the user is a member.

    Returns:
        ProcessResult with the validated response and execution metadata.
    """
    # Count every inbound query regardless of outcome
    user_queries.labels(
        user_role=user_role or "unknown",
        query_type="agent_query",
    ).inc()

    audit_logger.log_action(
        user_id=user_id,
        action="REQUEST_RECEIVED",
        resource_type="AGENT",
        resource_id=session_id,
        changes={"input_length": len(user_input)},
        status="INITIATED",
    )

    try:
        with request_processing_latency.time():
            # ----------------------------------------------------------------
            # 0. CIRCUIT BREAKER PRE-CHECK
            # ----------------------------------------------------------------
            # Checked before anything else so a hard emergency stop is
            # enforced at the outermost boundary with zero agent work done.
            approval_workflow = get_approval_workflow()
            if approval_workflow.is_circuit_breaker_active():
                return ProcessResult(response=_block(
                    user_id=user_id,
                    session_id=session_id,
                    action="REQUEST_BLOCKED_CIRCUIT_BREAKER",
                    control="circuit_breaker",
                    reason="emergency_stop",
                    response=_RESP_CIRCUIT_BREAKER,
                    log_level="critical",
                ))

            # ----------------------------------------------------------------
            # 1. INPUT VALIDATION  (NeMo Guardrails)
            # ----------------------------------------------------------------
            logger.info("[%s] Control 1: input validation", session_id)
            nemo_filter = get_nemo_filter()

            # When this is a follow-up query, fetch the last Q&A pair from the
            # prior session so NeMo sees conversational context. Without this,
            # follow-up queries like "Can you lookup the member?" look like cold
            # PII probes and get blocked. Fetching is best-effort and lightweight
            # — only the Session node is read, and the AI response is truncated
            # to ~200 chars to keep NeMo's token budget minimal.
            prior_turns = []
            if prior_session_id:
                try:
                    _prior_session = get_context_graph_manager().get_session_context(prior_session_id)
                    if _prior_session and _prior_session.get("session"):
                        import json as _json
                        _conv_raw = _prior_session["session"].get("conversationMessages", "")
                        if _conv_raw:
                            _conv = _json.loads(_conv_raw) if isinstance(_conv_raw, str) else _conv_raw
                            _last_human = ""
                            _last_ai = ""
                            for _m in _conv:
                                _mtype = _m.get("type", "")
                                _mcontent = (_m.get("data") or {}).get("content", "")
                                if _mtype == "human" and _mcontent:
                                    _last_human = _mcontent
                                elif _mtype == "ai" and _mcontent:
                                    _last_ai = _mcontent
                            if _last_human:
                                prior_turns.append({"role": "user", "content": _last_human[:300]})
                            if _last_ai:
                                prior_turns.append({"role": "assistant", "content": _last_ai[:200]})
                except Exception as _prior_exc:
                    logger.debug(
                        "[%s] Failed to fetch prior conversation for NeMo (non-fatal): %s",
                        session_id, _prior_exc,
                    )

            t0 = time.time()
            with input_validation_latency.time():
                validation_result = nemo_filter.validate_input(
                    user_input,
                    context={
                        "session_id":  session_id,
                        "user_id":     user_id,
                        "user_role":   user_role,
                        "prior_turns": prior_turns,
                    },
                )
            track_input_validation(
                result=validation_result,
                user_role=user_role or "unknown",
                latency=time.time() - t0,
            )

            if not validation_result["safe"]:
                return ProcessResult(response=_block(
                    user_id=user_id,
                    session_id=session_id,
                    action="INPUT_BLOCKED",
                    control="input_validation",
                    reason=validation_result.get("reason", "unknown"),
                    response=validation_result["response"],
                    log_level="warning",
                    extra={"reason": validation_result["reason"]},
                ))

            # NeMo returns the input with its own light sanitization applied
            sanitized_input = validation_result["sanitized_input"]
            logger.info("[%s] Control 1 passed", session_id)

            # ----------------------------------------------------------------
            # 2. AUTHORIZATION  (Casbin RBAC)
            # ----------------------------------------------------------------
            logger.info("[%s] Control 2: authorization", session_id)

            if not rbac_service.check_permission(user_role, "AGENT", "QUERY"):
                track_authorization_denial(
                    user_role=user_role or "unknown",
                    resource_type="AGENT",
                    action="QUERY",
                )
                return ProcessResult(response=_block(
                    user_id=user_id,
                    session_id=session_id,
                    action="QUERY_DENIED",
                    control="authorization",
                    reason="insufficient_permissions",
                    response=_RESP_UNAUTHORIZED,
                    log_level="warning",
                ))

            logger.info("[%s] Control 2 passed", session_id)

            # ----------------------------------------------------------------
            # 3. GLOBAL RATE LIMITING
            # ----------------------------------------------------------------
            logger.info("[%s] Control 3: rate limiting", session_id)

            try:
                get_rate_limiter().check_rate_limit(
                    user_id=user_id,
                    resource_type="REQUEST",
                    resource_name="global_request",
                    limit_per_minute=120,
                )
                logger.info("[%s] Control 3 passed", session_id)
            except RateLimitError as exc:
                track_rate_limit_exceeded(
                    user_id=user_id,
                    tool_name="global_request",
                    user_role=user_role or "unknown",
                )
                return ProcessResult(response=_block(
                    user_id=user_id,
                    session_id=session_id,
                    action="REQUEST_RATE_LIMITED",
                    control="rate_limiting",
                    reason="global_request_limit",
                    response=_RESP_RATE_LIMITED,
                    log_level="warning",
                    extra={"reason": str(exc)},
                ))

            # ----------------------------------------------------------------
            # 3.5 TOKEN BUDGET ENFORCEMENT
            # ----------------------------------------------------------------
            try:
                from agents.core.budget_controller import get_budget_controller, BudgetExceededError
                _budget_ctrl = get_budget_controller()
                if _budget_ctrl is not None:
                    _budget_ctrl.check_budget(user_id=user_id, user_role=user_role or "")
                    logger.info("[%s] Control 3.5 passed (budget ok)", session_id)
            except BudgetExceededError as _be:
                logger.warning("[%s] Control 3.5: budget exceeded — %s", session_id, _be)
                return ProcessResult(response=_block(
                    user_id=user_id,
                    session_id=session_id,
                    action="TOKEN_BUDGET_EXCEEDED",
                    control="token_budget",
                    reason=str(_be),
                    response="Your daily token usage limit has been reached. Please try again tomorrow or contact your administrator.",
                    log_level="warning",
                    extra={"used": _be.used, "limit": _be.limit},
                ))
            except Exception as _be:
                # Budget enforcement errors are non-fatal — log and continue
                logger.warning("[%s] Control 3.5: budget check error (non-fatal): %s", session_id, _be)

            # ----------------------------------------------------------------
            # 4. INPUT SANITIZATION  (nh3 HTML stripping)
            # ----------------------------------------------------------------
            logger.info("[%s] Control 4: input sanitization", session_id)

            sanitized_input = nh3.clean(sanitized_input, tags=set(), attributes={})
            input_sanitization_operations.labels(
                sanitization_type="xss_prevention"
            ).inc()

            # ----------------------------------------------------------------
            # 5. MEMORY SECURITY  (Presidio PII/PHI vaulting)
            # ----------------------------------------------------------------
            logger.info("[%s] Control 5: memory security", session_id)

            t0 = time.time()
            with memory_security_latency.time():
                scrubbed_input, vault_id, entities_found = (
                    get_presidio_security().scrub_before_storage(
                        sanitized_input,
                        namespace=f"session:{session_id}",
                    )
                )
            track_memory_security(
                entities_found=entities_found,
                latency=time.time() - t0,
            )
            logger.info(
                "[%s] Control 5 passed — vault_id=%s entities_scrubbed=%d types=%s",
                session_id,
                vault_id,
                sum(entities_found.values()) if entities_found else 0,
                list(entities_found.keys()) if entities_found else [],
            )

            # ----------------------------------------------------------------
            # 6. AGENT EXECUTION  (CentralSupervisor)
            # ----------------------------------------------------------------
            # Controls 7–9 (tool authorization, human-in-the-loop, tool-level
            # rate limiting, and tool execution) are enforced inside the graph
            # by MCP decorators and individual team supervisors.
            logger.info("[%s] Invoking CentralSupervisor", session_id)

            initial_state: SupervisorState = {
                # HumanMessage is required — the central supervisor reads
                # state["messages"][-1].content to extract the user query.
                "messages":           [HumanMessage(content=scrubbed_input)],
                "session_id":         session_id,
                "user_id":            user_id,
                "user_role":          user_role or "",
                "member_id":          member_id or "",
                # Execution tracking — must be dict, not list (team supervisors
                # merge tool results by worker name key).
                "tool_results":       {},
                "execution_path":     [],
                # Error tracking — field names match SupervisorState exactly.
                "error_count":        0,
                "error_history":      [],
                "retry_count":        0,
                "prior_session_id":   prior_session_id or "",
            }

            # ── Create Session node in the Context Graph ──────────────────
            # Must be called BEFORE graph.invoke() so the Session node exists
            # when the central supervisor fires track_agent_execution and
            # store_plan. Without this, the planner AgentExecution and
            # CentralPlan nodes are created with no Session to link to,
            # resulting in an orphaned subgraph in the CG.
            try:
                get_context_graph_manager().create_session(
                    session_id=session_id,
                    user_id=user_id,
                    user_role=user_role or "",
                    member_id=member_id or "",
                )
            except Exception as _cg_exc:
                logger.warning(
                    "[%s] CG create_session failed (non-fatal): %s",
                    session_id, _cg_exc,
                )

            # Link the new session to the prior session so the CG captures
            # the conversation chain:
            #   Session1 -[:HAS_FOLLOW_UP]-> Session2 -[:HAS_FOLLOW_UP]-> Session3
            if prior_session_id:
                try:
                    get_context_graph_manager().link_follow_up_session(
                        prior_session_id=prior_session_id,
                        new_session_id=session_id,
                    )
                except Exception as _link_exc:
                    logger.warning(
                        "[%s] CG link_follow_up_session failed (non-fatal): %s",
                        session_id, _link_exc,
                    )

            graph = get_central_supervisor_graph()

            # thread_id = current session_id always.
            # ContextGraphStore.get_tuple() reconstructs prior conversation
            # history by traversing the HAS_FOLLOW_UP chain via chainDepth
            # and rootSessionId properties — no compound IDs needed.
            _graph_config = RunnableConfig(
                configurable={"thread_id": session_id}
            )
            result = graph.invoke(initial_state, config=_graph_config)

            # Extract the final agent response from the last AIMessage
            raw_response = _extract_response(result)
            logger.info("[%s] CentralSupervisor complete", session_id)

            # ----------------------------------------------------------------
            # 7. OUTPUT VALIDATION  (Guardrails AI)
            # ----------------------------------------------------------------
            logger.info("[%s] Control 7: output validation", session_id)

            t0 = time.time()
            with output_validation_latency.time():
                output_result = get_output_validator().validate_output(
                    raw_response,
                    guard_type="standard",
                    metadata={
                        "user_id":    user_id,
                        "session_id": session_id,
                        "user_role":  user_role,
                    },
                )
            track_output_validation(
                result=output_result,
                guard_type="standard",
                latency=time.time() - t0,
            )

            if not output_result["valid"]:
                return ProcessResult(response=_block(
                    user_id=user_id,
                    session_id=session_id,
                    action="OUTPUT_BLOCKED",
                    control="output_validation",
                    reason=output_result.get("reason", "unknown"),
                    response=_RESP_OUTPUT_BLOCKED,
                    log_level="warning",
                    extra={"reason": output_result.get("reason", "unknown")},
                ))

            final_response = output_result["sanitized_output"]
            logger.info("[%s] Control 7 passed", session_id)

            # ----------------------------------------------------------------
            # 8. DLP POST-SCAN  (defence-in-depth, best-effort)
            # ----------------------------------------------------------------
            # Runs on the already-sanitized output. Never blocks the response —
            # uses DLP-redacted text when flagged and escalates via audit log.
            final_response = _run_dlp_scan(
                response=final_response,
                session_id=session_id,
                user_id=user_id,
            )

            # ----------------------------------------------------------------
            # 9. AUDIT LOGGING
            # ----------------------------------------------------------------
            t0 = time.time()
            audit_logger.log_action(
                user_id=user_id,
                action="REQUEST_COMPLETE",
                resource_type="AGENT",
                resource_id=session_id,
                changes={
                    "execution_path":  result.get("execution_path", []),
                    "tool_count":      len(result.get("tool_results", {})),
                    "response_length": len(final_response),
                },
                status="SUCCESS",
            )
            track_audit_log(
                action_type="REQUEST_COMPLETE",
                resource_type="AGENT",
                status="SUCCESS",
                latency=time.time() - t0,
            )

            # ----------------------------------------------------------------
            successful_resolutions.labels(query_type="agent_query").inc()
            logger.info("[%s] Request complete", session_id)
            return ProcessResult(
                response=final_response,
                execution_path=result.get("execution_path", []),
                tool_results=result.get("tool_results", {}),
                error_count=result.get("error_count", 0),
            )

    except CircuitBreakerError as exc:
        logger.critical("[%s] CircuitBreakerError: %s", session_id, exc)
        audit_logger.log_action(
            user_id=user_id,
            action="REQUEST_BLOCKED_CIRCUIT_BREAKER",
            resource_type="AGENT",
            resource_id=session_id,
            changes={"error": str(exc)},
            status="BLOCKED",
        )
        return ProcessResult(response=_RESP_CIRCUIT_BREAKER)

    except Exception as exc:
        logger.error("[%s] Unhandled error: %s", session_id, exc, exc_info=True)
        audit_logger.log_action(
            user_id=user_id,
            action="REQUEST_FAILED",
            resource_type="AGENT",
            resource_id=session_id,
            changes={"error": str(exc)},
            status="ERROR",
        )
        track_audit_log(
            action_type="REQUEST_FAILED",
            resource_type="AGENT",
            status="ERROR",
        )
        return ProcessResult(response=_RESP_GENERIC_ERROR)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _block(
    user_id: str,
    session_id: str,
    action: str,
    control: str,
    reason: str,
    response: str,
    log_level: str = "warning",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log, audit, and return a blocked-request response.

    Centralises the repetitive log + metric + audit pattern that every
    blocked control path shares, removing six copies of identical code.
    """
    getattr(logger, log_level)(
        "[%s] %s blocked by %s: %s", session_id, action, control, reason
    )
    requests_blocked.labels(control_name=control, reason=reason).inc()
    audit_logger.log_action(
        user_id=user_id,
        action=action,
        resource_type="AGENT",
        resource_id=session_id,
        changes=extra or {"reason": reason},
        status="BLOCKED",
    )
    return response


def _extract_response(result: Dict[str, Any]) -> str:
    """
    Extract the final text response from the CentralSupervisor result state.

    The supervisor returns state with messages as LangChain message objects.
    The last AIMessage carries the final answer; earlier messages may include
    intermediate HumanMessages injected during step delegation.

    Falls back to a generic error string if no AIMessage is found, which
    prevents a bare empty string from reaching output validation.
    """
    messages = result.get("messages", [])

    # Walk backwards to find the last AIMessage — most reliable approach
    # since the graph may append multiple messages during multi-step execution.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content or ""
        # Tolerate plain dicts if an older code path produces them
        if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
            return msg.get("content", "")

    # If the graph produced an error state, surface it as the response
    # so output validation can catch and sanitize it rather than returning
    # a misleading empty string.
    error = result.get("error", "")
    if error:
        return error

    logger.warning("CentralSupervisor returned no AIMessage in state")
    return _RESP_GENERIC_ERROR


def _run_dlp_scan(response: str, session_id: str, user_id: str) -> str:
    """
    Run DLP post-scan on the Guardrails-sanitized output.

    This is a defence-in-depth layer — it never blocks the response.
    If flagged, the DLP-redacted text replaces the response and an
    audit escalation is logged. Any scan failure is swallowed so the
    response always reaches the user.

    Returns the (possibly redacted) response string.
    """
    try:
        dlp_result = get_dlp_scanner().scan_output(
            text=response,
            agent_id=f"request_processor:{session_id}",
            action="post_guardrails_scan",
        )

        if not dlp_result.safe:
            logger.warning(
                "[%s] DLP flagged RESTRICTED content (%d entities) "
                "after Guardrails — escalating",
                session_id, len(dlp_result.entities),
            )
            if dlp_result.redacted_text:
                response = dlp_result.redacted_text

            requests_blocked.labels(
                control_name="dlp_post_scan",
                reason="restricted_content_after_guardrails",
            ).inc()
            audit_logger.log_action(
                user_id=user_id,
                action="DLP_ESCALATION",
                resource_type="AGENT",
                resource_id=session_id,
                changes={
                    "sensitivity":  dlp_result.sensitivity.value,
                    "entity_count": len(dlp_result.entities),
                    "warnings":     dlp_result.warnings,
                },
                status="WARNING",
            )
        else:
            logger.info(
                "[%s] DLP post-scan clean — sensitivity=%s entities=%d",
                session_id, dlp_result.sensitivity.value, len(dlp_result.entities),
            )

    except Exception as exc:
        logger.error("[%s] DLP post-scan error (non-fatal): %s", session_id, exc)

    return response
