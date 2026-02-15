"""
Prometheus Metrics for Security Controls and System Observability

This module defines all Prometheus metrics for monitoring the health insurance AI platform,
with a focus on security control effectiveness, performance, and incident detection.

Metrics are organized by:
1. Security Controls (10 layers)
2. Agent Performance
3. System Health
4. Business Metrics

Usage:
    from observability.prometheus_metrics import (
        input_validation_latency,
        track_input_validation,
        track_memory_security
    )
    
    # Track input validation
    with input_validation_latency.time():
        result = validate_input(user_input)
    track_input_validation(result, user_role)
"""

from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# ============================================
# CONTROL 1: INPUT VALIDATION (NeMo Guardrails)
# ============================================

input_validation_failures = Counter(
    'input_validation_failures_total',
    'Total number of inputs blocked by NeMo Guardrails',
    ['reason', 'user_role', 'severity']
)

input_validation_latency = Histogram(
    'input_validation_latency_seconds',
    'Input validation latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
)

jailbreak_attempts = Counter(
    'jailbreak_attempts_total',
    'Total number of jailbreak attempts detected',
    ['attack_type', 'user_role']
)

topic_violations = Counter(
    'topic_violations_total',
    'Total number of off-topic inputs blocked',
    ['detected_topic', 'user_role']
)


# ============================================
# CONTROL 2: AUTHENTICATION & AUTHORIZATION (Casbin)
# ============================================

authentication_attempts = Counter(
    'authentication_attempts_total',
    'Total number of authentication attempts',
    ['status', 'method']  # status: success, failure; method: jwt, oauth, basic
)

authorization_checks = Counter(
    'authorization_checks_total',
    'Total number of authorization checks performed',
    ['resource_type', 'action', 'result']  # result: allowed, denied
)

authorization_denials = Counter(
    'authorization_denials_total',
    'Total number of authorization denials',
    ['user_role', 'resource_type', 'action']
)

rbac_policy_evaluations = Histogram(
    'rbac_policy_evaluation_latency_seconds',
    'RBAC policy evaluation latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)


# ============================================
# CONTROL 3: INPUT SANITIZATION
# ============================================

input_sanitization_operations = Counter(
    'input_sanitization_operations_total',
    'Total number of input sanitization operations',
    ['sanitization_type']  # xss, sql_injection, html_strip
)

malicious_patterns_detected = Counter(
    'malicious_patterns_detected_total',
    'Total number of malicious patterns detected and sanitized',
    ['pattern_type']  # script_tag, sql_keyword, command_injection
)


# ============================================
# CONTROL 4: MEMORY SECURITY (Microsoft Presidio)
# ============================================

memory_security_scrubs = Counter(
    'memory_security_scrubs_total',
    'Total number of PII/PHI entities scrubbed by Presidio',
    ['entity_type']  # PERSON, SSN, PHONE_NUMBER, EMAIL_ADDRESS, etc.
)

memory_security_latency = Histogram(
    'memory_security_latency_seconds',
    'Memory security scrubbing latency in seconds',
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
)

phi_entities_detected = Counter(
    'phi_entities_detected_total',
    'Total number of PHI entities detected',
    ['phi_type']  # MEMBER_ID, CLAIM_NUMBER, PA_NUMBER, MEDICAL_RECORD_NUMBER
)

anonymization_operations = Counter(
    'anonymization_operations_total',
    'Total number of anonymization operations',
    ['operation_type']  # encrypt, hash, redact, mask
)

vault_storage_operations = Counter(
    'vault_storage_operations_total',
    'Total number of vault storage operations',
    ['operation']  # store, retrieve
)


# ============================================
# CONTROL 5: TOOL AUTHORIZATION
# ============================================

tool_authorization_checks = Counter(
    'tool_authorization_checks_total',
    'Total number of tool authorization checks',
    ['tool_name', 'user_role', 'result']
)

tool_execution_denials = Counter(
    'tool_execution_denials_total',
    'Total number of tool execution denials',
    ['tool_name', 'user_role', 'reason']
)


# ============================================
# CONTROL 6: HUMAN-IN-THE-LOOP
# ============================================

approval_requests = Counter(
    'approval_requests_total',
    'Total number of approval requests sent to humans',
    ['action_type', 'risk_level']
)

approval_responses = Counter(
    'approval_responses_total',
    'Total number of approval responses',
    ['action_type', 'decision']  # decision: approved, rejected, timeout
)

approval_latency = Histogram(
    'approval_latency_seconds',
    'Time taken for human approval',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
)

circuit_breaker_triggers = Counter(
    'circuit_breaker_triggers_total',
    'Total number of circuit breaker triggers',
    ['agent_name', 'reason']
)


# ============================================
# CONTROL 7: RATE LIMITING & CIRCUIT BREAKER
# ============================================

rate_limit_checks = Counter(
    'rate_limit_checks_total',
    'Total number of rate limit checks',
    ['endpoint', 'user_id', 'result']
)

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['endpoint', 'user_id', 'limit_type']  # limit_type: per_minute, per_hour, per_day
)

circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Current circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['service_name']
)

circuit_breaker_transitions = Counter(
    'circuit_breaker_transitions_total',
    'Total number of circuit breaker state transitions',
    ['service_name', 'from_state', 'to_state']
)


# ============================================
# CONTROL 8: TOOL EXECUTION
# ============================================

tool_executions = Counter(
    'tool_executions_total',
    'Total number of tool executions',
    ['tool_name', 'status']  # status: success, failure, timeout
)

tool_execution_latency = Histogram(
    'tool_execution_latency_seconds',
    'Tool execution latency in seconds',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

encrypted_communications = Counter(
    'encrypted_communications_total',
    'Total number of encrypted communications',
    ['protocol', 'direction']  # protocol: tls, mtls; direction: inbound, outbound
)


# ============================================
# CONTROL 9: OUTPUT VALIDATION (Guardrails AI)
# ============================================

output_validation_failures = Counter(
    'output_validation_failures_total',
    'Total number of outputs blocked by Guardrails AI',
    ['reason', 'guard_type', 'severity']
)

output_validation_latency = Histogram(
    'output_validation_latency_seconds',
    'Output validation latency in seconds',
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
)

pii_detected_in_output = Counter(
    'pii_detected_in_output_total',
    'Total number of PII entities detected in output',
    ['entity_type']
)

toxicity_detected = Counter(
    'toxicity_detected_total',
    'Total number of toxic outputs blocked',
    ['toxicity_type']  # profanity, hate_speech, threat
)

hallucination_detected = Counter(
    'hallucination_detected_total',
    'Total number of hallucinations detected',
    ['detection_method']  # citation_missing, fact_check_failed
)


# ============================================
# CONTROL 10: AUDIT LOGGING
# ============================================

audit_log_entries = Counter(
    'audit_log_entries_total',
    'Total number of audit log entries created',
    ['action_type', 'resource_type', 'status']
)

audit_log_write_latency = Histogram(
    'audit_log_write_latency_seconds',
    'Audit log write latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

audit_log_failures = Counter(
    'audit_log_failures_total',
    'Total number of audit log write failures',
    ['reason']
)


# ============================================
# OVERALL REQUEST METRICS
# ============================================

request_processing_latency = Histogram(
    'request_processing_latency_seconds',
    'Total request processing latency including all security controls',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0]
)

security_controls_applied = Counter(
    'security_controls_applied_total',
    'Total number of security controls successfully applied',
    ['control_name', 'status']  # status: success, skipped, failed
)

security_control_latency = Histogram(
    'security_control_latency_seconds',
    'Latency of individual security controls',
    ['control_name'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

requests_blocked = Counter(
    'requests_blocked_total',
    'Total number of requests blocked by security controls',
    ['control_name', 'reason']
)


# ============================================
# SECURITY INCIDENTS
# ============================================

security_incidents = Counter(
    'security_incidents_total',
    'Total number of security incidents detected',
    ['incident_type', 'severity']  # severity: low, medium, high, critical
)

false_positives = Counter(
    'security_false_positives_total',
    'Total number of false positive security blocks',
    ['control_name', 'reported_by']
)

security_alerts = Counter(
    'security_alerts_total',
    'Total number of security alerts triggered',
    ['alert_type', 'severity']
)


# ============================================
# AGENT PERFORMANCE METRICS
# ============================================

agent_invocations = Counter(
    'agent_invocations_total',
    'Total number of agent invocations',
    ['agent_name', 'status']
)

agent_execution_latency = Histogram(
    'agent_execution_latency_seconds',
    'Agent execution latency in seconds',
    ['agent_name'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

agent_errors = Counter(
    'agent_errors_total',
    'Total number of agent errors',
    ['agent_name', 'error_type']
)


# ============================================
# SYSTEM HEALTH METRICS
# ============================================

active_sessions = Gauge(
    'active_sessions',
    'Number of currently active user sessions'
)

database_connections = Gauge(
    'database_connections',
    'Number of active database connections',
    ['database_type']  # neo4j_kg, neo4j_cg, mysql, chroma
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    ['component']
)


# ============================================
# BUSINESS METRICS
# ============================================

user_queries = Counter(
    'user_queries_total',
    'Total number of user queries',
    ['user_role', 'query_type']
)

successful_resolutions = Counter(
    'successful_resolutions_total',
    'Total number of successfully resolved queries',
    ['query_type']
)

user_satisfaction = Histogram(
    'user_satisfaction_score',
    'User satisfaction scores (1-5)',
    buckets=[1, 2, 3, 4, 5]
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def track_input_validation(
    result: Dict[str, Any],
    user_role: str,
    latency: Optional[float] = None
) -> None:
    """
    Track input validation metrics.
    
    Args:
        result: Validation result from NeMo Guardrails
        user_role: User's role (member, csr, csr_supervisor)
        latency: Validation latency in seconds (optional)
    """
    if not result.get("safe", True):
        reason = result.get("reason", "unknown")
        severity = result.get("severity", "medium")
        
        input_validation_failures.labels(
            reason=reason,
            user_role=user_role,
            severity=severity
        ).inc()
        
        # Track specific violation types
        if "jailbreak" in reason.lower():
            jailbreak_attempts.labels(
                attack_type=result.get("attack_type", "unknown"),
                user_role=user_role
            ).inc()
        
        if "topic" in reason.lower():
            topic_violations.labels(
                detected_topic=result.get("detected_topic", "unknown"),
                user_role=user_role
            ).inc()
    
    if latency:
        input_validation_latency.observe(latency)


def track_memory_security(
    entities_found: Dict[str, int],
    latency: Optional[float] = None
) -> None:
    """
    Track memory security scrubbing metrics.
    
    Args:
        entities_found: Dictionary of entity types and counts
        latency: Scrubbing latency in seconds (optional)
    """
    for entity_type, count in entities_found.items():
        memory_security_scrubs.labels(
            entity_type=entity_type
        ).inc(count)
        
        # Track PHI-specific entities
        if entity_type in ["MEMBER_ID", "CLAIM_NUMBER", "PA_NUMBER", "MEDICAL_RECORD_NUMBER"]:
            phi_entities_detected.labels(
                phi_type=entity_type
            ).inc(count)
    
    if latency:
        memory_security_latency.observe(latency)


def track_output_validation(
    result: Dict[str, Any],
    guard_type: str,
    latency: Optional[float] = None
) -> None:
    """
    Track output validation metrics.
    
    Args:
        result: Validation result from Guardrails AI
        guard_type: Type of guard used (standard, strict, minimal)
        latency: Validation latency in seconds (optional)
    """
    if not result.get("valid", True):
        reason = result.get("reason", "unknown")
        severity = result.get("severity", "medium")
        
        output_validation_failures.labels(
            reason=reason,
            guard_type=guard_type,
            severity=severity
        ).inc()
        
        # Track specific failure types
        if "pii" in reason.lower() or "phi" in reason.lower():
            pii_detected_in_output.labels(
                entity_type=result.get("entity_type", "unknown")
            ).inc()
        
        if "toxic" in reason.lower():
            toxicity_detected.labels(
                toxicity_type=result.get("toxicity_type", "unknown")
            ).inc()
        
        if "hallucination" in reason.lower():
            hallucination_detected.labels(
                detection_method=result.get("detection_method", "unknown")
            ).inc()
    
    if latency:
        output_validation_latency.observe(latency)


def track_authorization_denial(
    user_role: str,
    resource_type: str,
    action: str
) -> None:
    """
    Track authorization denials.
    
    Args:
        user_role: User's role
        resource_type: Type of resource (MEMBER, CLAIM, PA, PROVIDER)
        action: Action attempted (READ, WRITE, DELETE, QUERY)
    """
    authorization_denials.labels(
        user_role=user_role,
        resource_type=resource_type,
        action=action
    ).inc()


def track_rate_limit_exceeded(
    endpoint: str,
    user_id: str,
    limit_type: str = "per_minute"
) -> None:
    """
    Track rate limit violations.
    
    Args:
        endpoint: API endpoint
        user_id: User identifier
        limit_type: Type of rate limit (per_minute, per_hour, per_day)
    """
    rate_limit_exceeded.labels(
        endpoint=endpoint,
        user_id=user_id,
        limit_type=limit_type
    ).inc()


def track_security_incident(
    incident_type: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Track security incidents.
    
    Args:
        incident_type: Type of incident (jailbreak, phi_leak, unauthorized_access, etc.)
        severity: Severity level (low, medium, high, critical)
        details: Additional incident details (optional)
    """
    security_incidents.labels(
        incident_type=incident_type,
        severity=severity
    ).inc()
    
    logger.warning(
        f"Security incident detected: {incident_type} (severity: {severity})",
        extra={"details": details or {}}
    )


def track_agent_execution(
    agent_name: str,
    latency: float,
    status: str = "success",
    error_type: Optional[str] = None
) -> None:
    """
    Track agent execution metrics.
    
    Args:
        agent_name: Name of the agent
        latency: Execution latency in seconds
        status: Execution status (success, failure, timeout)
        error_type: Type of error if failed (optional)
    """
    agent_invocations.labels(
        agent_name=agent_name,
        status=status
    ).inc()
    
    agent_execution_latency.labels(
        agent_name=agent_name
    ).observe(latency)
    
    if status == "failure" and error_type:
        agent_errors.labels(
            agent_name=agent_name,
            error_type=error_type
        ).inc()


def track_audit_log(
    action_type: str,
    resource_type: str,
    status: str,
    latency: Optional[float] = None
) -> None:
    """
    Track audit logging metrics.
    
    Args:
        action_type: Type of action (READ, WRITE, DELETE, QUERY, etc.)
        resource_type: Type of resource
        status: Status (SUCCESS, FAILURE, BLOCKED)
        latency: Log write latency in seconds (optional)
    """
    audit_log_entries.labels(
        action_type=action_type,
        resource_type=resource_type,
        status=status
    ).inc()
    
    if latency:
        audit_log_write_latency.observe(latency)


# ============================================
# CONTEXT MANAGERS FOR TIMING
# ============================================

class SecurityControlTimer:
    """Context manager for timing security controls."""
    
    def __init__(self, control_name: str):
        self.control_name = control_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        latency = time.time() - self.start_time
        
        security_control_latency.labels(
            control_name=self.control_name
        ).observe(latency)
        
        status = "success" if exc_type is None else "failed"
        security_controls_applied.labels(
            control_name=self.control_name,
            status=status
        ).inc()


# ============================================
# INITIALIZATION
# ============================================

def initialize_metrics():
    """Initialize all metrics with default values."""
    logger.info("Initializing Prometheus metrics for security controls")
    
    # Initialize gauges
    active_sessions.set(0)
    
    for db_type in ["neo4j_kg", "neo4j_cg", "mysql", "chroma"]:
        database_connections.labels(database_type=db_type).set(0)
    
    for service in ["central_supervisor", "member_services", "claim_services", "pa_services", "provider_services"]:
        circuit_breaker_state.labels(service_name=service).set(0)  # 0 = closed
    
    logger.info("Prometheus metrics initialized successfully")


# Initialize on module import
initialize_metrics()
