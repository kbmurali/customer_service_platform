"""
Error handling utilities for the hierarchical agent system.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


# Error type classifications
class ErrorType:
    """Error type constants for classification."""
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_NOT_FOUND = "resource_not_found"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    VALIDATION_ERROR = "validation_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


# User-friendly error messages
ERROR_MESSAGES = {
    ErrorType.PERMISSION_DENIED: "You don't have permission to access this information. Please contact your administrator.",
    ErrorType.RESOURCE_NOT_FOUND: "The requested information could not be found. Please verify the ID and try again.",
    ErrorType.TOOL_EXECUTION_FAILED: "We're experiencing technical difficulties. Please try again in a few moments.",
    ErrorType.RATE_LIMIT_EXCEEDED: "Too many requests. Please wait a moment and try again.",
    ErrorType.VALIDATION_ERROR: "The provided information is invalid. Please check your input and try again.",
    ErrorType.DATABASE_ERROR: "We're having trouble accessing our database. Please try again later.",
    ErrorType.NETWORK_ERROR: "Network connection issue. Please check your connection and try again.",
    ErrorType.TIMEOUT_ERROR: "The request took too long to process. Please try again.",
    ErrorType.UNKNOWN_ERROR: "An unexpected error occurred. Please contact support if this persists."
}


def classify_error(error_message: str) -> str:
    """
    Classify an error message into an error type.
    
    Args:
        error_message: The error message to classify
        
    Returns:
        Error type constant
    """
    error_lower = error_message.lower()
    
    if "permission" in error_lower or "access denied" in error_lower or "unauthorized" in error_lower:
        return ErrorType.PERMISSION_DENIED
    elif "not found" in error_lower or "does not exist" in error_lower:
        return ErrorType.RESOURCE_NOT_FOUND
    elif "rate limit" in error_lower or "too many requests" in error_lower:
        return ErrorType.RATE_LIMIT_EXCEEDED
    elif "validation" in error_lower or "invalid" in error_lower:
        return ErrorType.VALIDATION_ERROR
    elif "database" in error_lower or "sql" in error_lower or "connection" in error_lower:
        return ErrorType.DATABASE_ERROR
    elif "network" in error_lower or "timeout" in error_lower:
        return ErrorType.NETWORK_ERROR
    elif "timeout" in error_lower or "timed out" in error_lower:
        return ErrorType.TIMEOUT_ERROR
    else:
        return ErrorType.UNKNOWN_ERROR


def is_retryable_error(error_message: str) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error_message: The error message to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    error_type = classify_error(error_message)
    
    # Retryable errors
    # Rate limits are NOT retryable here — fail fast and let the caller retry
    # after the limit window resets rather than burning time on backoff loops.
    retryable_types = {
        ErrorType.TOOL_EXECUTION_FAILED,
        ErrorType.DATABASE_ERROR,
        ErrorType.NETWORK_ERROR,
        ErrorType.TIMEOUT_ERROR,
    }
    
    return error_type in retryable_types


def format_error_for_user(error_message: str) -> str:
    """
    Format an error message for user-friendly display.
    
    Args:
        error_message: The technical error message
        
    Returns:
        User-friendly error message
    """
    error_type = classify_error(error_message)
    return ERROR_MESSAGES.get(error_type, ERROR_MESSAGES[ErrorType.UNKNOWN_ERROR])


def create_error_record(
    worker_name: str,
    error_message: str,
    error_type: Optional[str] = None,
    is_retryable: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Create a structured error record for error history.
    
    Args:
        worker_name: Name of the worker that encountered the error
        error_message: The error message
        error_type: Optional error type (will be classified if not provided)
        is_retryable: Optional retryable flag (will be determined if not provided)
        
    Returns:
        Structured error record
    """
    if error_type is None:
        error_type = classify_error(error_message)
    
    if is_retryable is None:
        is_retryable = is_retryable_error(error_message)
    
    return {
        "worker": worker_name,
        "error": error_message,
        "error_type": error_type,
        "is_retryable": is_retryable,
        "timestamp": datetime.now().isoformat(),
        "user_message": format_error_for_user(error_message)
    }


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay in seconds
        
    Returns:
        Wrapped function with retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        retry_count = 0
        delay = initial_delay
        
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = str(e)
                
                # Check if error is retryable
                if not is_retryable_error(error_message):
                    logger.error(f"Non-retryable error in {func.__name__}: {error_message}")
                    raise
                
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {error_message}")
                    raise
                
                # Log retry attempt
                logger.warning(f"Retry {retry_count}/{max_retries} for {func.__name__} after {delay}s: {error_message}")
                
                # Wait with exponential backoff
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
        
        # Should not reach here, but just in case
        raise Exception(f"Max retries exceeded for {func.__name__}")
    
    return wrapper


class ErrorMetrics:
    """
    Error metrics tracking for monitoring.
    Uses Prometheus metrics if available.
    """
    
    def __init__(self):
        """Initialize error metrics."""
        try:
            from prometheus_client import Counter, Histogram
            
            def get_or_create_counter(name, description, labels):
                try:
                    return Counter(name, description, labels)
                except ValueError:
                    # Already registered, create reference without registering
                    return Counter(name, description, labels, registry=None)

            def get_or_create_histogram(name, description, labels):
                try:
                    return Histogram(name, description, labels)
                except ValueError:
                    return Histogram(name, description, labels, registry=None)
            
            self.error_counter = get_or_create_counter(
                'agent_errors_total',
                'Total number of agent errors',
                ['worker', 'error_type', 'is_retryable']
            )
            
            self.retry_counter = get_or_create_counter(
                'agent_retries_total',
                'Total number of retry attempts',
                ['worker', 'retry_count']
            )
            
            self.error_recovery_counter = get_or_create_counter(
                'agent_error_recoveries_total',
                'Total number of successful error recoveries',
                ['worker', 'error_type']
            )
            
            self.error_duration_histogram = get_or_create_histogram(
                'agent_error_duration_seconds',
                'Time spent handling errors',
                ['worker', 'error_type']
            )
            
            self.metrics_enabled = True
            logger.info("Error metrics initialized with Prometheus")
        except ImportError:
            self.metrics_enabled = False
            logger.warning("Prometheus not available, error metrics disabled")
    
    def record_error(self, worker: str, error_type: str, is_retryable: bool):
        """Record an error occurrence."""
        if self.metrics_enabled:
            self.error_counter.labels(
                worker=worker,
                error_type=error_type,
                is_retryable=str(is_retryable)
            ).inc()
    
    def record_retry(self, worker: str, retry_count: int):
        """Record a retry attempt."""
        if self.metrics_enabled:
            self.retry_counter.labels(
                worker=worker,
                retry_count=str(retry_count)
            ).inc()
    
    def record_recovery(self, worker: str, error_type: str):
        """Record a successful error recovery."""
        if self.metrics_enabled:
            self.error_recovery_counter.labels(
                worker=worker,
                error_type=error_type
            ).inc()
    
    def record_error_duration(self, worker: str, error_type: str, duration: float):
        """Record time spent handling an error."""
        if self.metrics_enabled:
            self.error_duration_histogram.labels(
                worker=worker,
                error_type=error_type
            ).observe(duration)


# Global error metrics instance
_error_metrics = None

def get_error_metrics() -> ErrorMetrics:
    """Get or create the global error metrics instance."""
    global _error_metrics
    if _error_metrics is None:
        _error_metrics = ErrorMetrics()
    return _error_metrics


def check_tool_result_for_errors(agent_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Inspect the agent message list for tool-level error responses.

    MCP tool decorators (rate limit, circuit breaker, permission denied,
    pending approval) and tool runtime failures all return structured JSON
    with an "error" key.  The LangGraph ReAct agent captures this as a
    ToolMessage in the message list and then composes a natural-language
    summary as the final AIMessage — so checking only the last message
    text misses the error entirely.

    This function scans every ToolMessage in the result for an "error" key
    and returns a normalised error dict if one is found, or None if all
    tool calls succeeded.

    Args:
        agent_result: The dict returned by agent.invoke()

    Returns:
        {"error": str, "error_type": str, "is_retryable": bool} if a tool
        error was found, None otherwise.
    """
    import json as _json

    messages = agent_result.get("messages", [])

    for msg in messages:
        # ToolMessage carries the raw JSON string the MCP tool returned.
        # content may be a list of content blocks (MCP protocol) or a plain string.
        if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                # Extract text from MCP content blocks: [{"type":"text","text":"..."}]
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text") or block.get("content") or "")
                    elif isinstance(block, str):
                        parts.append(block)
                content = "".join(parts)
            try:
                payload = _json.loads(content)
            except (ValueError, TypeError):
                logger.warning("  ToolMessage content not JSON: %s", repr(content)[:200])
                continue

            # langchain_mcp_adapters wraps the MCP response as a JSON array:
            # ["<escaped-json-string>", null]
            # Unwrap: take the first non-null element and parse it again.
            if isinstance(payload, list):
                inner = next((x for x in payload if x is not None), None)
                if inner is None:
                    continue
                if isinstance(inner, str):
                    try:
                        payload = _json.loads(inner)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(inner, dict):
                    payload = inner
                else:
                    continue

            if isinstance(payload, dict) and "error" in payload:
                error_msg  = payload["error"]
                # Preserve structured error_type from decorator if present,
                # otherwise classify from the message text
                error_type = payload.get(
                    "error_type",
                    classify_error(error_msg),
                )
                is_retryable = is_retryable_error(error_msg)
                logger.debug(
                    "Tool error detected in ToolMessage: type=%s retryable=%s msg=%s",
                    error_type, is_retryable, error_msg,
                )
                return {
                    "error":       error_msg,
                    "error_type":  error_type,
                    "is_retryable": is_retryable,
                }
    return None


def handle_worker_error(
    worker_name: str,
    error: Exception,
    state: Dict[str, Any],
    retry_count: int = 0
) -> Dict[str, Any]:
    """
    Handle a worker error and update state accordingly.
    
    Args:
        worker_name: Name of the worker that encountered the error
        error: The exception that was raised
        state: Current state dictionary
        retry_count: Number of retries attempted
        
    Returns:
        Updated state dictionary with error information
    """
    error_message = str(error)
    error_type = classify_error(error_message)
    is_retryable = is_retryable_error(error_message)
    
    # Create error record
    error_record = create_error_record(worker_name, error_message, error_type, is_retryable)
    
    # Update error history
    error_history = state.get("error_history", [])
    error_history.append(error_record)
    
    # Record metrics
    metrics = get_error_metrics()
    metrics.record_error(worker_name, error_type, is_retryable)
    if retry_count > 0:
        metrics.record_retry(worker_name, retry_count)
    
    # Log error
    logger.error(f"Worker {worker_name} error (retry {retry_count}): {error_message}")
    
    # Update state
    return {
        "error": error_message,
        "error_count": state.get("error_count", 0) + 1,
        "error_history": error_history,
        "retry_count": retry_count,
        "is_recoverable": is_retryable,
        "execution_path": state.get("execution_path", []) + [f"{worker_name}_error"]
    }
