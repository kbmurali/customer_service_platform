"""
Member Services: Check Eligibility Worker Agent
"""

from typing import Dict, Any
import logging
import time

from langgraph.prebuilt import create_react_agent

from agents.teams.member_services.member_services_mcp_tool_client import MemberServicesMCPToolClient

from agents.security import RBACService, AuditLogger

from llm_providers.llm_provider_factory import LLMProviderFactory, get_factory, ChatModel
from agents.core.error_handling import get_error_metrics, is_retryable_error, classify_error, check_tool_result_for_errors
from agents.core.mcp_tool_client_base import ToolExecutionError

from security.presidio_memory_security import get_presidio_security
from security.nh3_sanitization import sanitize_html

from config.settings import get_settings, Settings

from observability.langfuse_integration import get_langfuse_tracer
from observability.prometheus_metrics import track_memory_security


logger = logging.getLogger(__name__)

class EligibilityCheckWorker:
    """Worker agent for eligibility check operations."""
    
    def __init__(self):
        self.name = "check_eligibility_worker"

        mcp_client = MemberServicesMCPToolClient()
        tool = mcp_client.get_tool("check_eligibility")
        if tool is None:
            raise RuntimeError("check_eligibility tool not found in MemberServicesMCPToolClient")
        self.tool = tool
        self.tool_name = self.tool.name

        self.rbac = RBACService()
        self.audit = AuditLogger()
        self.presidio = get_presidio_security()
        
        llm_factory: LLMProviderFactory = get_factory()
        llm: ChatModel = llm_factory.get_llm_provider()
        
        prompt = ( 
                    "You are an eligibility verification specialist for a health insurance company. "
                    "Your role is to check member eligibility and coverage status. "
                    "check eligibility using member ID and service date. "
                    "You must also use user ID, user role, and session ID to provide accurate details. "
                    "The query may include a line of the form 'execution_id: <value>'. "
                    "If present, extract that value and pass it as the execution_id "
                    "argument when calling the tool so the Context Graph can trace this execution."
        )
                   
        self.agent = create_react_agent(llm, [self.tool], prompt=prompt)
    
    def execute(self, query: str, user_id: str, user_role: str, session_id: str) -> Dict[str, Any]:
        """Execute EligibilityCheckWorker task with error handling and retry logic."""
        settings: Settings = get_settings()
        tracer = get_langfuse_tracer()
        metrics = get_error_metrics()
        max_retries = settings.AGENT_MAX_RETRIES
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Check tool permission
                if not self.rbac.check_tool_permission(user_role, self.tool_name):
                    error_msg = "Permission denied for check_eligibility"
                    metrics.record_error(self.name, "permission_denied", False)
                    return {"error": error_msg, "error_type": "permission_denied"}

                # Sanitize input
                clean_query = sanitize_html(query)
                
                # Trace agent execution start
                if tracer.enabled:
                    tracer.trace_agent_execution(
                        name=self.name,
                        agent_type="worker",
                        input_data=clean_query,
                        output_data=None,  # Will update on success
                        user_id=user_id,
                        session_id=session_id
                    )
                
                # Prepend context to user message
                # Extract execution_id appended by the supervisor.
                # Arrives as a trailing line: "\nexecution_id: <val>".
                _exec_id  = ""
                _query_lines = clean_query.splitlines()
                _clean_lines = []
                for _line in _query_lines:
                    if _line.startswith("execution_id: "):
                        _exec_id = _line[len("execution_id: "):].strip()
                    else:
                        _clean_lines.append(_line)
                _bare_query = "\n".join(_clean_lines)

                contextualized_query = (
                    f"User Role: {user_role}\n"
                    f"User ID: {user_id}\n"
                    f"Session ID: {session_id}\n"
                    f"Execution ID: {_exec_id}\n"
                    f"Query: {_bare_query}"
                )

                agent_inputs = {"messages": [("user", contextualized_query)]}
                
                # Execute agent with LangFuse callback
                callback_handler = tracer.get_callback_handler()
                if callback_handler:
                    result = self.agent.invoke(
                        agent_inputs,
                        config={"callbacks": [callback_handler]}
                    )
                else:
                    result = self.agent.invoke(agent_inputs)
                
                # Extract output from last message
                output_text = result["messages"][-1].content

                # Scan ToolMessages for structured error JSON before treating the
                # output as success. MCP decorator errors (rate limit, circuit
                # breaker, permission denied, pending approval) and runtime tool
                # failures all return {"error": ...} JSON. The LLM receives this
                # as the ToolMessage content and summarises it — checking the raw
                # ToolMessage catches it before the summary hides it.
                tool_error = check_tool_result_for_errors(result)
                if tool_error:
                    logger.error(
                        f"{self.name} tool error detected: {tool_error['error']}"
                    )
                    metrics.record_error(
                        self.name, tool_error["error_type"], tool_error["is_retryable"]
                    )
                    if not tool_error["is_retryable"] or retry_count >= max_retries:
                        return {
                            "error":        tool_error["error"],
                            "error_type":   tool_error["error_type"],
                            "is_retryable": tool_error["is_retryable"],
                            "retry_count":  retry_count,
                        }
                    retry_count += 1
                    metrics.record_retry(self.name, retry_count)
                    backoff_delay = min(
                        2 ** retry_count,
                        settings.AGENT_RETRY_BACKOFF_DELAY_SECONDS
                    )
                    logger.info(f"Retrying {self.name} in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                    continue

                # Scrub PII/PHI from output
                memory_start = time.time()
                scrubbed_output, vault_id, entities_found = self.presidio.scrub_before_storage(
                    output_text,
                    namespace=session_id,
                    ttl_hours=24
                )
                memory_latency = time.time() - memory_start
                
                if entities_found:
                    track_memory_security(
                        entities_found=entities_found,
                        latency=memory_latency
                    )

                    logger.info(
                        f"[{session_id}] Memory security applied, vault_id: {vault_id}, "
                        f"entities_scrubbed: {sum(entities_found.values()) if entities_found else 0}, "
                        f"types: {list(entities_found.keys()) if entities_found else []}"
                    )
                    
                # Audit action
                self.audit.log_action(
                    user_id=user_id,
                    action=self.tool_name,
                    resource_type="MEMBER",
                    resource_id=""
                )

                # Trace successful execution
                if tracer.enabled:
                    tracer.trace_agent_execution(
                        name=self.name,
                        agent_type="worker",
                        input_data=query,
                        output_data=scrubbed_output,
                        tools_used=[self.tool_name],
                        user_id=user_id,
                        session_id=session_id
                    )
                
                # Record recovery if this was a retry
                if retry_count > 0:
                    metrics.record_recovery(self.name, "retry_success")
                
                return {
                    "output": scrubbed_output,
                    "vault_id": vault_id,
                    "worker": self.name,
                    "retry_count": retry_count
                }
                
            except ToolExecutionError as e:
                logger.error(f"{self.name} tool error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                metrics.record_error(self.name, e.error_type or "tool_error", e.is_retryable())
                if not e.is_retryable() or retry_count >= max_retries:
                    return {
                        "error":       str(e),
                        "error_type":  e.error_type or classify_error(str(e)),
                        "is_retryable": e.is_retryable(),
                        "retry_count": retry_count,
                    }
                retry_count += 1
                metrics.record_retry(self.name, retry_count)
                backoff_delay = min(2 ** retry_count, settings.AGENT_RETRY_BACKOFF_DELAY_SECONDS)
                logger.info(f"Retrying {self.name} in {backoff_delay}s...")
                time.sleep(backoff_delay)
                continue

            except Exception as e:
                logger.error(f"{self.name} error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                
                # Check if error is retryable
                from agents.core.error_handling import is_retryable_error, classify_error
                error_msg = str(e)
                is_retryable = is_retryable_error(error_msg)
                error_type = classify_error(error_msg)
                
                # Record error metrics
                metrics.record_error(self.name, error_type, is_retryable)
                
                # If not retryable or max retries exceeded, return error
                if not is_retryable or retry_count >= max_retries:
                    return {
                        "error": error_msg,
                        "error_type": error_type,
                        "is_retryable": is_retryable,
                        "retry_count": retry_count
                    }
                
                # Retry with exponential backoff
                retry_count += 1
                metrics.record_retry(self.name, retry_count)
                
                default_backoff_delay = settings.AGENT_RETRY_BACKOFF_DELAY_SECONDS
                backoff_delay = min(2 ** retry_count, default_backoff_delay)
                
                logger.info(f"Retrying {self.name} in {backoff_delay} seconds...")
                
                time.sleep(backoff_delay)
        
        # Should not reach here
        return {"error": "Max retries exceeded", "error_type": "max_retries_exceeded"}