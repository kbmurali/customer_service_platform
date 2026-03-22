"""
Claims Services: Member Claims Worker Agent

Retrieves all claims for a given member ID using the member_claims
MCP tool. This enables the CSR to ask "What claims does member M-12345
have?" without knowing individual claim numbers.
"""

from typing import Dict, Any
import logging
import time

from langgraph.prebuilt import create_react_agent

from agents.teams.claims_services.claims_services_mcp_tool_client import ClaimServicesMCPToolClient

from agents.security import AuditLogger

from llm_providers.llm_provider_factory import LLMProviderFactory, get_factory, ChatModel
from agents.core.error_handling import get_error_metrics, is_retryable_error, classify_error, check_tool_result_for_errors
from agents.core.mcp_tool_client_base import ToolExecutionError

from security.presidio_memory_security import get_presidio_security
from security.nh3_sanitization import sanitize_html

from config.settings import get_settings, Settings

from observability.langfuse_integration import get_langfuse_tracer
from observability.prometheus_metrics import track_memory_security


logger = logging.getLogger(__name__)

WORKER_PROMPT = (
    "You are a claims information specialist for a health insurance company. "
    "Your role is to retrieve all claims filed by a specific member. "
    "You MUST call the member_claims tool to answer — never answer from memory or context. "
    "Look up the claims using the member ID provided in the query. "
    "If the user asks about a specific status (e.g. pending, denied), pass it as the status argument. "
    "You must also use user ID, user role, and session ID to provide accurate details. "
    "The context includes an 'Execution ID' value. "
    "Pass it as the execution_id argument when calling the tool "
    "so the Context Graph can trace this execution."
)


class MemberClaimsWorker:
    """Worker agent for retrieving claims by member ID."""

    def __init__(self):
        self.name = "member_claims_worker"

        mcp_client = ClaimServicesMCPToolClient()
        tool = mcp_client.get_tool("member_claims")
        if tool is None:
            raise RuntimeError("member_claims not found in ClaimServicesMCPToolClient")
        self.tool = tool
        self.tool_name = self.tool.name

        self.audit = AuditLogger()
        self.presidio = get_presidio_security()

        llm_factory: LLMProviderFactory = get_factory()
        llm: ChatModel = llm_factory.get_llm_provider()

        self.agent = create_react_agent(llm, [self.tool], prompt=WORKER_PROMPT)

    def execute(self, query: str, user_id: str, user_role: str, session_id: str, execution_id: str = "") -> Dict[str, Any]:
        """Execute MemberClaimsWorker task with error handling and retry logic."""
        settings: Settings = get_settings()

        tracer = get_langfuse_tracer()
        metrics = get_error_metrics()
        max_retries = settings.AGENT_MAX_RETRIES
        retry_count = 0

        while retry_count <= max_retries:
            try:
                clean_query = sanitize_html(query)

                if tracer.enabled:
                    tracer.trace_agent_execution(
                        name=self.name,
                        agent_type="worker",
                        input_data=clean_query,
                        output_data=None,
                        user_id=user_id,
                        session_id=session_id
                    )

                contextualized_query = (
                    f"User Role: {user_role}\n"
                    f"User ID: {user_id}\n"
                    f"Session ID: {session_id}\n"
                    f"Execution ID: {execution_id}\n"
                    f"Query: {clean_query}"
                )

                agent_inputs = {"messages": [("user", contextualized_query)]}

                callback_handler = tracer.get_callback_handler()
                if callback_handler:
                    result = self.agent.invoke(
                        agent_inputs,
                        config={"callbacks": [callback_handler]}
                    )
                else:
                    result = self.agent.invoke(agent_inputs)

                output_text = result["messages"][-1].content

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

                self.audit.log_action(
                    user_id=user_id,
                    action=self.tool_name,
                    resource_type="CLAIM",
                    resource_id=""
                )

                if tracer.enabled:
                    tracer.trace_agent_execution(
                        name=self.name,
                        agent_type="worker",
                        input_data=clean_query,
                        output_data=scrubbed_output,
                        tools_used=[self.tool_name],
                        user_id=user_id,
                        session_id=session_id
                    )

                if retry_count > 0:
                    metrics.record_recovery(self.name, "retry_success")

                _raw_tool_output = ""
                for _msg in result.get("messages", []):
                    if getattr(_msg, "type", None) == "tool" or _msg.__class__.__name__ == "ToolMessage":
                        _raw_tool_output = getattr(_msg, "content", "") or ""
                        if isinstance(_raw_tool_output, list):
                            _raw_tool_output = "".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in _raw_tool_output
                            )
                        break

                return {
                    "output": scrubbed_output,
                    "tool_raw_output": _raw_tool_output,
                    "vault_id": vault_id,
                    "worker": self.name,
                    "retry_count": retry_count
                }

            except ToolExecutionError as e:
                logger.error(f"{self.name} tool error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                metrics.record_error(self.name, e.error_type or "tool_error", e.is_retryable())
                if not e.is_retryable() or retry_count >= max_retries:
                    return {
                        "error":        str(e),
                        "error_type":   e.error_type or classify_error(str(e)),
                        "is_retryable": e.is_retryable(),
                        "retry_count":  retry_count,
                    }
                retry_count += 1
                metrics.record_retry(self.name, retry_count)
                backoff_delay = min(2 ** retry_count, settings.AGENT_RETRY_BACKOFF_DELAY_SECONDS)
                logger.info(f"Retrying {self.name} in {backoff_delay}s...")
                time.sleep(backoff_delay)
                continue

            except Exception as e:
                logger.error(f"{self.name} error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                error_msg = str(e)
                is_retryable = is_retryable_error(error_msg)
                error_type = classify_error(error_msg)
                metrics.record_error(self.name, error_type, is_retryable)
                if not is_retryable or retry_count >= max_retries:
                    return {
                        "error":        error_msg,
                        "error_type":   error_type,
                        "is_retryable": is_retryable,
                        "retry_count":  retry_count
                    }
                retry_count += 1
                metrics.record_retry(self.name, retry_count)
                backoff_delay = min(2 ** retry_count, settings.AGENT_RETRY_BACKOFF_DELAY_SECONDS)
                logger.info(f"Retrying {self.name} in {backoff_delay} seconds...")
                time.sleep(backoff_delay)

        return {"error": "Max retries exceeded", "error_type": "max_retries_exceeded"}
