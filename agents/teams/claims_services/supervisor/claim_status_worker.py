"""
Claims Services: Claim Status Worker Agent
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
            "You are a claims status specialist for a health insurance company. "
            "Your role is to check the current status of a claim by claim number. "
            "You MUST call the claim_status tool to answer — never answer from memory or context. "
            "Look up the claim status using the claim number provided in the query. "
            "Note: claim number (e.g. CLM-123456) is different from claim ID. "
            "You must also use user ID, user role, and session ID to provide accurate details. "
            "The context includes an 'Execution ID' value. "
            "Pass it as the execution_id argument when calling the tool "
            "so the Context Graph can trace this execution."
        )

class ClaimStatusWorker:
    """Worker agent for claim status check operations."""

    def __init__(self):
        self.name = "claim_status_worker"

        mcp_client = ClaimServicesMCPToolClient()
        tool = mcp_client.get_tool("claim_status")
        if tool is None:
            raise RuntimeError("claim_status not found in ClaimServicesMCPToolClient")
        self.tool = tool
        self.tool_name = self.tool.name

        self.audit = AuditLogger()
        self.presidio = get_presidio_security()

        llm_factory: LLMProviderFactory = get_factory()
        llm: ChatModel = llm_factory.get_llm_provider()


        # Prompt versioning: fetch from LangFuse if enabled, else use module constant
        _prompt = WORKER_PROMPT
        try:
            from config.settings import get_settings as _gs
            _s = _gs()
            if getattr(_s, "LANGFUSE_PROMPT_VERSIONING_ENABLED", False):
                from observability.langfuse_integration import get_langfuse_tracer
                _tracer = get_langfuse_tracer()
                _label = getattr(_s, "LANGFUSE_PROMPT_LABEL", "production")
                _prompt = _tracer.get_prompt_or_default(
                    "csip-claims-status-worker-prompt", WORKER_PROMPT, label=_label
                )
        except Exception:
            pass  # Fall back to hardcoded default

        self.agent = create_react_agent(llm, [self.tool], prompt=_prompt)

    def execute(self, query: str, user_id: str, user_role: str, session_id: str, execution_id: str = "") -> Dict[str, Any]:
        """Execute ClaimStatusWorker task with error handling and retry logic."""
        settings: Settings = get_settings()

        tracer = get_langfuse_tracer()
        metrics = get_error_metrics()
        max_retries = settings.AGENT_MAX_RETRIES
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Sanitize input
                clean_query = sanitize_html(query)

                # Trace agent execution start
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

                # Execute agent with LangFuse callback — session-aware handler
                # so traces are tagged with CSIP session for CG Explorer linking.
                callback_handler = tracer.get_session_callback_handler(
                    session_id=session_id,
                    user_id=user_id,
                ) if tracer.enabled else None
                if callback_handler:
                    result = self.agent.invoke(
                        agent_inputs,
                        config={"callbacks": [callback_handler]}
                    )
                else:
                    result = self.agent.invoke(agent_inputs)

                # Write LangFuse trace ID back to the worker's CG node
                if callback_handler and execution_id:
                    try:
                        from databases.context_graph_data_access import get_cg_data_access
                        lf_trace_id = callback_handler.get_trace_id()
                        if lf_trace_id:
                            get_cg_data_access().set_langfuse_trace_id(execution_id, lf_trace_id)
                    except Exception:
                        pass

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

                # Audit action — claim_status is keyed by claim_number, not claim_id
                self.audit.log_action(
                    user_id=user_id,
                    action=self.tool_name,
                    resource_type="CLAIM",
                    resource_id=""
                )

                # Trace successful execution
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

                # Record recovery if this was a retry
                if retry_count > 0:
                    metrics.record_recovery(self.name, "retry_success")

                # Extract raw ToolMessage content (structured JSON from MCP server,
                # before the LLM summarises and Presidio scrubs it).  Stored in
                # tool_results so downstream steps can access structured fields
                # that may be redacted from the scrubbed output.
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
