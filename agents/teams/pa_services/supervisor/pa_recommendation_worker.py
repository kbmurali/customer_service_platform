"""
PA Services: PA Recommendation Worker Agent
(Decision agent enhancements — M12)
Unlike standard read workers that call MCP tools, this decision worker
receives evidence from prior plan steps (pa_lookup, pa_requirements,
search_knowledge_base clinical guidelines, treatment_history from
member_services_team) already in its query context and applies clinical
criteria to produce a structured recommendation with justification.
The recommendation and evidence are persisted as first-class properties
directly on the worker's AgentExecution node in the Context Graph,
making them directly queryable without requiring a LangFuse trace lookup.
"""
from typing import Dict, Any
import json
import logging
import time
from agents.security import AuditLogger
from llm_providers.llm_provider_factory import LLMProviderFactory, get_factory, ChatModel
from agents.core.error_handling import get_error_metrics, is_retryable_error, classify_error
from security.presidio_memory_security import get_presidio_security
from security.nh3_sanitization import sanitize_html
from config.settings import get_settings, Settings
from observability.langfuse_integration import get_langfuse_tracer
from observability.prometheus_metrics import track_memory_security

logger = logging.getLogger(__name__)

WORKER_PROMPT = (
    "You are a prior authorization clinical review specialist for a health insurance company.\n"
    "You receive evidence gathered from prior steps about a specific PA request and "
    "must evaluate whether the prior authorization should be approved, denied, or if "
    "additional information is needed.\n\n"
    "EVIDENCE AVAILABLE IN THE QUERY:\n"
    "The query contains results from prior plan steps including:\n"
    "- pa_lookup: PA request details (procedure code, description, urgency, status)\n"
    "- pa_requirements: whether the procedure requires PA under the member's policy type\n"
    "- search_knowledge_base (clinical guidelines): procedure-specific approval criteria "
    "including conservative treatment requirements, imaging requirements, and provider qualifications\n"
    "- treatment_history: member's treatment records showing physical therapy sessions, "
    "medication trials, injections, imaging, and other conservative treatments\n\n"
    "CLINICAL REVIEW RULES:\n"
    "1. Check if the procedure requires PA under the member's policy type. If not required, "
    "recommend APPROVE with note that PA is not required.\n"
    "2. Compare treatment history against the clinical guideline requirements:\n"
    "   a. Has the required duration of conservative treatment been met?\n"
    "   b. Has the minimum number of therapy sessions been completed?\n"
    "   c. Have required medication trials been documented?\n"
    "   d. Is required diagnostic imaging on file?\n"
    "3. If all clinical criteria are met, recommend APPROVE.\n"
    "4. If clinical criteria are clearly NOT met (e.g. no conservative treatment documented "
    "when 6 months is required), recommend DENY with specific criteria not met.\n"
    "5. If some evidence is missing but available criteria are partially met, recommend "
    "REQUEST_INFO specifying exactly what documentation is needed.\n"
    "6. For URGENT or EMERGENCY cases, note the urgency level and whether expedited "
    "review criteria apply per the clinical guidelines.\n\n"
    "You MUST respond with ONLY a JSON object (no markdown, no explanation outside JSON):\n"
    "{\n"
    '    "recommendation": "APPROVE" or "DENY" or "REQUEST_INFO",\n'
    '    "justification_summary": "One paragraph explaining the clinical decision with specific evidence cited",\n'
    '    "evidence_used": ["conservative_tx:12_sessions_PT", "imaging:MRI_on_file", "medication_trial:NSAIDs_6mo"]\n'
    "}\n\n"
    "The evidence_used array must contain short key:value pairs summarizing each "
    "piece of clinical evidence you considered. Use descriptive formats like "
    "conservative_tx:12_sessions_PT, imaging:MRI_on_file, urgency:ROUTINE, "
    "pa_required:YES, guideline_met:YES."
)


class PARecommendationWorker:
    """Decision worker for prior authorization recommendations.

    Does NOT call MCP tools — reasons over evidence from prior plan steps
    and produces a structured recommendation persisted in the Context Graph.
    """

    def __init__(self):
        self.name = "pa_recommendation_worker"
        self.tool_name = "pa_recommendation"  # for audit logging
        self.audit = AuditLogger()
        self.presidio = get_presidio_security()

        llm_factory: LLMProviderFactory = get_factory()
        self.llm: ChatModel = llm_factory.get_llm_provider()

        # Prompt versioning: fetch from LangFuse if enabled, else use module constant
        self.system_prompt = WORKER_PROMPT
        try:
            from config.settings import get_settings as _gs
            _s = _gs()
            if getattr(_s, "LANGFUSE_PROMPT_VERSIONING_ENABLED", False):
                from observability.langfuse_integration import get_langfuse_tracer
                _tracer = get_langfuse_tracer()
                _label = getattr(_s, "LANGFUSE_PROMPT_LABEL", "production")
                self.system_prompt = _tracer.get_prompt_or_default(
                    "csip-pa-recommendation-worker-prompt", WORKER_PROMPT, label=_label
                )
        except Exception:
            pass  # Fall back to hardcoded default

    def execute(self, query: str, user_id: str, user_role: str, session_id: str, execution_id: str = "") -> Dict[str, Any]:
        """Execute PA recommendation with error handling and retry logic."""
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

                # Build messages for the LLM — system prompt + evidence query
                messages = [
                    ("system", self.system_prompt),
                    ("user", clean_query),
                ]

                # Execute LLM with LangFuse callback
                callback_handler = tracer.get_session_callback_handler(
                    session_id=session_id,
                    user_id=user_id,
                ) if tracer.enabled else None

                if callback_handler:
                    result = self.llm.invoke(
                        messages,
                        config={"callbacks": [callback_handler]}
                    )
                else:
                    result = self.llm.invoke(messages)

                # Write LangFuse trace ID back to the worker's CG node
                if callback_handler and execution_id:
                    try:
                        from databases.context_graph_data_access import get_cg_data_access
                        lf_trace_id = callback_handler.get_trace_id()
                        if lf_trace_id:
                            get_cg_data_access().set_langfuse_trace_id(execution_id, lf_trace_id)
                    except Exception:
                        pass

                # Extract LLM response
                output_text = result.content if hasattr(result, "content") else str(result)

                # Parse the structured recommendation JSON
                recommendation = "REQUEST_INFO"
                justification_summary = ""
                evidence_used = []
                try:
                    # Strip markdown fences if present
                    cleaned = output_text.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                    cleaned = cleaned.strip()

                    parsed = json.loads(cleaned)
                    recommendation = parsed.get("recommendation", "REQUEST_INFO").upper()
                    justification_summary = parsed.get("justification_summary", "")
                    evidence_used = parsed.get("evidence_used", [])

                    if recommendation not in ("APPROVE", "DENY", "REQUEST_INFO"):
                        recommendation = "REQUEST_INFO"
                except (json.JSONDecodeError, AttributeError) as parse_err:
                    logger.warning(
                        "%s: failed to parse LLM JSON, defaulting to REQUEST_INFO: %s",
                        self.name, parse_err,
                    )
                    justification_summary = output_text
                    recommendation = "REQUEST_INFO"

                # Persist decision properties directly on the worker's existing
                # AgentExecution node (created by the team supervisor).  This
                # avoids a separate decision node — the CG Explorer renders
                # isDecisionNode properties (DECISION badge, Business Decision
                # panel) on any AgentExecution that carries them.
                try:
                    if execution_id:
                        from databases.context_graph_data_access import get_cg_data_access
                        import json as _json
                        get_cg_data_access().conn.execute_query("""
                            MATCH (w:AgentExecution {executionId: $execId})
                            SET w.isDecisionNode = true,
                                w.recommendation = $recommendation,
                                w.justificationSummary = $justification,
                                w.evidenceUsed = $evidence
                        """, {
                            "execId": execution_id,
                            "recommendation": recommendation,
                            "justification": justification_summary,
                            "evidence": _json.dumps(evidence_used),
                        })
                except Exception as cg_exc:
                    logger.warning(
                        "%s: CG decision tracking failed (non-fatal): %s",
                        self.name, cg_exc,
                    )

                # Build human-readable output for the consolidator
                decision_output = (
                    f"PA RECOMMENDATION: {recommendation}\n\n"
                    f"Justification: {justification_summary}\n\n"
                    f"Evidence considered: {', '.join(evidence_used)}"
                )

                # Scrub PII/PHI from output
                memory_start = time.time()
                scrubbed_output, vault_id, entities_found = self.presidio.scrub_before_storage(
                    decision_output,
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
                    resource_type="PA",
                    resource_id=""
                )

                # Trace successful execution
                if tracer.enabled:
                    tracer.trace_agent_execution(
                        name=self.name,
                        agent_type="worker",
                        input_data=clean_query,
                        output_data=scrubbed_output,
                        tools_used=[],
                        user_id=user_id,
                        session_id=session_id
                    )

                # Record recovery if this was a retry
                if retry_count > 0:
                    metrics.record_recovery(self.name, "retry_success")

                return {
                    "output": scrubbed_output,
                    "tool_raw_output": output_text,
                    "vault_id": vault_id,
                    "worker": self.name,
                    "retry_count": retry_count,
                    "recommendation": recommendation,
                }

            except Exception as e:
                logger.error(f"{self.name} error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                error_msg = str(e)
                is_retryable = is_retryable_error(error_msg)
                error_type = classify_error(error_msg)
                metrics.record_error(self.name, error_type, is_retryable)

                if not is_retryable or retry_count >= max_retries:
                    return {
                        "error": error_msg,
                        "error_type": error_type,
                        "is_retryable": is_retryable,
                        "retry_count": retry_count
                    }

                retry_count += 1
                metrics.record_retry(self.name, retry_count)
                backoff_delay = min(2 ** retry_count, settings.AGENT_RETRY_BACKOFF_DELAY_SECONDS)
                logger.info(f"Retrying {self.name} in {backoff_delay} seconds...")
                time.sleep(backoff_delay)

        return {"error": "Max retries exceeded", "error_type": "max_retries_exceeded"}
