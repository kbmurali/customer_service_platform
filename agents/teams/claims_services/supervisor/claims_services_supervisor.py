import logging
import json
import re

from typing import Literal
from datetime import datetime
from functools import lru_cache

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage

from agents.teams.claims_services.supervisor.claim_lookup_worker import ClaimLookupWorker
from agents.teams.claims_services.supervisor.claim_status_worker import ClaimStatusWorker
from agents.teams.claims_services.supervisor.claim_payment_info_worker import ClaimPaymentInfoWorker
from agents.teams.claims_services.supervisor.update_claim_status_worker import UpdateClaimStatusWorker
from agents.teams.claims_services.supervisor.member_claims_worker import MemberClaimsWorker
from agents.core.context_compressor import get_semantic_compressor, get_conversation_compressor
from agents.security import rbac_service, AuditLogger
from agents.core.context_graph import get_context_graph_manager
from agents.core.state import SupervisorState
from agents.core.error_handling import get_error_metrics, create_error_record, format_error_for_user

from databases.chroma_vector_data_access import get_chroma_data_access
from observability.langfuse_integration import get_langfuse_tracer
from llm_providers.llm_provider_factory import LLMProviderFactory, get_factory, ChatModel
from security.approval_workflow import get_approval_workflow
from security.presidio_memory_security import get_presidio_security
from security.nh3_sanitization import sanitize_html

from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
settings: Settings = get_settings()


class ClaimsServicesSupervisor:
    """
    LangGraph-based supervisor for Claims Services team.
    Routes queries to appropriate workers using LangGraph state machine.
    """

    def __init__(self):
        self.name = "claims_services_supervisor"
        self.workers = {
            "claim_lookup":         ClaimLookupWorker(),
            "claim_status":         ClaimStatusWorker(),
            "claim_payment_info":   ClaimPaymentInfoWorker(),
            "update_claim_status":  UpdateClaimStatusWorker(),
            "member_claims":        MemberClaimsWorker(),
        }

        self.rbac = rbac_service
        self.audit = AuditLogger()
        self.presidio = get_presidio_security()
        self.cg_manager = get_context_graph_manager()

        llm_factory: LLMProviderFactory = get_factory()
        self.llm: ChatModel = llm_factory.get_llm_provider()

        # Routing prompt — called once per step to confirm the assigned worker.
        # The supervisor node provides the current step and its pre-assigned worker.
        # The LLM only confirms the worker or returns SKIP if required data is missing.
        # It must never output FINISH or CONTINUE — step advancement is
        # handled entirely by Python logic, not the LLM.
        self.system_prompt = """You are a routing supervisor for a claims services team.

You will be given ONE specific step with a pre-assigned worker.
Your ONLY job is to confirm that worker, or respond SKIP if required data is missing.

Available workers:
- claim_lookup: Look up full claim details by claim ID
- claim_status: Check the processing status of a claim by claim number
- claim_payment_info: Get payment amounts and processing info for a claim by claim ID
- update_claim_status: Update the status of a claim — requires claim ID, new status, and reason
- member_claims: Retrieve all claims for a member by member ID

STRICT RULES:
1. Respond with the exact worker name assigned to the current step.
2. If the step cannot be completed because required information is missing
   (no claim ID, no claim number, no member ID, no new status or reason for update), respond with "SKIP".
3. NEVER respond with FINISH, CONTINUE, or any value not in the worker list.
4. Only use exact worker names: claim_lookup, claim_status, claim_payment_info, update_claim_status, member_claims.

Respond with JSON only — no markdown, no explanation outside the JSON:
{{"next": "worker_name_or_SKIP", "reasoning": "one sentence"}}"""

        # Planning prompt — called once at the start to decompose the query
        # into an ordered list of goals and steps.
        # Goals describe intent only — no worker assignment at goal level.
        # Each step maps to exactly one worker.
        self.planning_prompt = """You are a planning agent for a health insurance customer service system.
Analyze the user query and create an ordered execution plan.

User Query: {user_query}

Relevant Knowledge Base Context:
{semantic_context}

Available workers (use EXACT names only):
- claim_lookup: Looks up full claim details — requires a claim ID
- claim_status: Checks claim processing status — requires a claim number (e.g. CLM-123456)
- claim_payment_info: Retrieves payment amounts and dates — requires a claim ID
- update_claim_status: Updates a claim status — requires a claim ID, new status, and reason
- member_claims: Retrieves all claims for a member — requires a member ID

RULES:
1. Goals describe WHAT to accomplish — no worker assignment at the goal level.
2. Each step must have exactly ONE worker assigned. Use EXACT worker names only.
3. A goal can have one or more steps. Steps sharing a goal_id execute in step_id order.
4. If the query requires claim lookup, make it the first step.
5. Only include steps supported by the available workers.
6. Keep the plan minimal — do not add steps for information not requested.
7. Note: claim_status uses claim NUMBER (e.g. CLM-123456); claim_lookup and
   claim_payment_info use claim ID (UUID). Include the correct identifier in
   the step action so the worker knows which to use.
8. GOAL DECOMPOSITION — create a separate goal for each distinct user intent.
   Distinct intents are questions or requests that address different subjects or
   require different information to answer. Examples of distinct intents that
   MUST be separate goals:
     - "What is the status of claim CLM-123?" AND "How much was paid on claim X?" → 2 goals
     - "Look up claim X" AND "Check status of claim CLM-456" → 2 goals
   A single goal is only correct when all steps serve one unified intent, e.g.:
     - "Look up claim X and tell me how much was paid" → 1 goal, 2 steps (same subject)
9. claim_payment_info and claim_status do not require claim_lookup first unless
   full claim details were explicitly requested — each worker only needs its own
   identifier (claim ID or claim number).
10. update_claim_status is a write operation — only include it when the user
    explicitly requests a status change. It requires claim ID, new status
    (SUBMITTED, UNDER_REVIEW, APPROVED, DENIED), and a reason.
11. member_claims retrieves claims by MEMBER ID (UUID), not by claim ID or claim number.
    Use it when the user asks about "claims for member X" or "what claims does
    member X have". Do not use claim_lookup or claim_status for this — those
    require a specific claim identifier.

Return JSON only (no markdown fences, no explanation):
{{
    "goals": [
        {{
            "id": "goal_1",
            "description": "Short description of what to accomplish",
            "priority": 1
        }}
    ],
    "steps": [
        {{
            "step_id": "step_1",
            "goal_id": "goal_1",
            "action": "Look up claim by ID",
            "worker": "claim_lookup"
        }}
    ]
}}"""

    def create_routing_chain(self):
        """Create the LangGraph routing chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{messages}")
        ])

        return prompt | self.llm | JsonOutputParser()

    def create_plan_node(self, state: SupervisorState) -> SupervisorState:
        """
        First LLM call: Create execution plan with ordered goals.
        """
        # Check if plan already exists
        if state.get("plan"):
            return state

        tracer = get_langfuse_tracer()
        session_id = state.get("session_id", "default")

        try:
            # Get user query
            user_query = state["messages"][-1].content if state["messages"] else "No query"

            # Enrich planning with semantic search from Chroma vector DB
            semantic_context_json = {}

            try:
                chroma = get_chroma_data_access()
                policy_context = chroma.search_policies(query=user_query, n_results=2)
                faq_context = chroma.search_faqs(query=user_query, n_results=2)

                # Compress Chroma documents before injecting into planning prompt
                # to reduce token usage while preserving domain-critical terms.
                _semantic_compressor = get_semantic_compressor()
                _compressed_policies = _semantic_compressor.compress_documents(
                    [{'content': r['document']} for r in policy_context],
                    query=user_query,
                )
                _compressed_faqs = _semantic_compressor.compress_documents(
                    [{'content': r['document']} for r in faq_context],
                    query=user_query,
                )
                semantic_context_json = {
                    'relevant_policies': [d['content'] for d in _compressed_policies],
                    'relevant_faqs':     [d['content'] for d in _compressed_faqs]
                }
            except Exception:
                semantic_context_json = {}

            semantic_context = json.dumps(semantic_context_json, indent=2) if semantic_context_json else 'No additional context available.'

            # Create planning prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a planning agent. Create structured execution plans in JSON format."),
                ("human", self.planning_prompt)
            ])

            # ── CG: Create team planner AgentExecution node ─────────────────
            # Mirrors the central_supervisor_planner pattern so the CG Explorer
            # shows the team planning LLM call with its LangFuse trace card.
            planner_execution_id = None
            try:
                planner_execution_id = self.cg_manager.track_agent_execution(
                    session_id=session_id,
                    agent_name=f"{self.name}_planner",
                    agent_type="team_planner",
                    status="running",
                )
            except Exception:
                pass  # Non-fatal — CG traceability degrades gracefully

            # Trace plan creation — session-aware handler so LangFuse traces
            # are tagged with the CSIP session and user for correlation.
            callback_handler = tracer.get_session_callback_handler(
                session_id=session_id,
                user_id=state.get("user_id"),
            ) if tracer.enabled else None

            # Call LLM to create plan
            inputs = {
                "user_query":       user_query,
                "semantic_context": semantic_context,
            }

            if callback_handler:
                result = (prompt | self.llm).invoke(inputs, config={"callbacks": [callback_handler]})
            else:
                result = (prompt | self.llm).invoke(inputs)

            # Write LangFuse trace ID back to the planner CG node
            if callback_handler and planner_execution_id:
                try:
                    lf_trace_id = callback_handler.get_trace_id()
                    if lf_trace_id:
                        self.cg_manager.set_langfuse_trace_id(planner_execution_id, lf_trace_id)
                except Exception:
                    pass

            # Parse plan
            raw = result.content
            plan_text = re.sub(r"```json|```", "", str(raw)).strip()
            plan = json.loads(plan_text)

            # Store plan in CG as a team plan.
            plan_result = self.cg_manager.store_plan(
                session_id=session_id,
                plan=plan,
                agent_name=self.name,
                plan_type=state.get("plan_type", "team"),
                team_name=state.get("team_name", "claims_services"),
            )
            plan_id  = plan_result.get("plan_id")  if plan_result else None
            step_map = plan_result.get("step_map") if plan_result else {}

            # Link (planner)-[:HAS_PLAN]->(teamPlan) and mark planner completed
            if planner_execution_id and plan_id:
                try:
                    self.cg_manager.link_planner_to_plan(planner_execution_id, plan_id)
                    self.cg_manager.update_execution_status(
                        execution_id=planner_execution_id, status="completed",
                    )
                except Exception:
                    pass

            # Update state
            state["plan_id"]             = plan_id
            state["plan"]                = plan
            state["step_map"]            = step_map
            state["current_step_index"]  = 0
            state["completed_goals"]     = []

            logger.info(f"{self.name}: Created plan with {len(plan.get('goals', []))} goals")

            return state

        except Exception as e:
            logger.error(f"{self.name}: Error creating plan: {e}")

            # Fallback plan
            state["plan"] = {
                "goals": [{"id": "goal_1", "description": "Handle user query",
                           "priority": 1}],
                "steps": [{"step_id": "step_1", "goal_id": "goal_1",
                           "action": "process_query", "worker": ""}]
            }
            state["current_step_index"] = 0
            state["completed_goals"]    = []
            state["step_map"]           = {}
            return state

    def supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """
        Orchestrator node — iterates steps directly, one LLM call per step.

        Flow per invocation:
          1. Check circuit breaker.
          2. Read current_step_index from state.
          3. Build sorted steps list from plan; if all steps done → FINISH.
          4. Read step["worker"] — authoritative, no ambiguity.
          5. Call LLM only to confirm worker / detect SKIP for missing data.
          6. Create AgentExecution, link Step->AgentExecution, store execution_id.
          7. Return next=<worker> or next=FINISH/CONTINUE.

        Step advancement happens in _advance_step (goal_advance node).
        Goal completion is detected there as a side effect of step advancement.
        """
        VALID_WORKERS = {"claim_lookup", "claim_status", "claim_payment_info", "update_claim_status", "member_claims"}

        user_id    = state.get("user_id", "unknown")
        session_id = state.get("session_id", "default")
        tracer     = get_langfuse_tracer()

        # ── Circuit Breaker ──────────────────────────────────────────────────
        try:
            approval_wf = get_approval_workflow()
            if approval_wf.is_circuit_breaker_active():
                logger.warning(f"{self.name}: Circuit breaker active — halting")
                self.audit.log_action(
                    user_id=user_id,
                    action="circuit_breaker_block",
                    resource_type="CLAIMS_SERVICES_AGENT",
                    resource_id=""
                )
                return {
                    "next": "FINISH",
                    "execution_path": state.get("execution_path", []) + ["BLOCKED_BY_CIRCUIT_BREAKER"],
                    "messages": [AIMessage(content="System temporarily unavailable.")],
                }
        except Exception as e:
            logger.warning(f"{self.name}: Circuit breaker check failed (fail-open): {e}")

        # ── Read plan state ──────────────────────────────────────────────────
        plan             = state.get("plan", {})
        plan_id          = state.get("plan_id", "")
        all_steps        = sorted(
            plan.get("steps", []),
            key=lambda s: s.get("step_id", "")
        )
        current_step_idx = state.get("current_step_index", 0)
        completed_goals  = list(state.get("completed_goals", []))
        execution_path   = list(state.get("execution_path", []))

        # ── All steps done ───────────────────────────────────────────────────
        if current_step_idx >= len(all_steps):
            logger.info(f"{self.name}: All {len(all_steps)} steps completed → FINISH")
            try:
                self.cg_manager.complete_plan(session_id, plan_id)
            except Exception:
                pass
            return {
                "next": "FINISH",
                "current_step_index": current_step_idx,
                "completed_goals": completed_goals,
                "execution_path": execution_path + [f"{self.name} -> FINISH (all steps done)"],
            }

        current_step = all_steps[current_step_idx]
        step_id      = current_step.get("step_id", f"step_{current_step_idx}")
        goal_id      = current_step.get("goal_id", "")
        step_worker  = current_step.get("worker", "")

        logger.info(
            f"{self.name}: Step {current_step_idx + 1}/{len(all_steps)} "
            f"(goal={goal_id}): '{current_step.get('action', '')}' worker={step_worker}"
        )

        # ── Context Graph (best-effort) ──────────────────────────────────────
        session_context      = {}
        conversation_history = []
        execution_id         = None    # This is the ROUTER execution node
        try:
            session_context      = self.cg_manager.get_session_context(session_id) or {}
            conversation_history = self.cg_manager.get_conversation_history(session_id, limit=5) or []
            execution_id         = self.cg_manager.track_agent_execution(
                session_id=session_id,
                agent_name=f"{self.name}_router",
                agent_type="team_router",
                status="running",
            )
        except Exception as e:
            logger.warning(f"{self.name}: CG unavailable (non-fatal): {e}")

        # ── Build routing messages ───────────────────────────────────────────
        routing_messages: list[BaseMessage] = list(state.get("messages", []))

        if session_context:
            session_node = session_context.get("session", {})
            routing_messages.append(SystemMessage(content=(
                f"Session: user={user_id} role={session_node.get('userRole', '?')} "
                f"member={session_node.get('memberId', 'N/A')}"
            )))

        if conversation_history:
            # Compress older turns via LLMLingua; keep the most recent 2 verbatim.
            # Returns ready-to-use list[BaseMessage] — no manual role_map needed.
            conversation_compressor = get_conversation_compressor()
            routing_messages.extend(
                conversation_compressor.compress_history(conversation_history)
            )

        # Inject results from previously completed steps so the routing LLM
        # can confirm the assigned worker has the data it needs (e.g. a claim
        # number extracted from a prior claim_lookup result).
        # tool_raw_output is the unredacted structured JSON from the MCP server —
        # preferred over 'output' which may have PII scrubbed (e.g. claim numbers).
        tool_results = state.get("tool_results", {})
        if tool_results and current_step_idx > 0:
            prior_context_parts = []
            for completed_step in all_steps[:current_step_idx]:
                w = completed_step.get("worker", "")
                if w in tool_results:
                    raw = tool_results[w].get("tool_raw_output", "")
                    display = raw if raw else tool_results[w].get("output", "")
                    if display:
                        prior_context_parts.append(f"- {w}: {display}")
            if prior_context_parts:
                routing_messages.append(SystemMessage(content=(
                    "Results from prior steps:\n" + "\n".join(prior_context_parts)
                )))

        # Inject the current step — worker is pre-assigned, LLM only confirms or SKIPs
        routing_messages.append(SystemMessage(content=(
            f"CURRENT STEP ({current_step_idx + 1}/{len(all_steps)}): "
            f"{current_step.get('action', '')}\n"
            f"Assigned worker: {step_worker}\n"
            "Confirm this worker, or respond SKIP if required data is missing."
        )))

        # ── LLM routing call ─────────────────────────────────────────────────
        callback_handler = tracer.get_session_callback_handler(
            session_id=session_id,
            user_id=user_id,
        ) if tracer.enabled else None
        chain = self.create_routing_chain()
        try:
            llm_result  = chain.invoke(
                {"messages": routing_messages},
                config={"callbacks": [callback_handler]} if callback_handler else {},
            )
            next_worker = llm_result.get("next", "SKIP")
            reasoning   = llm_result.get("reasoning", "")
            # Write LangFuse trace ID back to the CG execution node
            if callback_handler and execution_id:
                try:
                    lf_trace_id = callback_handler.get_trace_id()
                    if lf_trace_id:
                        self.cg_manager.set_langfuse_trace_id(execution_id, lf_trace_id)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"{self.name}: LLM routing failed: {e}")
            # Fall back to the step-assigned worker
            next_worker = step_worker if step_worker in VALID_WORKERS else "SKIP"
            reasoning   = f"LLM error fallback: {e}"

        logger.info(f"{self.name}: LLM chose '{next_worker}' — {reasoning}")

        # ── Validate LLM output ──────────────────────────────────────────────
        # If LLM returned something invalid, fall back to the step-assigned worker
        if next_worker not in VALID_WORKERS and next_worker != "SKIP":
            logger.warning(
                f"{self.name}: LLM returned invalid worker '{next_worker}', "
                f"falling back to step worker={step_worker!r}"
            )
            next_worker = step_worker if step_worker in VALID_WORKERS else "SKIP"

        # ── Handle SKIP — advance step without calling a worker ──────────────
        if next_worker == "SKIP":
            logger.info(f"{self.name}: Skipping step {step_id} ({reasoning})")
            execution_path.append(f"{self.name} -> SKIP step {step_id}")

            # Link Step->AgentExecution before marking skipped, mirroring the
            # valid-worker path. Without this the AgentExecution node is orphaned
            # — no (Step)-[:EXECUTED_BY]->(AgentExecution) edge exists for the
            # skipped step, breaking CG traversal and audit queries.
            try:
                if execution_id and plan_id:
                    self.cg_manager.link_step_to_execution(
                        plan_id=plan_id,
                        step_id=step_id,
                        execution_id=execution_id,
                    )
            except Exception as e:
                logger.warning(f"{self.name}: Failed to link skipped step to execution (non-fatal): {e}")

            try:
                if execution_id:
                    self.cg_manager.update_execution_status(execution_id, "skipped")
            except Exception:
                pass

            # Check if this was the last step for its goal
            next_step_idx = current_step_idx + 1
            remaining_goal_steps = [
                s for s in all_steps[next_step_idx:]
                if s.get("goal_id") == goal_id
            ]
            if not remaining_goal_steps:
                # Last step of this goal — mark goal skipped
                completed_goals.append(goal_id)
                try:
                    self.cg_manager.update_plan_progress(
                        session_id=session_id, plan_id=plan_id,
                        goal_id=goal_id, goal_result="skipped",
                    )
                except Exception:
                    pass

            if next_step_idx >= len(all_steps):
                try:
                    self.cg_manager.complete_plan(session_id, plan_id)
                except Exception:
                    pass
                return {
                    "next": "FINISH",
                    "current_step_index": next_step_idx,
                    "completed_goals": completed_goals,
                    "execution_path": execution_path + [f"{self.name} -> FINISH"],
                }
            return {
                "next": "CONTINUE",
                "current_step_index": next_step_idx,
                "completed_goals": completed_goals,
                "execution_path": execution_path,
            }

        # ── Valid worker — link Step->AgentExecution before worker fires ─────
        # Scoped by planId+stepId — no cross-session ambiguity possible.
        try:
            if execution_id and plan_id:
                self.cg_manager.link_step_to_execution(
                    plan_id=plan_id,
                    step_id=step_id,
                    execution_id=execution_id,
                )
        except Exception as e:
            logger.warning(f"{self.name}: Failed to link step to execution (non-fatal): {e}")

        # ── Valid worker — return it for graph routing ────────────────────────
        execution_path.append(f"{self.name} -> {next_worker} (step {step_id})")

        # Mark the router node as completed (do NOT rename to worker)
        try:
            if execution_id:
                self.cg_manager.update_execution_status(
                    execution_id, "completed",
                    routing_note=f"Routing: {next_worker} — {reasoning}",
                )
        except Exception:
            pass

        # Create a separate worker execution node so tools link to it
        worker_execution_id = None
        try:
            worker_execution_id = self.cg_manager.track_agent_execution(
                session_id=session_id,
                agent_name=f"{next_worker}_worker",
                agent_type="worker",
                status="running",
            )
            if execution_id and worker_execution_id:
                self.cg_manager.link_router_to_worker(execution_id, worker_execution_id)
        except Exception:
            pass

        self.audit.log_action(
            user_id=user_id,
            action="claims_services_routing",
            resource_type="CLAIMS_SERVICES_AGENT",
            resource_id="",
        )

        return {
            "next":                 next_worker,
            "execution_path":       execution_path,
            # Stored so worker_node can pass it to the MCP tool, completing:
            #   (AgentExecution)-[:CALLED_TOOL]->(ToolExecution)
            "current_execution_id": worker_execution_id or execution_id or "",
        }

    def _advance_step(self, state: SupervisorState) -> SupervisorState:
        """
        Called after each successful worker execution (via goal_advance node).
        Advances current_step_index. When the last step of a goal completes,
        marks that goal complete as a side effect.
        """
        plan             = state.get("plan", {})
        all_steps        = sorted(
            plan.get("steps", []),
            key=lambda s: s.get("step_id", "")
        )
        current_step_idx = state.get("current_step_index", 0)
        completed_goals  = list(state.get("completed_goals", []))
        session_id       = state.get("session_id", "default")
        plan_id          = state.get("plan_id", "")

        if current_step_idx < len(all_steps):
            current_step = all_steps[current_step_idx]
            goal_id      = current_step.get("goal_id", "")
            next_step_idx = current_step_idx + 1

            # Detect if this was the last step for its goal
            remaining_goal_steps = [
                s for s in all_steps[next_step_idx:]
                if s.get("goal_id") == goal_id
            ]
            if not remaining_goal_steps:
                completed_goals.append(goal_id)
                try:
                    self.cg_manager.update_plan_progress(
                        session_id=session_id, plan_id=plan_id,
                        goal_id=goal_id, goal_result="completed",
                    )
                except Exception:
                    pass
                logger.info(f"Goal {goal_id} completed — all steps done")
        else:
            next_step_idx = current_step_idx + 1

        return {
            "current_step_index": next_step_idx,
            "completed_goals":    completed_goals,
        }

    def worker_node(self, worker_name: str):
        """Create a worker node function for LangGraph with error handling."""
        def node(state: SupervisorState) -> SupervisorState:
            worker     = self.workers[worker_name]
            metrics    = get_error_metrics()
            start_time = datetime.now()

            # Sanitize input
            messages = state.get("messages")

            if messages:
                user_message: BaseMessage = messages[0]
                query = sanitize_html(str(user_message.content))
            else:
                query = "Query not specified"

            # Inject execution_id so the ReAct agent passes it to the MCP tool,
            # completing: (AgentExecution)-[:CALLED_TOOL]->(ToolExecution)
            # EXECUTED_BY is already linked in the routing block above via
            # link_step_to_execution(planId, stepId, executionId).
            _exec_id = state.get("current_execution_id", "")

            # Append results from prior steps so the ReAct agent has upstream
            # data available (e.g. claim number returned by claim_lookup).
            # tool_raw_output is the unredacted structured JSON from the MCP server —
            # preferred over 'output' which may have PII scrubbed.
            tool_results = state.get("tool_results", {})
            if tool_results:
                all_steps_sorted = sorted(
                    state.get("plan", {}).get("steps", []),
                    key=lambda s: s.get("step_id", "")
                )
                current_step_idx = state.get("current_step_index", 0)
                prior_parts = []
                for prior_step in all_steps_sorted[:current_step_idx]:
                    w = prior_step.get("worker", "")
                    if w in tool_results:
                        raw = tool_results[w].get("tool_raw_output", "")
                        display = raw if raw else tool_results[w].get("output", "")
                        if display:
                            prior_parts.append(f"- {w} result: {display}")
                if prior_parts:
                    query += "\nPrior step results:\n" + "\n".join(prior_parts)

            # Execute worker
            result = worker.execute(
                query=query,
                user_id=state.get("user_id", "unknown"),
                user_role=state.get("user_role", "unknown"),
                session_id=state.get("session_id", "default"),
                execution_id=_exec_id
            )

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Check for errors
            if "error" in result:
                error_msg    = result["error"]
                error_type   = result.get("error_type", "unknown")
                is_retryable = result.get("is_retryable", False)
                retry_count  = result.get("retry_count", 0)

                # Create error record
                error_record = create_error_record(
                    worker_name=worker_name,
                    error_message=error_msg,
                    error_type=error_type,
                    is_retryable=is_retryable
                )

                error_record["retry_count"] = retry_count
                error_record["duration_ms"] = duration_ms

                # Update error history
                error_history = state.get("error_history", [])
                error_history.append(error_record)

                # Record error duration metric
                metrics.record_error_duration(worker_name, error_type, duration_ms / 1000)

                # Mark the AgentExecution node as failed — it was set to
                # "completed" when the supervisor routed to this worker, but
                # the worker itself failed, so the status must be corrected.
                try:
                    _exec_id = state.get("current_execution_id", "")
                    if _exec_id:
                        self.cg_manager.update_execution_status(
                            _exec_id, "failed",
                            error_message=error_msg,
                        )
                except Exception:
                    pass

                # Mark the current step's goal as failed and cancel all
                # remaining pending goals so no nodes are left orphaned.
                try:
                    plan          = state.get("plan", {})
                    all_steps     = sorted(
                        plan.get("steps", []),
                        key=lambda s: s.get("step_id", "")
                    )
                    current_idx   = state.get("current_step_index", 0)
                    plan_id_cg    = state.get("plan_id", "")
                    session_id_cg = state.get("session_id", "default")
                    if current_idx < len(all_steps):
                        goal_id = all_steps[current_idx].get("goal_id", "")
                        if goal_id:
                            self.cg_manager.update_plan_progress(
                                session_id=session_id_cg,
                                plan_id=plan_id_cg,
                                goal_id=goal_id,
                                goal_result="failed",
                            )
                    # Cancel every goal still pending — downstream goals that
                    # will never run because this step failed.
                    self.cg_manager.cancel_remaining_goals(
                        session_id=session_id_cg,
                        plan_id=plan_id_cg,
                    )
                    # Mark the plan itself as incomplete.
                    self.cg_manager.fail_plan(
                        session_id=session_id_cg,
                        plan_id=plan_id_cg,
                    )
                except Exception:
                    pass

                logger.error(f"Worker {worker_name} failed: {error_msg}")

                return {
                    "messages":       [RemoveMessage(id=msg.id) for msg in state["messages"] if msg.id],
                    "error":          error_msg,
                    "error_count":    state.get("error_count", 0) + 1,
                    "error_history":  error_history,
                    "retry_count":    retry_count,
                    "is_recoverable": is_retryable,
                    "execution_path": state.get("execution_path", []) + [f"{worker_name}_failed"],
                    "duration_ms":    duration_ms
                }

            # Success path
            # Mark the worker AgentExecution as completed
            try:
                _exec_id = state.get("current_execution_id", "")
                if _exec_id:
                    self.cg_manager.update_execution_status(_exec_id, "completed")
            except Exception:
                pass

            return {
                "messages":       [AIMessage(content=result.get("output", ""))],
                "error":          None,
                "execution_path": state.get("execution_path", []) + [f"{worker_name}_executed"],
                "tool_results": {
                    **state.get("tool_results", {}),
                    worker_name: result
                },
                "duration_ms": duration_ms
            }

        return node

    def error_handler_node(self, state: SupervisorState) -> SupervisorState:
        """Handle errors and determine if recovery is possible."""
        error_history  = state.get("error_history", [])
        error_count    = state.get("error_count", 0)
        current_error  = state.get("error") or "Unknown error"
        is_recoverable = state.get("is_recoverable", False)

        logger.error(f"{self.name} error handler: {error_count} errors, recoverable={is_recoverable}")

        self.audit.log_action(
            user_id=state.get("user_id", "unknown"),
            action="supervisor_workflow_error",
            resource_type="CLAIMS_SERVICES_AGENT",
            resource_id=""
        )

        # Format user-friendly error message
        user_message = format_error_for_user(current_error)

        # Record final error metrics
        metrics = get_error_metrics()
        for error_record in error_history:
            metrics.record_error(
                error_record.get("worker", "unknown"),
                error_record.get("error_type", "unknown"),
                error_record.get("is_retryable", False)
            )

        return {
            "messages":       [SystemMessage(content=user_message)],
            "next":           "FINISH",
            "error":          current_error,
            "execution_path": state.get("execution_path", []) + ["error_handler"]
        }

    def create_graph(self) -> CompiledStateGraph:
        """
        Create LangGraph state machine for the team.

        Flow:
            create_plan
                → supervisor
                    → claim_lookup         → goal_advance → supervisor
                    → claim_status         → goal_advance → supervisor
                    → claim_payment_info   → goal_advance → supervisor
                    → update_claim_status  → goal_advance → supervisor
                    → error_handler → END
                    → END  (when FINISH)
        """
        workflow = StateGraph(SupervisorState)

        workflow.add_node("create_plan",   self.create_plan_node)
        workflow.add_node("supervisor",    self.supervisor_node)
        workflow.add_node("error_handler", self.error_handler_node)
        workflow.add_node("goal_advance",  self._advance_step)

        for worker_name in self.workers.keys():
            workflow.add_node(worker_name, self.worker_node(worker_name))

        # create_plan → supervisor
        workflow.add_edge("create_plan", "supervisor")

        # workers → goal_advance → supervisor (success path)
        # workers → error_handler             (error path, skips goal_advance
        #                                      so the failed goal is never
        #                                      marked completed or advanced)
        def worker_router(state: SupervisorState) -> Literal["goal_advance", "error_handler"]:
            return "error_handler" if state.get("error") else "goal_advance"

        for worker_name in self.workers.keys():
            workflow.add_conditional_edges(
                worker_name,
                worker_router,
                {
                    "goal_advance":  "goal_advance",
                    "error_handler": "error_handler",
                },
            )
        workflow.add_edge("goal_advance", "supervisor")

        # error_handler always terminates
        workflow.add_edge("error_handler", END)

        VALID_WORKERS = {"claim_lookup", "claim_status", "claim_payment_info", "update_claim_status", "member_claims"}

        def router(
            state: SupervisorState,
        ) -> Literal["claim_lookup", "claim_status", "claim_payment_info",
                     "update_claim_status", "member_claims",
                     "error_handler", "supervisor", "__end__"]:
            # Hard error → error handler
            if state.get("error"):
                return "error_handler"

            next_node = state.get("next", "FINISH")

            if next_node == "FINISH":
                return "__end__"

            if next_node == "CONTINUE":
                # Supervisor decided to skip a goal and loop back
                return "supervisor"

            if next_node in VALID_WORKERS:
                return next_node  # type: ignore[return-value]

            # Anything else (bad LLM output that slipped through) → end safely
            logger.warning(
                f"Router received unexpected next='{next_node}', terminating."
            )
            return "__end__"

        workflow.add_conditional_edges(
            "supervisor",
            router,
            {
                "supervisor":          "supervisor",
                "claim_lookup":        "claim_lookup",
                "claim_status":        "claim_status",
                "claim_payment_info":  "claim_payment_info",
                "update_claim_status": "update_claim_status",
                "member_claims":       "member_claims",
                "error_handler":       "error_handler",
                "__end__":             END,
            },
        )

        workflow.set_entry_point("create_plan")
        return workflow.compile()


@lru_cache
def get_claims_services_graph():
    """Get or create claims services LangGraph."""
    claims_services_supervisor = ClaimsServicesSupervisor()

    return claims_services_supervisor.create_graph()
