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
from agents.security import RBACService, AuditLogger
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
            "claim_lookup":       ClaimLookupWorker(),
            "claim_status":       ClaimStatusWorker(),
            "claim_payment_info": ClaimPaymentInfoWorker(),
        }

        self.rbac = RBACService()
        self.audit = AuditLogger()
        self.presidio = get_presidio_security()
        self.cg_manager = get_context_graph_manager()

        llm_factory: LLMProviderFactory = get_factory()
        self.llm: ChatModel = llm_factory.get_llm_provider()

        # Routing prompt — called once per goal to pick the right worker.
        # The supervisor node provides the current goal and required_workers
        # so the LLM only confirms the best worker from that restricted list.
        # It must never output FINISH or CONTINUE — goal advancement is
        # handled entirely by Python logic, not the LLM.
        self.system_prompt = """You are a routing supervisor for a claims services team.

You will be given ONE specific goal and the list of workers that can handle it.
Your ONLY job is to select the single best worker from the provided list.

Available workers:
- claim_lookup: Look up full claim details by claim ID
- claim_status: Check the processing status of a claim by claim number
- claim_payment_info: Get payment amounts and processing info for a claim by claim ID

STRICT RULES:
1. Respond with exactly one worker name from the required_workers list for the current goal.
2. If the goal cannot be completed because required information is missing
   (no claim ID, no claim number), respond with "SKIP".
3. NEVER respond with FINISH, CONTINUE, or any value not in the worker list.
4. Only use exact worker names: claim_lookup, claim_status, claim_payment_info.

Respond with JSON only — no markdown, no explanation outside the JSON:
{{"next": "worker_name_or_SKIP", "reasoning": "one sentence"}}"""

        # Planning prompt — called once at the start to decompose the query
        # into an ordered list of goals, each mapped to exactly one worker.
        self.planning_prompt = """You are a planning agent for a health insurance customer service system.
Analyze the user query and create an ordered execution plan.

User Query: {user_query}

Relevant Knowledge Base Context:
{semantic_context}

Available workers (use EXACT names only):
- claim_lookup: Looks up full claim details — requires a claim ID
- claim_status: Checks claim processing status — requires a claim number (e.g. CLM-123456)
- claim_payment_info: Retrieves payment amounts and dates — requires a claim ID

RULES:
1. Each goal must have exactly ONE worker in required_workers.
2. Use EXACT worker names: claim_lookup, claim_status, claim_payment_info.
3. If the query requires claim lookup, make it goal_1.
4. Only include goals supported by the available workers.
5. Keep the plan minimal — do not add goals for information not requested.
6. Note: claim_status uses claim NUMBER (e.g. CLM-123456); claim_lookup and
   claim_payment_info use claim ID (UUID). Include the correct identifier in
   the goal description so the worker knows which to use.

Return JSON only (no markdown fences, no explanation):
{{
    "goals": [
        {{
            "id": "goal_1",
            "description": "Short description of what to accomplish",
            "priority": 1,
            "required_workers": ["claim_lookup"]
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

                semantic_context_json = {
                    'relevant_policies': [r['document'] for r in policy_context],
                    'relevant_faqs': [r['document'] for r in faq_context]
                }
            except Exception:
                semantic_context_json = {}

            semantic_context = json.dumps(semantic_context_json, indent=2) if semantic_context_json else 'No additional context available.'

            # Create planning prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a planning agent. Create structured execution plans in JSON format."),
                ("human", self.planning_prompt)
            ])

            # Trace plan creation
            callback_handler = tracer.get_callback_handler() if tracer.enabled else None

            # Call LLM to create plan
            inputs = {
                "user_query":       user_query,
                "semantic_context": semantic_context,
            }

            if callback_handler:
                result = (prompt | self.llm).invoke(inputs, config={"callbacks": [callback_handler]})
            else:
                result = (prompt | self.llm).invoke(inputs)

            # Parse plan
            raw = result.content
            plan_text = re.sub(r"```json|```", "", str(raw)).strip()
            plan = json.loads(plan_text)

            # Store plan in CG as a team plan.
            # store_plan returns {"plan_id": ..., "step_map": {step_id: step_id}}.
            # central_step_id (from state) creates:
            #   (CentralStep)-[:DELEGATED_TO]->(TeamPlan)  [central supervisor only]
            plan_result = self.cg_manager.store_plan(
                session_id=session_id,
                plan=plan,
                agent_name=self.name,
                plan_type=state.get("plan_type", "team"),
                team_name=state.get("team_name", "claims_services"),
                central_step_id=state.get("central_step_id") or None,
            )
            plan_id  = plan_result.get("plan_id")  if plan_result else None
            step_map = plan_result.get("step_map") if plan_result else {}

            # Update state
            state["plan_id"]            = plan_id
            state["plan"]               = plan
            state["step_map"]           = step_map
            state["current_goal_index"] = 0
            state["completed_goals"]    = []

            logger.info(f"{self.name}: Created plan with {len(plan.get('goals', []))} goals")

            return state

        except Exception as e:
            logger.error(f"{self.name}: Error creating plan: {e}")

            # Fallback plan
            state["plan"] = {
                "goals": [{"id": "goal_1", "description": "Handle user query",
                           "priority": 1, "required_workers": []}],
                "steps": [{"step_id": "step_1", "goal_id": "goal_1",
                           "action": "process_query", "worker": ""}]
            }
            state["current_goal_index"] = 0
            state["completed_goals"]    = []
            state["step_map"]           = {}
            return state

    def supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """
        Orchestrator node — pure Python goal advancement + one LLM call per goal.

        Flow per invocation:
          1. Check circuit breaker.
          2. Read current_goal_index from state.
          3. If all goals done → FINISH.
          4. Determine the worker for the current goal from required_workers
             (deterministic — no LLM needed for routing if there is only one).
          5. Call LLM only to confirm worker / detect SKIP for missing data.
          6. Return next=<worker> or next=FINISH (never CONTINUE).

        Goal advancement happens in this node BEFORE returning, so the graph
        edge from worker→supervisor simply re-enters here for the next goal.
        The router never sees CONTINUE — it only sees a valid worker name,
        FINISH, or SKIP (treated as advance-and-continue).
        """
        VALID_WORKERS = {"claim_lookup", "claim_status", "claim_payment_info"}

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
        plan            = state.get("plan", {})
        plan_id         = state.get("plan_id", "")
        goals           = plan.get("goals", [])
        current_index   = state.get("current_goal_index", 0)
        completed_goals = list(state.get("completed_goals", []))
        execution_path  = list(state.get("execution_path", []))

        # ── All goals done ───────────────────────────────────────────────────
        if current_index >= len(goals):
            logger.info(f"{self.name}: All {len(goals)} goals completed → FINISH")
            try:
                self.cg_manager.complete_plan(session_id, plan_id)
            except Exception:
                pass
            return {
                "next": "FINISH",
                "current_goal_index": current_index,
                "completed_goals": completed_goals,
                "execution_path": execution_path + [f"{self.name} -> FINISH (all goals done)"],
            }

        current_goal = goals[current_index]
        goal_id      = current_goal.get("id", f"goal_{current_index}")
        required     = current_goal.get("required_workers", [])

        logger.info(
            f"{self.name}: Goal {current_index + 1}/{len(goals)}: "
            f"'{current_goal.get('description', '')}' required_workers={required}"
        )

        # ── Context Graph (best-effort) ──────────────────────────────────────
        session_context      = {}
        conversation_history = []
        execution_id         = None
        try:
            session_context      = self.cg_manager.get_session_context(session_id) or {}
            conversation_history = self.cg_manager.get_conversation_history(session_id, limit=5) or []
            execution_id         = self.cg_manager.track_agent_execution(
                session_id=session_id,
                agent_name=self.name,
                agent_type="worker",   # renamed to worker via worker_name in update_execution_status
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
            role_map = {"user": HumanMessage, "human": HumanMessage,
                        "assistant": AIMessage, "ai": AIMessage, "system": SystemMessage}
            routing_messages.append(SystemMessage(
                content=f"Last {len(conversation_history)} messages from this session:"
            ))
            for msg in reversed(conversation_history):
                cls = role_map.get(msg.get("role", "system").lower(), SystemMessage)
                routing_messages.append(cls(content=msg.get("content", "")))

        # Always inject the current goal so the LLM knows exactly what to do
        routing_messages.append(SystemMessage(content=(
            f"CURRENT GOAL ({current_index + 1}/{len(goals)}): "
            + f"{current_goal.get('description', '')}" + "\n"
            + f"Required workers for this goal: {required}" + "\n"
            + "Select ONE worker from the required_workers list, or SKIP if data is missing."
        )))

        # ── LLM routing call ─────────────────────────────────────────────────
        callback_handler = tracer.get_callback_handler() if tracer.enabled else None
        chain = self.create_routing_chain()
        try:
            llm_result  = chain.invoke(
                {"messages": routing_messages},
                config={"callbacks": [callback_handler]} if callback_handler else {},
            )
            next_worker = llm_result.get("next", "SKIP")
            reasoning   = llm_result.get("reasoning", "")
        except Exception as e:
            logger.error(f"{self.name}: LLM routing failed: {e}")
            # Fall back to first required worker if LLM fails
            next_worker = required[0] if required else "SKIP"
            reasoning   = f"LLM error fallback: {e}"

        logger.info(f"{self.name}: LLM chose '{next_worker}' — {reasoning}")

        # ── Validate LLM output ──────────────────────────────────────────────
        # If LLM returned something invalid, fall back to first required worker
        if next_worker not in VALID_WORKERS and next_worker != "SKIP":
            logger.warning(
                f"{self.name}: LLM returned invalid worker '{next_worker}', "
                f"falling back to required_workers={required}"
            )
            next_worker = required[0] if required else "SKIP"

        # ── Handle SKIP — advance goal index without calling a worker ────────
        if next_worker == "SKIP":
            logger.info(
                f"{self.name}: Skipping goal {goal_id} ({reasoning})"
            )
            completed_goals.append(goal_id)
            next_index = current_index + 1
            execution_path.append(f"{self.name} -> SKIP goal {goal_id}")

            try:
                self.cg_manager.update_plan_progress(
                    session_id=session_id, plan_id=plan_id,
                    goal_id=goal_id, goal_result="skipped",
                )
            except Exception:
                pass

            # Close the AgentExecution that was opened for this goal
            try:
                if execution_id:
                    self.cg_manager.update_execution_status(execution_id, "skipped")
            except Exception:
                pass

            # If more goals remain, re-enter supervisor for next goal
            # by returning CONTINUE — but we remap it to the next goal
            # directly here rather than looping through the graph edge.
            # We do this by checking if next index is done.
            if next_index >= len(goals):
                try:
                    self.cg_manager.complete_plan(session_id, plan_id)
                except Exception:
                    pass
                return {
                    "next": "FINISH",
                    "current_goal_index": next_index,
                    "completed_goals": completed_goals,
                    "execution_path": execution_path + [f"{self.name} -> FINISH"],
                }
            else:
                # Return CONTINUE so the graph loops back to supervisor
                # with the updated goal index
                return {
                    "next": "CONTINUE",
                    "current_goal_index": next_index,
                    "completed_goals": completed_goals,
                    "execution_path": execution_path,
                }

        # ── Valid worker — link Step->AgentExecution before worker fires ──────────
        # Uses already-resolved locals: plan_id, goal_id, plan["steps"].
        # Scoped by planId+stepId — no cross-session ambiguity possible.
        try:
            if execution_id and plan_id:
                _steps       = plan.get("steps", [])
                _lnk_step_id = next(
                    (s.get("step_id", "") for s in _steps if s.get("goal_id") == goal_id), ""
                )
                if _lnk_step_id:
                    self.cg_manager.link_step_to_execution(
                        plan_id=plan_id,
                        step_id=_lnk_step_id,
                        execution_id=execution_id,
                    )
                else:
                    logger.warning(
                        f"{self.name}: No step found for goal_id={goal_id!r} in plan steps={_steps}"
                    )
        except Exception as e:
            logger.warning(f"{self.name}: Failed to link step to execution (non-fatal): {e}")

        # ── Valid worker — return it for graph routing ────────────────────────
        execution_path.append(f"{self.name} -> {next_worker}")

        try:
            if execution_id:
                self.cg_manager.update_execution_status(
                    execution_id, "completed",
                    routing_note=f"Routing: {next_worker} — {reasoning}",
                    worker_name=f"{next_worker}_worker",
                )
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
            "current_execution_id": execution_id or "",
        }

    def _advance_goal(self, state: SupervisorState) -> SupervisorState:
        """
        Called by each worker edge back to supervisor.
        Marks the just-completed goal as done and advances current_goal_index.
        Returns updated state fields — supervisor_node picks them up next call.
        """
        plan            = state.get("plan", {})
        goals           = plan.get("goals", [])
        current_index   = state.get("current_goal_index", 0)
        completed_goals = list(state.get("completed_goals", []))
        session_id      = state.get("session_id", "default")
        plan_id         = state.get("plan_id", "")

        if current_index < len(goals):
            goal_id = goals[current_index].get("id", f"goal_{current_index}")
            completed_goals.append(goal_id)
            try:
                self.cg_manager.update_plan_progress(
                    session_id=session_id, plan_id=plan_id,
                    goal_id=goal_id, goal_result="completed",
                )
            except Exception:
                pass
            logger.info(
                f"Goal {goal_id} completed "
                f"({current_index + 1}/{len(goals)})"
            )

        return {
            "current_goal_index": current_index + 1,
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
            query = query + "\nexecution_id: " + _exec_id

            # Execute worker
            result = worker.execute(
                query=query,
                user_id=state.get("user_id", "unknown"),
                user_role=state.get("user_role", "unknown"),
                session_id=state.get("session_id", "default")
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

                # Mark the current goal as failed and cancel all remaining
                # pending goals so no nodes are left in an orphaned state.
                try:
                    plan          = state.get("plan", {})
                    goals         = plan.get("goals", [])
                    current_idx   = state.get("current_goal_index", 0)
                    plan_id_cg    = state.get("plan_id", "")
                    session_id_cg = state.get("session_id", "default")
                    if current_idx < len(goals):
                        goal_id = goals[current_idx].get("id", f"goal_{current_idx}")
                        self.cg_manager.update_plan_progress(
                            session_id=session_id_cg,
                            plan_id=plan_id_cg,
                            goal_id=goal_id,
                            goal_result="failed",
                        )
                    # Cancel every goal still pending — downstream goals that
                    # will never run because this goal failed.
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
                    → claim_lookup       → goal_advance → supervisor
                    → claim_status       → goal_advance → supervisor
                    → claim_payment_info → goal_advance → supervisor
                    → error_handler → END
                    → END  (when FINISH)
        """
        workflow = StateGraph(SupervisorState)

        workflow.add_node("create_plan",   self.create_plan_node)
        workflow.add_node("supervisor",    self.supervisor_node)
        workflow.add_node("error_handler", self.error_handler_node)
        workflow.add_node("goal_advance",  self._advance_goal)

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

        VALID_WORKERS = {"claim_lookup", "claim_status", "claim_payment_info"}

        def router(
            state: SupervisorState,
        ) -> Literal["claim_lookup", "claim_status", "claim_payment_info",
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
                "supervisor":        "supervisor",
                "claim_lookup":      "claim_lookup",
                "claim_status":      "claim_status",
                "claim_payment_info": "claim_payment_info",
                "error_handler":     "error_handler",
                "__end__":           END,
            },
        )

        workflow.set_entry_point("create_plan")
        return workflow.compile()


@lru_cache
def get_claims_services_graph():
    """Get or create claims services LangGraph."""
    claims_services_supervisor = ClaimsServicesSupervisor()

    return claims_services_supervisor.create_graph()
