"""
Central Supervisor
==================
The top-level orchestrator in the CSIP hierarchical agent system.

Architecture position:
    User → CentralSupervisor → A2AClientNode → TeamSupervisor → Workers → MCP Tools

The CentralSupervisor receives a natural-language query, builds an execution
plan decomposed into Goals and Steps, then delegates each Step to the
appropriate remote team supervisor via the A2A protocol.

Remote teams available:
    - member_services_team    → member lookup, eligibility, coverage
    - claims_services_team    → claim lookup, claim status, payment info
    - pa_services_team        → PA lookup, PA status, PA requirements
    - provider_services_team  → provider lookup, network check, specialty search
    - search_services_team    → knowledge base, medical codes, policy info

Graph topology:
    create_plan
        → supervisor
            → member_services_team   → goal_advance → supervisor
            → claims_services_team   → goal_advance → supervisor
            → pa_services_team       → goal_advance → supervisor
            → provider_services_team → goal_advance → supervisor
            → search_services_team   → goal_advance → supervisor
            → error_handler → END
            → END  (FINISH)

Plan lifecycle in Context Graph:
    (Session)-[:HAS_PLAN]->(Plan {planType:'central'})
        -[:HAS_GOAL]->(Goal)
            -[:HAS_STEP]->(Step)
                -[:DELEGATED_TO]->(TeamPlan)   # created by receiving team

CG traceability per step:
    track_agent_execution → link_step_to_execution → update_execution_status
"""

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from agents.core.a2a_agent_card import AgentCardRegistry, get_agent_card_registry
from agents.core.a2a_client_node import A2AClientNode
from agents.core.context_graph import ContextGraphManager, get_context_graph_manager
from agents.core.context_compressor import get_semantic_compressor
from agents.teams.claims_services.claims_services_a2a_agent_card import build_claims_services_agent_card
from agents.teams.member_services.member_services_a2a_agent_card import build_member_services_agent_card
from agents.teams.pa_services.pa_services_a2a_agent_card import build_pa_services_agent_card
from agents.teams.provider_services.provider_services_a2a_agent_card import build_provider_services_agent_card
from agents.teams.search_services.search_services_a2a_agent_card import build_search_services_agent_card
from agents.core.error_handling import (
    classify_error,
    create_error_record,
    format_error_for_user,
    get_error_metrics,
    is_retryable_error,
)
from agents.core.state import SupervisorState
from config.settings import get_settings
from observability.langfuse_integration import get_langfuse_tracer
from llm_providers.llm_provider_factory import LLMProviderFactory, get_factory, ChatModel

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Remote Agent Names
# These must match the `name` field in each team's A2AAgentCard.
# ---------------------------------------------------------------------------
MEMBER_SERVICES   = "member_services_team"
CLAIMS_SERVICES   = "claims_services_team"
PA_SERVICES       = "pa_services_team"
PROVIDER_SERVICES = "provider_services_team"
SEARCH_SERVICES   = "search_services_team"

VALID_REMOTE_AGENTS = {
    MEMBER_SERVICES,
    CLAIMS_SERVICES,
    PA_SERVICES,
    PROVIDER_SERVICES,
    SEARCH_SERVICES,
}

# ---------------------------------------------------------------------------
# Environment — remote agent base URLs
# Resolved once at module load; override via Docker Swarm secrets or env vars.
# ---------------------------------------------------------------------------
AGENT_URLS: Dict[str, str] = {
    MEMBER_SERVICES:   os.getenv("A2A_MEMBER_SERVICES_URL",   "https://api-gateway:8443/a2a/member"),
    CLAIMS_SERVICES:   os.getenv("A2A_CLAIMS_SERVICES_URL",   "https://api-gateway:8443/a2a/claims"),
    PA_SERVICES:       os.getenv("A2A_PA_SERVICES_URL",       "https://api-gateway:8443/a2a/pa"),
    PROVIDER_SERVICES: os.getenv("A2A_PROVIDER_SERVICES_URL", "https://api-gateway:8443/a2a/provider"),
    SEARCH_SERVICES:   os.getenv("A2A_SEARCH_SERVICES_URL",   "https://api-gateway:8443/a2a/search"),
}

# ---------------------------------------------------------------------------
# Prompt: Planning
# ---------------------------------------------------------------------------
PLANNING_SYSTEM_PROMPT = """You are the central planner for a health insurance AI system.

Your job is to decompose the user's query into a structured execution plan.
The plan will be executed step-by-step by delegating to remote specialist agents.

=== AVAILABLE REMOTE AGENTS ===
{agent_skills_summary}

=== PLAN STRUCTURE ===
Return ONLY a valid JSON object with this exact structure:

{{
  "goals": [
    {{
      "id": "goal_1",
      "description": "One-sentence description of this goal",
      "priority": 1
    }}
  ],
  "steps": [
    {{
      "step_id": "step_1",
      "goal_id": "goal_1",
      "agent": "<agent_name>",
      "instruction": "Internal planner note: what this step accomplishes",
      "query": "Natural-language question sent directly to the remote agent",
      "order": 1
    }}
  ]
}}

=== STRICT PLANNING RULES ===

1. DECOMPOSE thoughtfully. Each goal represents a distinct information need.
   Each step is a single delegation to exactly one remote agent.

2. ASSIGN CORRECTLY. The "agent" field must be one of:
   - member_services_team
   - claims_services_team
   - pa_services_team
   - provider_services_team
   - search_services_team

3. ONE STEP PER AGENT PER GOAL. If a goal requires two different agents,
   create two separate goals — one per agent.

4. ORDER MATTERS. Use the "order" field to sequence steps that depend on
   each other. Steps with the same order may execute in any sequence.

5. MINIMUM STEPS. Do not create steps for information you already have or
   that was not requested. Every step must be necessary to answer the query.

6. MEMBER CONTEXT FIRST. If the query references member ID, eligibility,
   or coverage and ALSO requires another team, place member_services_team
   steps at order 1 so downstream steps have member context.

7. SEARCH AUGMENTS, NOT REPLACES. Use search_services_team to look up
   policy rules, clinical guidelines, or medical codes — not to answer
   questions that a specific team (PA, claims, provider) can answer directly.

8. PARALLEL WHEN INDEPENDENT. Assign the same "order" value to steps that
   do not depend on each other's results so they can be dispatched together.

9. NO SPECULATION. Only create steps for information explicitly requested
   or clearly implied by the query. Do not add "helpful extras".

10. QUERY vs INSTRUCTION. Every step must have both fields:
    - "instruction": your internal planner note describing what this step does.
      Written for the plan record — terse, structured, may reference step IDs.
    - "query": the natural-language question or request sent directly to the
      remote agent as if the user asked it. Must be self-contained and specific:
      include all relevant identifiers (member ID, claim number, PA ID, procedure
      code, plan type, ZIP code, etc.) so the remote agent needs no other context.
      Do NOT include plan metadata (step_id, goal_id, order) in the query.

Return ONLY the JSON. No explanation, no markdown, no preamble.
"""

PLANNING_USER_PROMPT = """User query: {query}

Build the execution plan now."""


# ---------------------------------------------------------------------------
# Prompt: Supervisor Routing
# ---------------------------------------------------------------------------
SUPERVISOR_SYSTEM_PROMPT = """You are the central supervisor routing decisions engine.

You will be given the current execution plan and the next step to execute.
Your ONLY job is to confirm which remote agent should handle the next step.

=== STRICT ROUTING RULES ===

1. READ the "agent" field of the current step. That is the pre-assigned agent.

2. CONFIRM by outputting ONLY the agent name if the assignment is correct:
   - member_services_team
   - claims_services_team
   - pa_services_team
   - provider_services_team
   - search_services_team

3. OUTPUT "SKIP" if and only if:
   - The step instruction is empty or nonsensical
   - The assigned agent does not match any known agent name
   - The goal was already completed in a previous step

4. NEVER output FINISH. Step completion and plan termination are handled
   by the system — not by you.

5. NEVER output CONTINUE. Either confirm the agent name or output SKIP.

6. ONE WORD ONLY. Your entire response must be exactly one token:
   either a valid agent name or the word SKIP.

Current step:
Agent:       {agent}
Instruction: {instruction}
Goal:        {goal_description}
"""


# ---------------------------------------------------------------------------
# Central Supervisor
# ---------------------------------------------------------------------------
class CentralSupervisor:
    """
    Orchestrates the full CSIP agent hierarchy.

    Responsibilities:
        1. Build a Plan (Goals + Steps) from the user query via one LLM call.
        2. Store the Plan in the Context Graph.
        3. Execute Steps one at a time by delegating to remote team supervisors
           via A2AClientNode.
        4. Advance Goals and finalize the Plan in the CG on completion.
        5. Handle and record all errors at plan, goal, and step level.

    The CentralSupervisor never calls MCP tools directly.
    All tool execution happens inside the remote team supervisors.
    """

    def __init__(self, registry: Optional[AgentCardRegistry] = None):
        _llm_factory: LLMProviderFactory = get_factory()
        self._llm: ChatModel = _llm_factory.get_llm_provider()
        self._registry = registry or get_agent_card_registry()
        self._cg: ContextGraphManager = get_context_graph_manager()
        self._tracer = get_langfuse_tracer()

        # Build one A2AClientNode per remote agent, keyed by agent name.
        # Each node handles encryption, mTLS, and CG tracking for its agent.
        # AGENT_URLS already contains the full base path (e.g. /a2a/member).
        # A2AClientNode appends /tasks/send internally, so no suffix needed here.
        self._agent_nodes: Dict[str, A2AClientNode] = {
            agent_name: A2AClientNode(
                agent_name=agent_name,
                agent_url=AGENT_URLS[agent_name],
                schema_registry={},         # Central supervisor has no tool schemas;
                                            # schema validation is per-team.
                from_agent_name="central_supervisor",
            )
            for agent_name in VALID_REMOTE_AGENTS
        }

        # Populate the AgentCardRegistry so the planning prompt contains
        # accurate skill descriptions for every remote team.
        #
        # Strategy (per agent):
        #   1. Register the well-known URL for future on-demand refresh.
        #   2. Attempt a live HTTP fetch of the card from the remote agent.
        #   3. If the fetch fails (agent not yet reachable at startup),
        #      fall back to the locally-built card so the planner always
        #      has a description to work with.
        self._populate_registry()

        logger.info(
            "CentralSupervisor initialised with %d remote agents: %s",
            len(self._agent_nodes),
            ", ".join(self._agent_nodes),
        )

    def _populate_registry(self) -> None:
        """
        Populate the AgentCardRegistry with cards for all five remote teams.

        For each team:
          - Register the base URL so the registry can refresh the card later.
          - Try a synchronous HTTP fetch of the live card from the remote agent.
          - On any failure, register the locally-built fallback card instead.

        This is called once from __init__ and ensures the planning prompt
        always contains accurate (or at minimum reasonable) skill descriptions
        regardless of whether the remote agents are reachable at startup.
        """
        # Map each agent name to its local card builder function.
        # The builder accepts the base URL and returns an A2AAgentCard.
        local_card_builders = {
            MEMBER_SERVICES:   build_member_services_agent_card,
            CLAIMS_SERVICES:   build_claims_services_agent_card,
            PA_SERVICES:       build_pa_services_agent_card,
            PROVIDER_SERVICES: build_provider_services_agent_card,
            SEARCH_SERVICES:   build_search_services_agent_card,
        }

        for agent_name, base_url in AGENT_URLS.items():
            # Step 1 — register the well-known URL for future refreshes
            self._registry.register_card_url(agent_name, base_url)

            # Step 2 — attempt a live fetch from /.well-known/agent.json
            card = None
            try:
                import httpx
                well_known_url = f"{base_url}/.well-known/agent.json"
                with httpx.Client(verify=False, timeout=5.0) as client:
                    response = client.get(well_known_url)
                    response.raise_for_status()
                    data = response.json()

                from agents.core.a2a_agent_card import (
                    A2AAgentCard, A2AAuthentication, A2ACapabilities, A2ASkill
                )
                card = A2AAgentCard(
                    name=data["name"],
                    description=data["description"],
                    url=data["url"],
                    version=data.get("version", "1.0"),
                    protocol_version=data.get("protocolVersion", "0.2.2"),
                    capabilities=A2ACapabilities(
                        streaming=data.get("capabilities", {}).get("streaming", False),
                        push_notifications=data.get("capabilities", {}).get("pushNotifications", False),
                        state_transition_history=data.get("capabilities", {}).get("stateTransitionHistory", True),
                    ),
                    authentication=A2AAuthentication(
                        schemes=data.get("authentication", {}).get("schemes", ["hmac"]),
                    ),
                    skills=[
                        A2ASkill(
                            id=s["id"],
                            name=s["name"],
                            description=s["description"],
                            tags=s.get("tags", []),
                            examples=s.get("examples", []),
                        )
                        for s in data.get("skills", [])
                    ],
                )
                logger.info("CentralSupervisor: fetched live agent card for %s", agent_name)

            except Exception as exc:
                # Step 3 — remote agent unreachable; use locally-built fallback card
                logger.warning(
                    "CentralSupervisor: could not fetch agent card for %s (%s) "
                    "— registering local fallback card",
                    agent_name, exc,
                )
                builder = local_card_builders.get(agent_name)
                if builder:
                    card = builder(base_url)

            if card:
                self._registry.register_card(agent_name, card)

    # -----------------------------------------------------------------------
    # Routing chain (reused by supervisor_node on every step)
    # -----------------------------------------------------------------------
    def _create_routing_chain(self):
        """
        Build the supervisor routing chain.

        Chain: ChatPromptTemplate | llm | StrOutputParser

        SUPERVISOR_SYSTEM_PROMPT instructs the LLM to respond with ONE WORD
        only — a valid agent name or "SKIP". StrOutputParser extracts the
        plain text response directly; JsonOutputParser is intentionally not
        used here because the output is not JSON.
        """
        from langchain_core.output_parsers import StrOutputParser
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("human",  "{messages}"),
        ])
        return prompt | self._llm | StrOutputParser()

    # -----------------------------------------------------------------------
    # Node: create_plan
    # -----------------------------------------------------------------------
    def create_plan_node(self, state: SupervisorState) -> SupervisorState:
        """
        Phase 1 — Planning.

        Calls the LLM once to decompose the user query into a structured
        Plan (Goals + Steps). Stores the Plan in the Context Graph.

        Returns state with:
            plan, plan_id, step_map, current_step_index,
            completed_goals, start_time, plan_type, team_name
        """
        session_id = state.get("session_id", "default")
        user_id    = state.get("user_id", "unknown")
        messages   = state.get("messages", [])

        # Extract the user's query from the last message
        query = ""
        if messages:
            last = messages[-1]
            query = last.content if hasattr(last, "content") else str(last)

        logger.info("CentralSupervisor: building plan for session=%s", session_id)

        # Build a skills summary so the planner knows what each agent can do
        skills_summary = self._registry.get_skills_summary()
        if not skills_summary.strip():
            logger.warning("AgentCardRegistry has no registered cards — using fallback summary")
            skills_summary = _fallback_skills_summary()

        # Compress the skills summary before injecting into the planning prompt.
        # Agent cards can reach 600-800 tokens when all five teams are live;
        # gentle compression keeps team-skill mappings intact while reducing
        # token overhead. Non-fatal — falls back to uncompressed on any error.
        try:
            _sem_compressor = get_semantic_compressor()
            skills_summary = _sem_compressor.compress_text(skills_summary)
        except Exception as _exc:
            logger.warning(
                "CentralSupervisor: skills summary compression failed (non-fatal): %s", _exc
            )

        planning_prompt = PLANNING_SYSTEM_PROMPT.format(
            agent_skills_summary=skills_summary
        )
        user_prompt = PLANNING_USER_PROMPT.format(query=query)

        # Build the planning chain: llm | JsonOutputParser.
        # NOTE: ChatPromptTemplate is intentionally NOT used here.
        # planning_prompt is already fully .format()-ted — all {{ }} escapes
        # have resolved into literal { } JSON braces. Passing through
        # ChatPromptTemplate causes LangChain to re-interpret those braces as
        # template variables, crashing with INVALID_PROMPT_INPUT.
        # We pass SystemMessage + HumanMessage directly so the already-rendered
        # strings reach the LLM unchanged.
        planning_chain = self._llm | JsonOutputParser()

        # Obtain a Langfuse callback handler for this invocation.
        # Called fresh per node (not cached) so each LLM call gets its own trace span.
        tracer = get_langfuse_tracer()
        callback_handler = tracer.get_callback_handler() if tracer.enabled else None

        # Open an AgentExecution node for the planning LLM call.
        # Named "central_supervisor_planner" so it is distinguishable in the CG
        # from routing executions (which are named "central_supervisor").
        planning_execution_id: Optional[str] = None
        try:
            planning_execution_id = self._cg.track_agent_execution(
                session_id=session_id,
                agent_name="central_supervisor_planner",
                agent_type="supervisor",
                status="running",
                metadata={"user_id": user_id},
            )
            # DAL only creates (Session)-[:HAS_EXECUTION] for a2a_client type.
            # Explicitly link the planner node to the Session so it is
            # reachable from the root:
            #   (Session)-[:HAS_EXECUTION]->(central_supervisor_planner)
            if planning_execution_id:
                self._cg.link_session_to_execution(session_id, planning_execution_id)
        except Exception as exc:
            logger.warning("CentralSupervisor: CG track planning execution (non-fatal): %s", exc)

        plan: Optional[Dict[str, Any]] = None
        try:
            plan = planning_chain.invoke(
                [SystemMessage(content=planning_prompt),
                 HumanMessage(content=user_prompt)],
                config={"callbacks": [callback_handler]} if callback_handler else {},
            )
            logger.info(
                "CentralSupervisor: plan created — %d goals, %d steps",
                len(plan.get("goals", [])),
                len(plan.get("steps", [])),
            )
        except Exception as exc:
            logger.error("CentralSupervisor: planning LLM failed: %s", exc)

            # Record the failure against the planning AgentExecution in CG
            if planning_execution_id:
                try:
                    self._cg.create_agent_error(
                        execution_id=planning_execution_id,
                        error_type=classify_error(str(exc)),
                        error_message=str(exc),
                    )
                    self._cg.update_execution_status(
                        execution_id=planning_execution_id,
                        status="failed",
                        error_message=str(exc),
                    )
                except Exception as cg_exc:
                    logger.warning("CentralSupervisor: CG plan error record (non-fatal): %s", cg_exc)

            # Propagate current_execution_id so error_handler_node can find the node
            return {
                "error": f"Planning failed: {exc}",
                "current_execution_id": planning_execution_id or "",
                "error_count": state.get("error_count", 0) + 1,
                "error_history": state.get("error_history", []) + [
                    create_error_record("central_supervisor_planner", str(exc))
                ],
                "execution_path": state.get("execution_path", []) + ["create_plan_failed"],
                "start_time": datetime.now(timezone.utc).isoformat(),
                "plan_type": "central",
                "team_name": "",
            }

        # Close the planning execution as completed
        if planning_execution_id:
            try:
                self._cg.update_execution_status(
                    execution_id=planning_execution_id,
                    status="completed",
                )
            except Exception as exc:
                logger.warning("CentralSupervisor: CG close planning execution (non-fatal): %s", exc)

        # Store Plan in Context Graph
        plan_id:  Optional[str] = None
        step_map: Dict[str, str] = {}
        try:
            cg_result = self._cg.store_plan(
                session_id=session_id,
                plan=plan,
                agent_name="central_supervisor",
                plan_type="central",
                team_name="",
            )
            if cg_result:
                plan_id  = cg_result.get("plan_id")
                step_map = cg_result.get("step_map", {})
                logger.info("CentralSupervisor: plan stored in CG — plan_id=%s", plan_id)
                # Link the planner AgentExecution to the CentralPlan so it is
                # reachable via:
                #   Session -[HAS_EXECUTION]→ planner AE -[HAS_PLAN]→ CentralPlan
                # This mirrors how team plans are linked:
                #   a2a_server AE -[HAS_PLAN]→ TeamPlan
                # Session-[:HAS_PLAN]→Plan is intentionally NOT created.
                if plan_id and planning_execution_id:
                    self._cg.link_planner_to_plan(planning_execution_id, plan_id)
        except Exception as exc:
            logger.warning("CentralSupervisor: CG store_plan failed (non-fatal): %s", exc)

        return {
            "plan":                plan,
            "plan_id":             plan_id,
            "step_map":            step_map,
            "current_step_index":  0,
            "completed_goals":     [],
            "execution_path":      state.get("execution_path", []) + ["create_plan"],
            "start_time":          datetime.now(timezone.utc).isoformat(),
            "plan_type":           "central",
            "team_name":           "",
        }

    # -----------------------------------------------------------------------
    # Node: supervisor
    # -----------------------------------------------------------------------
    def supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """
        Phase 2 — Routing.

        Reads the next unexecuted Step from the plan, asks the LLM to confirm
        the pre-assigned remote agent (or SKIP), and records the routing
        decision in the Context Graph.

        Returns state with:
            next, current_execution_id
        """
        session_id = state.get("session_id", "default")
        plan       = state.get("plan") or {}
        steps      = sorted(plan.get("steps", []), key=lambda s: s.get("order", 1))
        goals      = plan.get("goals", [])
        idx        = state.get("current_step_index", 0)

        # All steps done → compose combined response and signal FINISH
        if idx >= len(steps):
            logger.info("CentralSupervisor: all steps complete → FINISH")

            # Synthesise a combined response from all accumulated step outputs.
            # Without this, _extract_response() in request_processor returns only
            # the last AIMessage — which contains only the last step's answer.
            # For multi-step plans each step's answer must be included.
            tool_results = state.get("tool_results", {})
            step_outputs = [
                v.get("output", "")
                for v in tool_results.values()
                if isinstance(v, dict) and v.get("output", "").strip()
            ]
            if step_outputs:
                combined_response = "\n\n".join(step_outputs)
                return {
                    "next":     "FINISH",
                    "messages": [AIMessage(content=combined_response)],
                }
            return {"next": "FINISH"}

        current_step = steps[idx]
        agent_name   = current_step.get("agent", "")
        instruction  = current_step.get("instruction", "")
        goal_id      = current_step.get("goal_id", "")
        goal         = next((g for g in goals if g.get("id") == goal_id), {})
        goal_desc    = goal.get("description", "")

        # Track this routing decision in the CG.
        # The agent_name encodes the delegation target so each AgentExecution
        # node in the CG is uniquely identifiable, e.g.:
        #   "central_supervisor -> member_services_team"
        execution_id: Optional[str] = None
        try:
            execution_id = self._cg.track_agent_execution(
                session_id=session_id,
                agent_name=f"central_supervisor -> {agent_name}",
                agent_type="supervisor",
                status="running",
                metadata={"step_index": idx, "assigned_agent": agent_name},
            )
        except Exception as exc:
            logger.warning("CentralSupervisor: CG track_agent_execution failed (non-fatal): %s", exc)

        # Ask the LLM to confirm the routing or SKIP.
        # Uses the routing chain (prompt | llm | JsonOutputParser) so the
        # response is already a parsed dict — no manual JSON handling needed.
        # Langfuse callback is obtained fresh each invocation for its own span.
        tracer = get_langfuse_tracer()
        callback_handler = tracer.get_callback_handler() if tracer.enabled else None

        routing_response = "SKIP"
        try:
            routing_chain = self._create_routing_chain()
            # routing_chain returns a plain string (StrOutputParser) —
            # strip whitespace and take the first word to guard against
            # models that add punctuation or a brief explanation.
            raw = routing_chain.invoke(
                {
                    "agent":            agent_name,
                    "instruction":      instruction,
                    "goal_description": goal_desc,
                    "messages":         "Confirm the routing decision now.",
                },
                config={"callbacks": [callback_handler]} if callback_handler else {},
            )
            routing_response = raw.strip().split()[0] if raw.strip() else "SKIP"
        except Exception as exc:
            logger.error("CentralSupervisor: supervisor LLM failed: %s", exc)
            routing_response = "SKIP"

        logger.info(
            "CentralSupervisor: step[%d] agent=%s → routing=%s",
            idx, agent_name, routing_response,
        )

        # Validate the LLM's output — guard against unexpected values
        if routing_response not in VALID_REMOTE_AGENTS and routing_response != "SKIP":
            logger.warning(
                "CentralSupervisor: unexpected routing value '%s' → forcing SKIP",
                routing_response,
            )
            routing_response = "SKIP"

        # Update CG with the confirmed routing decision
        if execution_id:
            try:
                self._cg.update_execution_status(
                    execution_id=execution_id,
                    status="completed" if routing_response != "SKIP" else "skipped",
                    routing_note=f"Routing: {routing_response} — {goal_desc}",
                    worker_name=routing_response if routing_response != "SKIP" else None,
                )
                # Link this routing execution to the step in the CG
                plan_id  = state.get("plan_id")
                step_id  = current_step.get("step_id", "")
                mapped   = state.get("step_map", {}).get(step_id, step_id)
                if plan_id and mapped:
                    self._cg.link_step_to_execution(plan_id, mapped, execution_id)
            except Exception as exc:
                logger.warning("CentralSupervisor: CG update routing (non-fatal): %s", exc)

        return {
            "next":                  routing_response,
            "current_execution_id":  execution_id or "",
            # Signal goal_advance_node that this step was skipped so it can
            # mark the goal as "skipped" in the CG rather than leaving it pending.
            "last_step_was_skipped": routing_response == "SKIP",
        }

    # -----------------------------------------------------------------------
    # Node: agent delegation (one per remote agent)
    # -----------------------------------------------------------------------
    def _build_agent_node(self, agent_name: str):
        """
        Return a LangGraph node function that delegates to the named remote agent.

        The node:
            1. Injects the current step's instruction as the last message
               so A2AClientNode picks it up as the query.
            2. Calls A2AClientNode.__call__(state) which handles encryption,
               HTTP, decryption, and CG linking internally.
            3. Appends the agent name to the execution path.
        """
        a2a_node = self._agent_nodes[agent_name]

        def agent_node(state: SupervisorState) -> SupervisorState:
            plan  = state.get("plan") or {}
            steps = sorted(plan.get("steps", []), key=lambda s: s.get("order", 1))
            idx   = state.get("current_step_index", 0)

            # Build the delegation query for the remote agent.
            #
            # "query"       — user-intent-specific natural-language question
            #                 composed by the planner for this exact step.
            #                 Self-contained: includes all identifiers the
            #                 remote agent needs (member ID, claim number, etc.)
            #                 and is phrased as a direct user request.
            #
            # "instruction" — internal planner note; used as fallback only
            #                 when "query" is absent (e.g. plans produced by
            #                 an older prompt version).
            if idx < len(steps):
                current_step = steps[idx]
                delegation_query = (
                    current_step.get("query")
                    or current_step.get("instruction", "")
                )
                injected_state = dict(state)
                existing_messages = list(state.get("messages", []))

                # Preserve all messages except the last; append the delegation query
                if existing_messages:
                    injected_state["messages"] = existing_messages[:-1] + [
                        HumanMessage(content=delegation_query)
                    ]
                else:
                    injected_state["messages"] = [HumanMessage(content=delegation_query)]
            else:
                injected_state = state

            logger.info(
                "CentralSupervisor: delegating step[%d] to %s",
                idx, agent_name,
            )

            # Delegate to the remote agent via A2A
            result = a2a_node(injected_state)

            # ── Create the missing CG edge ────────────────────────────────
            # (routing AgentExecution)-[:CALLED_AGENT]->(a2a_client AgentExecution)
            #
            # A2AClientNode now returns a2a_task_id in state. Combined with
            # current_execution_id (the routing AgentExecution's ID stored by
            # supervisor_node), we can join the two chains that were previously
            # disconnected in the CG:
            #
            #   (CentralStep)
            #     -[:EXECUTED_BY]->(routing AgentExecution)        ← linked in supervisor_node
            #       -[:CALLED_AGENT]->(a2a_client AgentExecution)  ← linked HERE
            #         -[:CALLED_AGENT]->(a2a_server AgentExecution) ← linked by A2AClientNode
            #           -[:HAS_PLAN]->(TeamPlan)                   ← linked by A2A server
            routing_execution_id = state.get("current_execution_id", "")
            a2a_task_id          = result.get("a2a_task_id", "")
            if routing_execution_id and a2a_task_id:
                try:
                    self._cg.link_supervisor_to_a2a_client(
                        routing_execution_id=routing_execution_id,
                        a2a_task_id=a2a_task_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "CentralSupervisor: link_supervisor_to_a2a_client failed (non-fatal): %s", exc
                    )

            # Append this delegation to the execution path
            result["execution_path"] = (
                state.get("execution_path", [])
                + result.get("execution_path", [])
            )

            return result

        agent_node.__name__ = f"{agent_name}_node"
        return agent_node

    # -----------------------------------------------------------------------
    # Node: goal_advance
    # -----------------------------------------------------------------------
    def goal_advance_node(self, state: SupervisorState) -> SupervisorState:
        """
        Advance the plan after a step completes successfully.

        Responsibilities:
            - Mark the current Goal as completed in the CG if all its
              steps are done.
            - Increment current_step_index to move to the next step.
            - Mark the overall Plan as complete in the CG when all goals
              are done.

        Returns state with updated current_step_index and completed_goals.
        """
        session_id      = state.get("session_id", "default")
        plan            = state.get("plan") or {}
        plan_id         = state.get("plan_id")
        steps           = sorted(plan.get("steps", []), key=lambda s: s.get("order", 1))
        goals           = plan.get("goals", [])
        idx             = state.get("current_step_index", 0)
        completed_goals = list(state.get("completed_goals", []))

        if idx < len(steps):
            current_step      = steps[idx]
            goal_id           = current_step.get("goal_id", "")
            step_was_skipped  = state.get("last_step_was_skipped", False)

            # Check if all steps for this goal are now done
            goal_step_indices = [
                i for i, s in enumerate(steps) if s.get("goal_id") == goal_id
            ]
            all_goal_steps_done = all(i <= idx for i in goal_step_indices)

            if all_goal_steps_done and goal_id not in completed_goals:
                completed_goals.append(goal_id)

                # A goal is "skipped" when its last step was skipped (SKIP routing).
                # A goal is "completed" when its last step executed successfully.
                goal_result = "skipped" if step_was_skipped else "completed"
                logger.info("CentralSupervisor: goal %s → %s", goal_id, goal_result)

                if plan_id:
                    try:
                        self._cg.update_plan_progress(
                            session_id=session_id,
                            plan_id=plan_id,
                            goal_id=goal_id,
                            goal_result=goal_result,
                        )
                    except Exception as exc:
                        logger.warning("CentralSupervisor: CG update_plan_progress (non-fatal): %s", exc)

        next_idx = idx + 1

        # All steps complete → mark plan as done in CG
        if next_idx >= len(steps) and plan_id:
            try:
                self._cg.complete_plan(session_id=session_id, plan_id=plan_id)
                logger.info("CentralSupervisor: plan %s marked complete in CG", plan_id)
            except Exception as exc:
                logger.warning("CentralSupervisor: CG complete_plan (non-fatal): %s", exc)

        return {
            "current_step_index": next_idx,
            "completed_goals":    completed_goals,
            "execution_path":     state.get("execution_path", []) + ["goal_advance"],
        }

    # -----------------------------------------------------------------------
    # Node: error_handler
    # -----------------------------------------------------------------------
    def error_handler_node(self, state: SupervisorState) -> SupervisorState:
        """
        Handle an unrecoverable error in the plan execution.

        Responsibilities:
            - Record the error in the CG against the current AgentExecution.
            - Cancel all remaining Goals in the CG.
            - Mark the Plan as failed in the CG.
            - Compose a user-friendly error message as the final response.
        """
        session_id   = state.get("session_id", "default")
        plan_id      = state.get("plan_id")
        execution_id = state.get("current_execution_id", "")
        error_msg    = state.get("error", "Unknown error")
        error_count  = state.get("error_count", 1)

        logger.error(
            "CentralSupervisor: error_handler invoked — %s (plan_id=%s)",
            error_msg, plan_id,
        )

        # Record agent-level error in CG
        if execution_id:
            try:
                self._cg.create_agent_error(
                    execution_id=execution_id,
                    error_type=classify_error(error_msg),
                    error_message=error_msg,
                )
            except Exception as exc:
                logger.warning("CentralSupervisor: CG create_agent_error (non-fatal): %s", exc)

        # Cancel remaining goals and fail the plan
        if plan_id:
            try:
                cancelled = self._cg.cancel_remaining_goals(
                    session_id=session_id, plan_id=plan_id
                )
                self._cg.fail_plan(session_id=session_id, plan_id=plan_id)
                logger.info(
                    "CentralSupervisor: plan %s failed — %d goals cancelled",
                    plan_id, cancelled,
                )
            except Exception as exc:
                logger.warning("CentralSupervisor: CG fail_plan (non-fatal): %s", exc)

        # Compose a user-facing error message
        user_message = format_error_for_user(error_msg)
        end_time     = datetime.now(timezone.utc).isoformat()

        # Prometheus
        try:
            get_error_metrics().record_error(
                worker="central_supervisor",
                error_type=classify_error(error_msg),
                is_retryable=is_retryable_error(error_msg),
            )
        except Exception:
            pass

        return {
            "messages":      [AIMessage(content=user_message)],
            "execution_path": state.get("execution_path", []) + ["error_handler"],
            "end_time":       end_time,
            "is_recoverable": is_retryable_error(error_msg),
            "fallback_used":  True,
        }

    # -----------------------------------------------------------------------
    # Router: determines next node after supervisor_node
    # -----------------------------------------------------------------------
    def router(self, state: SupervisorState) -> str:
        """
        Route to the appropriate remote agent node, SKIP (advance without
        delegating), or FINISH (end the graph).

        Called by LangGraph after supervisor_node runs.
        """
        next_node = state.get("next", "FINISH")

        if next_node == "FINISH":
            return "FINISH"

        if next_node == "SKIP":
            # Treat SKIP as a no-op advance — bump the step index and re-enter supervisor
            return "goal_advance"

        if next_node in VALID_REMOTE_AGENTS:
            return next_node

        # Unexpected value — treat as error
        logger.warning("CentralSupervisor: unknown routing target '%s' → error", next_node)
        return "error_handler"

    # -----------------------------------------------------------------------
    # Error router: determines next node after a remote agent returns an error
    # -----------------------------------------------------------------------
    def error_router(self, state: SupervisorState) -> str:
        """
        Called after every remote agent node. Routes to error_handler if the
        agent returned an error, otherwise to goal_advance.
        """
        if state.get("error"):
            logger.warning(
                "CentralSupervisor: agent returned error — %s", state.get("error")
            )
            return "error_handler"
        return "goal_advance"

    # -----------------------------------------------------------------------
    # Graph assembly
    # -----------------------------------------------------------------------
    def create_graph(self) -> StateGraph:
        """
        Assemble and compile the LangGraph StateGraph.

        Nodes:
            create_plan          — decomposes query into Goals + Steps
            supervisor           — confirms routing for each step
            <agent_name>         — one node per remote team (5 total)
            goal_advance         — marks goals complete, increments step index
            error_handler        — records failure and terminates the plan

        Edges:
            create_plan → supervisor
            supervisor  → [agent nodes | goal_advance (SKIP) | END (FINISH) | error_handler]
            agent_node  → [goal_advance | error_handler]
            goal_advance → supervisor
            error_handler → END
        """
        graph = StateGraph(SupervisorState)

        # ── Add nodes ──────────────────────────────────────────────────────
        graph.add_node("create_plan",   self.create_plan_node)
        graph.add_node("supervisor",    self.supervisor_node)
        graph.add_node("goal_advance",  self.goal_advance_node)
        graph.add_node("error_handler", self.error_handler_node)

        # One delegation node per remote agent
        for agent_name in VALID_REMOTE_AGENTS:
            graph.add_node(agent_name, self._build_agent_node(agent_name))

        # ── Entry point ────────────────────────────────────────────────────
        graph.set_entry_point("create_plan")

        # ── Fixed edges ────────────────────────────────────────────────────
        graph.add_edge("create_plan",  "supervisor")
        graph.add_edge("goal_advance", "supervisor")
        graph.add_edge("error_handler", END)

        # ── Conditional edge: supervisor → agent | goal_advance | FINISH | error ──
        graph.add_conditional_edges(
            "supervisor",
            self.router,
            {
                **{agent: agent for agent in VALID_REMOTE_AGENTS},
                "goal_advance":  "goal_advance",
                "error_handler": "error_handler",
                "FINISH":        END,
            },
        )

        # ── Conditional edges: each agent node → goal_advance | error_handler ──
        for agent_name in VALID_REMOTE_AGENTS:
            graph.add_conditional_edges(
                agent_name,
                self.error_router,
                {
                    "goal_advance":  "goal_advance",
                    "error_handler": "error_handler",
                },
            )

        return graph.compile()


# ---------------------------------------------------------------------------
# Singleton graph factory
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_central_supervisor_graph():
    """
    Build and cache the compiled CentralSupervisor graph.

    Called once at application startup (or on first use).
    The AgentCardRegistry must be populated before this is called so that
    the planning prompt contains accurate agent skill descriptions.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke()
    """
    supervisor = CentralSupervisor()
    return supervisor.create_graph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fallback_skills_summary() -> str:
    """
    Return a hard-coded skills summary when the AgentCardRegistry is empty.

    This ensures the planner always has a minimum description of each team
    even if the remote agents are not yet reachable at startup.
    """
    return """
Agent: member_services_team
  Description: Member lookup, eligibility verification, and coverage/benefits inquiries.
  Skills:
    - member_lookup: Look up member demographics and contact details by member ID.
    - eligibility_check: Verify active coverage status for a service date.
    - coverage_lookup: Retrieve deductibles, copays, and benefits detail.

Agent: claims_services_team
  Description: Claim lookup, status checks, and payment/financial information.
  Skills:
    - claim_lookup: Look up full claim details by claim ID.
    - claim_status: Check processing status of a claim by claim number.
    - claim_payment_info: Retrieve payment and EOB information for a claim.

Agent: pa_services_team
  Description: Prior authorization lookup, status, and requirements.
  Skills:
    - pa_lookup: Look up full PA details by PA ID.
    - pa_status: Check current status of a prior authorization.
    - pa_requirements: Determine if a procedure requires PA under a given policy type.

Agent: provider_services_team
  Description: Provider lookup, network status, and specialty search.
  Skills:
    - provider_lookup: Look up provider details by provider ID or NPI.
    - provider_network_check: Check whether a provider is in-network for a policy.
    - provider_search_by_specialty: Search for providers by specialty and ZIP code.

Agent: search_services_team
  Description: Semantic search over knowledge base, medical codes, and policy documents.
  Skills:
    - search_knowledge_base: Search FAQs, clinical guidelines, and regulations.
    - search_medical_codes: Search CPT procedure codes and ICD-10 diagnosis codes.
    - search_policy_info: Search policy documents for plan-specific financial details.
"""
