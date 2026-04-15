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
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import JsonOutputParser
# Plan schema validation — catches malformed plans before CG writes
from agents.core.plan_schema import validate_central_plan
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
# Prometheus counters for experience-augmented planning hit rate.
# Read by the evaluation pipeline to compute csip_experience_hit_rate.
try:
    from prometheus_client import Counter as _PromCounter
    _experience_planning_total = _PromCounter(
        "csip_experience_planning_total",
        "Planning calls with experience store enabled",
    )
    _experience_planning_hits = _PromCounter(
        "csip_experience_planning_hits",
        "Planning calls where at least one experience was retrieved",
    )
except Exception:
    _experience_planning_total = None
    _experience_planning_hits = None
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
=== SIMILAR PAST SUCCESSFUL PLANS ===
{past_successful_plans}
Use the past successful plans above (if any) as reference patterns for
similar queries.  Adapt the structure to the current query — do not copy
blindly.  If no past plans are shown, plan from first principles using
the rules below.
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
6. MEMBER CONTEXT FIRST — but only for member-specific information.
   If the query needs member demographics, eligibility, or coverage AND also
   requires another team, place member_services_team at order 1.
   However, if the query asks for claims or prior authorizations belonging
   to a member, route directly to the team that owns that data — see Rule 11.
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
11. CROSS-ENTITY ROUTING. Route by the PRIMARY data entity, not the identifier type:
    - "claims for member X" or "claims associated with member X"
      → claims_services_team (has member_claims tool)
    - "prior authorizations for member X" or "PAs for member X"
      → pa_services_team (has member_prior_authorizations tool)
    - "member details" or "look up member X" or "eligibility for member X"
      → member_services_team
    The presence of a member ID does NOT mean member_services_team owns the query.
    Match the NOUN being requested (claims, PAs, member info), not the identifier.
12. DECISION AGENT QUERIES — adjudication and recommendation requests require
    multi-team evidence gathering BEFORE the decision step.
    When the query asks to "adjudicate", "validate", or "evaluate" a claim:
      a. Create evidence-gathering steps at order 1, each routed to its owning team:
         - claim_lookup → claims_services_team (get claim details)
         - check_eligibility → member_services_team (verify member is eligible on service date)
         - provider_network_check → provider_services_team (verify provider is in-network)
      b. Create the decision step at order 2, routed to claims_services_team:
         - claim_adjudication → claims_services_team
    When the query asks to "recommend" a PA decision:
      a. Create evidence-gathering steps at order 1:
         - pa_lookup → pa_services_team
         - pa_requirements → pa_services_team
         - search_knowledge_base → search_services_team (clinical guidelines)
         - treatment_history → member_services_team
      b. Create the decision step at order 2, routed to pa_services_team:
         - pa_recommendation → pa_services_team
    The decision step MUST have a higher order value than all evidence steps.
    Without this sequencing, the decision agent receives no evidence and returns
    REVIEW instead of APPROVE or DENY.
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
# Prompt: Response Consolidation
# ---------------------------------------------------------------------------
CONSOLIDATION_SYSTEM_PROMPT = """\
You are the central supervisor for a health insurance customer service platform.
You have just completed executing a multi-step plan on behalf of a CSR.
Each step was handled by a specialist team. You now have all the results.
Your job is to synthesise these results into a single, clear, coherent response
that directly answers the CSR's original question.
=== RULES ===
1. Answer the original question directly and completely.
2. Integrate all relevant results into one flowing response — do not list
   team names or step numbers. The CSR does not need to know about the
   internal architecture.
3. If results from different teams are related (e.g. a member's eligibility
   affects a claim decision), explain that relationship clearly.
4. If a step returned no useful data, omit it — do not mention it.
5. Be concise. Use plain language appropriate for a CSR talking to a member.
6. Do not add information that was not in the results.
7. Do not fabricate or speculate beyond what the results contain.
"""
CONSOLIDATION_USER_PROMPT = """\
Original question: {query}
Results from each specialist team:
{results_block}
Write a single consolidated response that directly answers the original question.
"""
# ---------------------------------------------------------------------------
# Central Supervisor
# ---------------------------------------------------------------------------
def _get_routing_prompt() -> str:
    """Return SUPERVISOR_SYSTEM_PROMPT from LangFuse if versioning is enabled."""
    try:
        from config.settings import get_settings as _gs
        _s = _gs()
        if getattr(_s, "LANGFUSE_PROMPT_VERSIONING_ENABLED", False):
            from observability.langfuse_integration import get_langfuse_tracer
            _label = getattr(_s, "LANGFUSE_PROMPT_LABEL", "production")
            return get_langfuse_tracer().get_prompt_or_default(
                "csip-central-routing-prompt", SUPERVISOR_SYSTEM_PROMPT, label=_label
            )
    except Exception:
        pass
    return SUPERVISOR_SYSTEM_PROMPT
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
        self._agent_nodes: Dict[str, A2AClientNode] = {
            agent_name: A2AClientNode(
                agent_name=agent_name,
                agent_url=AGENT_URLS[agent_name],
                schema_registry={},
                from_agent_name="central_supervisor",
            )
            for agent_name in VALID_REMOTE_AGENTS
        }
        self._populate_registry()
        logger.info(
            "CentralSupervisor initialised with %d remote agents: %s",
            len(self._agent_nodes),
            ", ".join(self._agent_nodes),
        )
    def _populate_registry(self) -> None:
        """
        Populate the AgentCardRegistry with cards for all five remote teams.
        """
        local_card_builders = {
            MEMBER_SERVICES:   build_member_services_agent_card,
            CLAIMS_SERVICES:   build_claims_services_agent_card,
            PA_SERVICES:       build_pa_services_agent_card,
            PROVIDER_SERVICES: build_provider_services_agent_card,
            SEARCH_SERVICES:   build_search_services_agent_card,
        }
        for agent_name, base_url in AGENT_URLS.items():
            self._registry.register_card_url(agent_name, base_url)
            card = None
            try:
                import httpx
                _card_base_url = base_url.replace(":8443/", ":443/")
                well_known_url = f"{_card_base_url}/.well-known/agent.json"
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
        """
        from langchain_core.output_parsers import StrOutputParser
        prompt = ChatPromptTemplate.from_messages([
            ("system", _get_routing_prompt()),
            ("human",  "{messages}"),
        ])
        return prompt | self._llm | StrOutputParser()
    # -----------------------------------------------------------------------
    # Node: create_plan
    # -----------------------------------------------------------------------
    def create_plan_node(self, state: SupervisorState) -> SupervisorState:
        """
        Phase 1 — Planning.
        """
        session_id = state.get("session_id", "default")
        user_id    = state.get("user_id", "unknown")
        messages   = state.get("messages", [])
        query = ""
        if messages:
            last = messages[-1]
            query = last.content if hasattr(last, "content") else str(last)
        logger.info("CentralSupervisor: building plan for session=%s", session_id)
        skills_summary = self._registry.get_skills_summary()
        if not skills_summary.strip():
            logger.warning("AgentCardRegistry has no registered cards — using fallback summary")
            skills_summary = _fallback_skills_summary()
        try:
            _sem_compressor = get_semantic_compressor()
            skills_summary = _sem_compressor.compress_text(skills_summary)
        except Exception as _exc:
            logger.warning(
                "CentralSupervisor: skills summary compression failed (non-fatal): %s", _exc
            )
        past_plans_text = ""
        try:
            from config.settings import get_settings as _exp_settings
            if getattr(_exp_settings(), "EXPERIENCE_STORE_ENABLED", False):
                from databases.chroma_experience_store import get_experience_store
                _top_k = getattr(_exp_settings(), "EXPERIENCE_TOP_K", 3)
                past_plans_text = get_experience_store().retrieve_similar_experiences(
                    query_text=query, top_k=_top_k,
                )
                if past_plans_text:
                    logger.info(
                        "CentralSupervisor: injecting %d experience examples for session=%s",
                        past_plans_text.count("Example"), session_id,
                    )
        except Exception as _exp_exc:
            logger.warning(
                "CentralSupervisor: experience retrieval failed (non-fatal): %s", _exp_exc
            )
        if _experience_planning_total is not None:
            try:
                _experience_planning_total.inc()
                if past_plans_text:
                    _experience_planning_hits.inc()
            except Exception:
                pass
        try:
            from config.settings import get_settings as _gs
            _s = _gs()
            if getattr(_s, "LANGFUSE_PROMPT_VERSIONING_ENABLED", False):
                tracer = get_langfuse_tracer()
                _label = getattr(_s, "LANGFUSE_PROMPT_LABEL", "production")
                _raw = tracer.get_prompt_or_default(
                    "csip-central-planning-prompt", PLANNING_SYSTEM_PROMPT, label=_label
                )
                planning_prompt = _raw.format(agent_skills_summary=skills_summary, past_successful_plans=past_plans_text)
            else:
                planning_prompt = PLANNING_SYSTEM_PROMPT.format(
                    agent_skills_summary=skills_summary,
                    past_successful_plans=past_plans_text,
                )
        except Exception as _ve:
            logger.warning("CentralSupervisor: prompt versioning error (using default): %s", _ve)
            planning_prompt = PLANNING_SYSTEM_PROMPT.format(
                agent_skills_summary=skills_summary,
                past_successful_plans=past_plans_text,
            )
        user_prompt = PLANNING_USER_PROMPT.format(query=query)
        session_id_val = state.get("session_id", "")
        thread_config  = RunnableConfig(
            configurable={"thread_id": session_id_val}
        )
        messages = state.get("messages", [])
        planning_chain = self._llm | JsonOutputParser()
        tracer = get_langfuse_tracer()
        planning_execution_id: Optional[str] = None
        try:
            planning_execution_id = self._cg.track_agent_execution(
                session_id=session_id,
                agent_name="central_supervisor_planner",
                agent_type="supervisor",
                status="running",
                metadata={"user_id": user_id},
            )
            if planning_execution_id:
                self._cg.link_session_to_execution(session_id, planning_execution_id)
        except Exception as exc:
            logger.warning("CentralSupervisor: CG track planning execution (non-fatal): %s", exc)
        callback_handler = tracer.get_session_callback_handler(
            session_id=session_id,
            user_id=user_id,
        ) if tracer.enabled else None
        _state_messages  = messages
        _prior_msgs      = _state_messages[:-1] if len(_state_messages) > 1 else []
        _messages_to_llm = [SystemMessage(content=planning_prompt)]
        if _prior_msgs:
            _messages_to_llm.extend(_prior_msgs)
        _messages_to_llm.append(HumanMessage(content=user_prompt))
        plan: Optional[Dict[str, Any]] = None
        try:
            plan = planning_chain.invoke(
                _messages_to_llm,
                config={**thread_config,
                        "callbacks": [callback_handler] if callback_handler else []},
            )
            if callback_handler and planning_execution_id:
                try:
                    lf_trace_id = callback_handler.get_trace_id()
                    if lf_trace_id:
                        self._cg.set_langfuse_trace_id(planning_execution_id, lf_trace_id)
                        logger.debug("CentralSupervisor: linked LangFuse trace %s to planner exec %s", lf_trace_id, planning_execution_id)
                    else:
                        logger.debug("CentralSupervisor: callback_handler.get_trace_id() returned None for planner exec %s", planning_execution_id)
                except Exception as _lf_exc:
                    logger.warning("CentralSupervisor: LangFuse trace writeback failed (non-fatal): %s", _lf_exc)
            try:
                validate_central_plan(plan)
            except Exception as _val_exc:
                logger.error("CentralSupervisor: plan schema invalid: %s", _val_exc)
                raise ValueError(f"Plan validation failed: {_val_exc}") from _val_exc
            logger.info(
                "CentralSupervisor: plan created and validated — %d goals, %d steps",
                len(plan.get("goals", [])),
                len(plan.get("steps", [])),
            )
        except Exception as exc:
            logger.error("CentralSupervisor: planning LLM failed: %s", exc)
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
        if planning_execution_id:
            try:
                self._cg.update_execution_status(
                    execution_id=planning_execution_id,
                    status="completed",
                )
            except Exception as exc:
                logger.warning("CentralSupervisor: CG close planning execution (non-fatal): %s", exc)
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
                if plan_id and planning_execution_id:
                    self._cg.link_planner_to_plan(planning_execution_id, plan_id)
        except Exception as exc:
            logger.warning("CentralSupervisor: CG store_plan failed (non-fatal): %s", exc)
        return {
            "plan":               plan,
            "plan_id":            plan_id,
            "step_map":           step_map,
            "current_step_index": 0,
            "completed_goals":    [],
            "execution_path":     state.get("execution_path", []) + ["create_plan"],
            "start_time":         datetime.now(timezone.utc).isoformat(),
            "plan_type":          "central",
            "team_name":          "",
            "planning_execution_id": planning_execution_id or "",
        }
    # -----------------------------------------------------------------------
    # Node: supervisor
    # -----------------------------------------------------------------------
    def supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """
        Phase 2 — Routing.
        """
        session_id = state.get("session_id", "default")
        user_id    = state.get("user_id", "unknown")
        plan       = state.get("plan") or {}
        steps      = sorted(plan.get("steps", []), key=lambda s: s.get("order", 1))
        goals      = plan.get("goals", [])
        idx        = state.get("current_step_index", 0)
        if idx >= len(steps):
            logger.info("CentralSupervisor: all steps complete → consolidate_response")
            return {"next": "consolidate_response"}
        current_step = steps[idx]
        agent_name   = current_step.get("agent", "")
        instruction  = current_step.get("instruction", "")
        goal_id      = current_step.get("goal_id", "")
        goal         = next((g for g in goals if g.get("id") == goal_id), {})
        goal_desc    = goal.get("description", "")
        execution_id: Optional[str] = None
        try:
            execution_id = self._cg.track_agent_execution(
                session_id=session_id,
                agent_name="central_supervisor_router",
                agent_type="supervisor",
                status="running",
                metadata={"step_index": idx, "assigned_agent": agent_name},
            )
        except Exception as exc:
            logger.warning("CentralSupervisor: CG track_agent_execution failed (non-fatal): %s", exc)
        tracer = get_langfuse_tracer()
        callback_handler = tracer.get_session_callback_handler(
            session_id=session_id,
            user_id=user_id,
        ) if tracer.enabled else None
        routing_response = "SKIP"
        try:
            routing_chain = self._create_routing_chain()
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
            if callback_handler and execution_id:
                try:
                    lf_trace_id = callback_handler.get_trace_id()
                    if lf_trace_id:
                        self._cg.set_langfuse_trace_id(execution_id, lf_trace_id)
                        logger.debug("CentralSupervisor: linked LangFuse trace %s to routing exec %s", lf_trace_id, execution_id)
                except Exception as _lf_exc:
                    logger.warning("CentralSupervisor: LangFuse routing trace writeback failed: %s", _lf_exc)
        except Exception as exc:
            logger.error("CentralSupervisor: supervisor LLM failed: %s", exc)
            routing_response = "SKIP"
        logger.info(
            "CentralSupervisor: step[%d] agent=%s → routing=%s",
            idx, agent_name, routing_response,
        )
        if routing_response not in VALID_REMOTE_AGENTS and routing_response != "SKIP":
            logger.warning(
                "CentralSupervisor: unexpected routing value '%s' → forcing SKIP",
                routing_response,
            )
            routing_response = "SKIP"
        if execution_id:
            try:
                self._cg.update_execution_status(
                    execution_id=execution_id,
                    status="completed" if routing_response != "SKIP" else "skipped",
                    routing_note=f"Routing: {routing_response} — {goal_desc}",
                )
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
            "last_step_was_skipped": routing_response == "SKIP",
        }
    # -----------------------------------------------------------------------
    # Node: agent delegation (one per remote agent)
    # -----------------------------------------------------------------------
    def _build_agent_node(self, agent_name: str):
        """
        Return a LangGraph node function that delegates to the named remote agent.
        """
        a2a_node = self._agent_nodes[agent_name]
        def agent_node(state: SupervisorState) -> SupervisorState:
            plan  = state.get("plan") or {}
            steps = sorted(plan.get("steps", []), key=lambda s: s.get("order", 1))
            idx   = state.get("current_step_index", 0)
            if idx < len(steps):
                current_step = steps[idx]
                delegation_query = (
                    current_step.get("query")
                    or current_step.get("instruction", "")
                )
                if idx > 0:
                    tool_results = state.get("tool_results", {})
                    if tool_results:
                        prior_context_parts = []
                        for worker_name, result in tool_results.items():
                            if isinstance(result, dict):
                                output = result.get("output", "").strip()
                                if output:
                                    prior_context_parts.append(
                                        f"- {worker_name} returned: {output[:2000]}"
                                    )
                        if prior_context_parts:
                            delegation_query = (
                                delegation_query
                                + "\n\n=== RESULTS FROM PRIOR STEPS (use these values directly) ===\n"
                                + "\n".join(prior_context_parts)
                                + "\n\nExtract any IDs, names, or values from the above results "
                                + "and use them directly in your plan. Do not ask for data that "
                                + "is already provided above."
                            )
                injected_state = dict(state)
                existing_messages = list(state.get("messages", []))
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
            result = a2a_node(injected_state)
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
            goal_step_indices = [
                i for i, s in enumerate(steps) if s.get("goal_id") == goal_id
            ]
            all_goal_steps_done = all(i <= idx for i in goal_step_indices)
            if all_goal_steps_done and goal_id not in completed_goals:
                completed_goals.append(goal_id)
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
    # Node: consolidate_response
    # -----------------------------------------------------------------------
    def consolidate_response_node(self, state: SupervisorState) -> SupervisorState:
        """
        Phase 4 — Response Consolidation.
        """
        session_id   = state.get("session_id", "default")
        user_id      = state.get("user_id", "unknown")
        messages     = state.get("messages", [])
        tool_results = state.get("tool_results", {})
        plan         = state.get("plan") or {}
        original_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                original_query = msg.content if hasattr(msg, "content") else str(msg)
                break
        worker_to_goal: Dict[str, str] = {}
        for step in plan.get("steps", []):
            worker = step.get("agent", "")
            goal_id = step.get("goal_id", "")
            goal = next(
                (g for g in plan.get("goals", []) if g.get("id") == goal_id), {}
            )
            if worker and goal:
                worker_to_goal[worker] = goal.get("description", worker)
        results_parts = []
        for worker_name, result in tool_results.items():
            if not isinstance(result, dict):
                continue
            output = result.get("output", "").strip()
            if not output:
                continue
            goal_desc = worker_to_goal.get(worker_name, worker_name)
            results_parts.append(f"[{goal_desc}]\n{output}")
        if not results_parts:
            logger.warning(
                "CentralSupervisor: consolidate_response_node: no tool outputs to consolidate"
            )
            return {
                "messages":       [AIMessage(content="I was unable to retrieve the information needed to answer your question.")],
                "execution_path": state.get("execution_path", []) + ["consolidate_response"],
                "next":           "FINISH",
            }
        results_block = "\n\n".join(results_parts)
        _consolidation_sys = CONSOLIDATION_SYSTEM_PROMPT
        _consolidation_usr_tpl = CONSOLIDATION_USER_PROMPT
        try:
            from config.settings import get_settings as _gs
            _s = _gs()
            if getattr(_s, "LANGFUSE_PROMPT_VERSIONING_ENABLED", False):
                from observability.langfuse_integration import get_langfuse_tracer as _get_tracer
                _tracer = _get_tracer()
                _label = getattr(_s, "LANGFUSE_PROMPT_LABEL", "production")
                _consolidation_sys = _tracer.get_prompt_or_default(
                    "csip-consolidation-system-prompt", CONSOLIDATION_SYSTEM_PROMPT, label=_label
                )
                _consolidation_usr_tpl = _tracer.get_prompt_or_default(
                    "csip-consolidation-user-prompt", CONSOLIDATION_USER_PROMPT, label=_label
                )
        except Exception:
            pass
        consolidation_user = _consolidation_usr_tpl.format(
            query=original_query,
            results_block=results_block,
        )
        logger.info(
            "CentralSupervisor: consolidating %d team results for session=%s",
            len(results_parts), session_id,
        )
        tracer = get_langfuse_tracer()
        consolidation_execution_id: Optional[str] = None
        try:
            consolidation_execution_id = self._cg.track_agent_execution(
                session_id=session_id,
                agent_name="response_consolidator",
                agent_type="consolidator",
                status="running",
                metadata={"user_id": user_id},
            )
            if consolidation_execution_id:
                try:
                    from databases.context_graph_data_access import get_cg_data_access
                    cg_dal = get_cg_data_access()
                    planner_result = cg_dal.conn.execute_query("""
                        MATCH (s:Session {sessionId: $sessionId})
                              -[:HAS_EXECUTION]->(planner:AgentExecution {agentName: 'central_supervisor_planner'})
                        RETURN planner.executionId AS plannerId LIMIT 1
                    """, {"sessionId": session_id})
                    planner_exec_id = planner_result[0]["plannerId"] if planner_result else ""
                    if planner_exec_id:
                        cg_dal.conn.execute_query("""
                            MATCH (planner:AgentExecution {executionId: $plannerId})
                            MATCH (consolidator:AgentExecution {executionId: $consolidatorId})
                            MERGE (planner)-[:HAS_EXECUTION]->(consolidator)
                        """, {
                            "plannerId": planner_exec_id,
                            "consolidatorId": consolidation_execution_id,
                        })
                    else:
                        logger.warning("CentralSupervisor: planner not found in CG — consolidator linked to session")
                        self._cg.link_session_to_execution(session_id, consolidation_execution_id)
                except Exception as _link_exc:
                    logger.warning("CentralSupervisor: failed to link consolidator to planner: %s", _link_exc)
                    self._cg.link_session_to_execution(session_id, consolidation_execution_id)
        except Exception as exc:
            logger.warning("CentralSupervisor: CG track consolidation execution (non-fatal): %s", exc)
        callback_handler = tracer.get_session_callback_handler(
            session_id=session_id,
            user_id=user_id,
        ) if tracer.enabled else None
        try:
            from langchain_core.output_parsers import StrOutputParser
            consolidation_chain = self._llm | StrOutputParser()
            consolidated = consolidation_chain.invoke(
                [
                    SystemMessage(content=_consolidation_sys),
                    HumanMessage(content=consolidation_user),
                ],
                config={"callbacks": [callback_handler]} if callback_handler else {},
            )
            if callback_handler and consolidation_execution_id:
                try:
                    lf_trace_id = callback_handler.get_trace_id()
                    if lf_trace_id:
                        self._cg.set_langfuse_trace_id(consolidation_execution_id, lf_trace_id)
                        logger.debug("CentralSupervisor: linked LangFuse trace %s to consolidation exec %s", lf_trace_id, consolidation_execution_id)
                except Exception as _lf_exc:
                    logger.warning("CentralSupervisor: LangFuse consolidation trace writeback failed: %s", _lf_exc)
            if consolidation_execution_id:
                try:
                    self._cg.update_execution_status(consolidation_execution_id, "completed")
                except Exception:
                    pass
            logger.info(
                "CentralSupervisor: consolidation complete for session=%s", session_id
            )
        except Exception as exc:
            logger.error(
                "CentralSupervisor: consolidation LLM failed (falling back to concatenation): %s",
                exc,
            )
            if consolidation_execution_id:
                try:
                    self._cg.update_execution_status(consolidation_execution_id, "failed", error=str(exc))
                except Exception:
                    pass
            consolidated = "\n\n".join(p.split("\n", 1)[-1] for p in results_parts)
        return {
            "messages":       [AIMessage(content=consolidated)],
            "execution_path": state.get("execution_path", []) + ["consolidate_response"],
            "next":           "FINISH",
        }
    # -----------------------------------------------------------------------
    # Node: error_handler
    # -----------------------------------------------------------------------
    def error_handler_node(self, state: SupervisorState) -> SupervisorState:
        """
        Handle an unrecoverable error in the plan execution.
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
        if execution_id:
            try:
                self._cg.create_agent_error(
                    execution_id=execution_id,
                    error_type=classify_error(error_msg),
                    error_message=error_msg,
                )
            except Exception as exc:
                logger.warning("CentralSupervisor: CG create_agent_error (non-fatal): %s", exc)
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
        user_message = format_error_for_user(error_msg)
        end_time     = datetime.now(timezone.utc).isoformat()
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
        """
        next_node = state.get("next", "FINISH")
        if next_node == "FINISH":
            return "FINISH"
        if next_node == "consolidate_response":
            return "consolidate_response"
        if next_node == "SKIP":
            return "goal_advance"
        if next_node in VALID_REMOTE_AGENTS:
            return next_node
        logger.warning("CentralSupervisor: unknown routing target '%s' → error", next_node)
        return "error_handler"
    # -----------------------------------------------------------------------
    # Error router: determines next node after a remote agent returns an error
    # -----------------------------------------------------------------------
    def error_router(self, state: SupervisorState) -> str:
        """
        Called after every remote agent node.
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
        """
        graph = StateGraph(SupervisorState)
        graph.add_node("create_plan",          self.create_plan_node)
        graph.add_node("supervisor",           self.supervisor_node)
        graph.add_node("goal_advance",         self.goal_advance_node)
        graph.add_node("consolidate_response", self.consolidate_response_node)
        graph.add_node("error_handler",        self.error_handler_node)
        for agent_name in VALID_REMOTE_AGENTS:
            graph.add_node(agent_name, self._build_agent_node(agent_name))
        graph.set_entry_point("create_plan")
        graph.add_edge("create_plan",          "supervisor")
        graph.add_edge("goal_advance",         "supervisor")
        graph.add_edge("consolidate_response", END)
        graph.add_edge("error_handler",        END)
        graph.add_conditional_edges(
            "supervisor",
            self.router,
            {
                **{agent: agent for agent in VALID_REMOTE_AGENTS},
                "goal_advance":         "goal_advance",
                "consolidate_response": "consolidate_response",
                "error_handler":        "error_handler",
                "FINISH":               END,
            },
        )
        for agent_name in VALID_REMOTE_AGENTS:
            graph.add_conditional_edges(
                agent_name,
                self.error_router,
                {
                    "goal_advance":  "goal_advance",
                    "error_handler": "error_handler",
                },
            )
        from agents.core.context_graph_store import get_context_graph_store
        return graph.compile(checkpointer=get_context_graph_store())
# ---------------------------------------------------------------------------
# Singleton graph factory
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_central_supervisor_graph():
    """
    Build and cache the compiled CentralSupervisor graph.
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
  Description: Member lookup, eligibility verification, coverage/benefits inquiries, and treatment history.
  Skills:
    - member_lookup: Look up member demographics and contact details by member ID.
    - check_eligibility: Verify active coverage status for a service date.
    - coverage_lookup: Retrieve deductibles, copays, and benefits detail.
    - member_policy_lookup: Look up member with their associated insurance policy.
    - update_member_info: Update member contact or demographic information.
    - treatment_history: Retrieve treatment records for a member (physical therapy, medications, imaging).

Agent: claims_services_team
  Description: Claim lookup, status checks, payment/financial information, and claim adjudication.
  Skills:
    - claim_lookup: Look up full claim details by claim ID.
    - claim_status: Check processing status of a claim by claim number.
    - claim_payment_info: Retrieve payment and EOB information for a claim.
    - member_claims: Retrieve all claims filed by a specific member.
    - update_claim_status: Update claim status (requires HITL approval).
    - claim_adjudication: Evaluate claim validity against coverage rules. Decision agent — requires prior evidence from claim_lookup, check_eligibility, and provider_network_check.

Agent: pa_services_team
  Description: Prior authorization lookup, status, requirements, and PA recommendation.
  Skills:
    - pa_lookup: Look up full PA details by PA ID.
    - pa_status: Check current status of a prior authorization.
    - pa_requirements: Determine if a procedure requires PA under a given policy type.
    - member_prior_authorizations: Retrieve all prior authorizations for a member.
    - approve_prior_auth: Approve a prior authorization (requires HITL approval).
    - deny_prior_auth: Deny a prior authorization (requires HITL approval).
    - pa_recommendation: Evaluate whether a PA should be approved or denied based on clinical criteria. Decision agent — requires prior evidence from pa_lookup, pa_requirements, search_knowledge_base, and treatment_history.

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
