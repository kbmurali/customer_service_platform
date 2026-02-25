"""Core state definitions for the hierarchical agent system."""

from typing import Annotated, Literal, TypedDict, Optional, List, Dict, Any
from datetime import datetime
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Base state for all agents."""
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    user_id: str
    user_role: str
    session_id: str


class SupervisorState(AgentState, total=False):
    """
    Extended state for supervisor agents.

    Shared by the central supervisor and all 5 team supervisors.
    Carries everything the Context Graph needs to build the full
    end-to-end trace across the entire agentic call chain:

        User -> CentralSup -> A2A -> TeamSup -> WorkerAgent -> MCP -> Tool

    CG graph built from this state:
        (Session)
          ├─[:HAS_PLAN]──▶ (Plan {planType:'central'})
          │                    └─[:HAS_GOAL]──▶ (Goal)
          │                                         └─[:HAS_STEP]──▶ (Step)
          │                                                              └─[:DELEGATED_TO]──▶ (Plan {planType:'team'})
          │                                                                                       └─[:HAS_GOAL]──▶ (Goal)
          │                                                                                                            └─[:HAS_STEP]──▶ (Step)
          │                                                                                                                                 └─[:EXECUTED_BY]──▶ (AgentExecution)
          │                                                                                                                                                         └─[:CALLED_TOOL]──▶ (ToolExecution)
          ├─[:HAS_EXECUTION]──▶ (AgentExecution)   # flat index, all supervisors
          └─[:USED_TOOL]──────▶ (ToolExecution)    # flat index, all tool calls
    """
    team: str
    worker: str
    execution_path: list[str]
    tool_results: dict

    # ── Plan identity ─────────────────────────────────────────────────────
    # "central" for the central supervisor, "team" for all team supervisors.
    plan_type: str                       # "central" | "team"
    team_name: str                       # e.g. "member_services"; "" for central

    # ── Plan content & progress ───────────────────────────────────────────
    plan_id: Optional[str]               # Neo4j Plan.planId for THIS supervisor's plan
    plan: Optional[Dict[str, Any]]       # Goals + steps created by first LLM call
    current_goal_index: int              # Index into plan["goals"] currently executing
    completed_goals: List[str]           # Goal IDs completed or skipped
    step_map: Dict[str, str]             # {step_id -> step_id} returned by store_plan()

    # ── Context Graph traceability ────────────────────────────────────────
    # current_execution_id: AgentExecution.executionId from the most recent
    # supervisor routing decision.  Stored so worker_node can pass it to the
    # MCP tool, completing:
    #   (AgentExecution)-[:CALLED_TOOL]->(ToolExecution)
    current_execution_id: str

    # central_step_id: Step.stepId from the CENTRAL plan that delegated
    # work here via A2A.  Injected into team state by the A2A server on
    # receipt.  Used by team store_plan() to create:
    #   (CentralStep)-[:DELEGATED_TO]->(TeamPlan)
    # Empty string for the central supervisor itself.
    central_step_id: str

    # ── Error handling ────────────────────────────────────────────────────
    error: Optional[str]                 # Current error message
    error_count: int                     # Total errors in this workflow
    error_history: List[Dict[str, Any]]  # Detailed error history
    retry_count: int                     # Retries attempted
    is_recoverable: bool                 # Whether the error is recoverable
    fallback_used: bool                  # Whether fallback was triggered

    # ── Monitoring ────────────────────────────────────────────────────────
    start_time: Optional[str]            # ISO format timestamp
    end_time: Optional[str]             # ISO format timestamp
    duration_ms: Optional[float]         # Total duration in milliseconds


# Routing types for central supervisor (all 5 expert teams)
CentralRouting = Literal[
    "member_services_team",
    "claim_services_team",
    "pa_services_team",
    "provider_services_team",
    "search_services_team",
    "FINISH"
]

# Routing types for Member Services team
MemberServicesRouting = Literal[
    "member_lookup_agent",
    "eligibility_agent", 
    "coverage_agent",
    "FINISH"
]

# Routing types for Claim Services team
ClaimServicesRouting = Literal[
    "claim_lookup_agent",
    "claim_status_agent",
    "FINISH"
]

# Routing types for PA Services team
PAServicesRouting = Literal[
    "pa_lookup_agent",
    "pa_policy_agent",
    "FINISH"
]

# Routing types for Provider Services team
ProviderServicesRouting = Literal[
    "provider_search_agent",
    "network_agent",
    "FINISH"
]
