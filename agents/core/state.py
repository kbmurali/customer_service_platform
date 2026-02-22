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
    """Extended state for supervisor agents with routing information and error handling."""
    team: str
    worker: str
    execution_path: list[str]
    tool_results: dict
    
    # Planning fields
    plan_id: Optional[str] 
    plan: Optional[Dict[str, Any]]  # Ordered goals/plan created by first LLM call
    current_goal_index: int  # Index of current goal being executed
    completed_goals: List[str]  # List of completed goal IDs
    
    # Error handling fields
    error: Optional[str]  # Current error message
    error_count: int  # Total number of errors in this workflow
    error_history: List[Dict[str, Any]]  # Detailed error history
    retry_count: int  # Number of retries attempted
    is_recoverable: bool  # Whether the error is recoverable
    fallback_used: bool  # Whether fallback mechanism was triggered
    
    # Monitoring fields
    start_time: Optional[str]  # ISO format timestamp
    end_time: Optional[str]  # ISO format timestamp
    duration_ms: Optional[float]  # Total duration in milliseconds


# Routing types for central supervisor
CentralRouting = Literal[
    "member_services_team",
    "claim_services_team", 
    "pa_services_team",
    "provider_services_team",
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
