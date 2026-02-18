"""
Central Supervisor - Routes queries to appropriate service teams.

Architecture:
    - member_services_team: Remote MCP agent (encrypted communication)
    - claim_services_team:  Remote MCP agent (encrypted communication)
    - pa_services_team:     Local in-process agent (in-memory state)
    - provider_services_team: Local in-process agent (in-memory state)
"""

import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

from agents.core.state import SupervisorState, CentralRouting
from agents.security import log_audit
from security.approval_workflow import get_approval_workflow, CircuitBreakerError

# Local teams (in-process, no encryption needed)
from agents.teams.pa_services import pa_services_team
from agents.teams.provider_services import provider_services_team

# Remote MCP node wrapper (encrypted communication)
from agents.core.remote_mcp_node import RemoteMCPNode


class CentralSupervisor:
    """Central supervisor that routes to service teams."""
    
    def __init__(self):
        self.name = "central_supervisor"
        self.teams = [
            "member_services_team",
            "claim_services_team",
            "pa_services_team",
            "provider_services_team"
        ]
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.system_prompt = """You are the central supervisor for a health insurance customer service AI system.
        
        Your service teams:
        - member_services_team: Handle member lookup, eligibility, and coverage questions (remote MCP agent)
        - claim_services_team: Handle claim lookup and status inquiries (remote MCP agent)
        - pa_services_team: Handle prior authorization lookup and requirements (local)
        - provider_services_team: Handle provider search and network verification (local)
        
        Route the user's query to the most appropriate team based on the content:
        - Member questions (eligibility, coverage, benefits) → member_services_team
        - Claim questions (status, payment, details) → claim_services_team
        - Prior auth questions (PA status, requirements) → pa_services_team
        - Provider questions (search, network status) → provider_services_team
        
        Respond with JSON: {"next": "team_name", "reasoning": "why this team"}
        Use "FINISH" only when the query has been fully answered by a team."""
        
    def create_routing_chain(self):
        """Create the routing chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{messages}")
        ])
        
        return prompt | self.llm | JsonOutputParser()
    
    def node(self, state: SupervisorState) -> SupervisorState:
        """Central supervisor node with circuit breaker gate."""
        # ── Control 5: Circuit Breaker Gate ──
        # If the kill switch is active, halt immediately.
        try:
            approval_wf = get_approval_workflow()
            if approval_wf.is_circuit_breaker_active():
                log_audit(
                    user_id=state.get("user_id", "unknown"),
                    action="circuit_breaker_block",
                    resource="central_supervisor",
                    details={"reason": "Circuit breaker active - all agent processing halted"}
                )
                return {
                    "next": "FINISH",
                    "team": None,
                    "execution_path": state.get("execution_path", []) + ["central_supervisor -> BLOCKED_BY_CIRCUIT_BREAKER"],
                    "messages": "System is temporarily unavailable. A supervisor has activated the emergency stop. Please try again later."
                }
        except Exception as e:
            logger.warning(f"Circuit breaker check failed (fail-open): {e}")

        chain = self.create_routing_chain()
        
        # Get routing decision
        result = chain.invoke({
            "messages": state["messages"],
            "options": self.teams + ["FINISH"]
        })
        
        next_team = result.get("next", "FINISH")
        
        # Update execution path
        execution_path = state.get("execution_path", [])
        execution_path.append(f"central_supervisor -> {next_team}")
        
        # Log routing decision with transport type annotation
        is_remote = next_team in ("member_services_team", "claim_services_team")
        log_audit(
            user_id=state.get("user_id", "unknown"),
            action="central_routing",
            resource="central_supervisor",
            details={
                "next_team": next_team,
                "reasoning": result.get("reasoning", ""),
                "transport": "remote_mcp_encrypted" if is_remote else "local_in_process",
            }
        )
        
        return {
            "next": next_team,
            "team": next_team,
            "execution_path": execution_path
        }


def create_hierarchical_graph():
    """
    Create the complete hierarchical agent graph.

    - member_services_team and claim_services_team are wired as RemoteMCPNode
      instances that communicate over encrypted HTTP (Control 8).
    - pa_services_team and provider_services_team remain local in-process
      subgraphs with direct .invoke() calls (no encryption needed).
    """
    workflow = StateGraph(SupervisorState)
    
    # Add central supervisor
    central = CentralSupervisor()
    workflow.add_node("central_supervisor", central.node)
    
    # ── Remote MCP agents (encrypted communication) ──
    member_services_node = RemoteMCPNode(
        agent_name="member_services_team",
        base_url=os.getenv("MCP_MEMBER_SERVICES_URL", "https://mcp-member:8443"),
    )
    claim_services_node = RemoteMCPNode(
        agent_name="claim_services_team",
        base_url=os.getenv("MCP_CLAIM_SERVICES_URL", "https://mcp-claims:8443"),
    )
    workflow.add_node("member_services_team", member_services_node)
    workflow.add_node("claim_services_team", claim_services_node)
    
    # ── Local in-process agents (no encryption needed) ──
    workflow.add_node("pa_services_team", pa_services_team.invoke)
    workflow.add_node("provider_services_team", provider_services_team.invoke)
    
    # Add edges from teams back to central supervisor
    for team in ["member_services_team", "claim_services_team", "pa_services_team", "provider_services_team"]:
        workflow.add_edge(team, "central_supervisor")
    
    # Add conditional edges from central supervisor
    def route_central(state: SupervisorState):
        return state["next"]
    
    workflow.add_conditional_edges(
        "central_supervisor",
        route_central,
        {
            "member_services_team": "member_services_team",
            "claim_services_team": "claim_services_team",
            "pa_services_team": "pa_services_team",
            "provider_services_team": "provider_services_team",
            "FINISH": END
        }
    )
    
    workflow.set_entry_point("central_supervisor")
    
    return workflow.compile()


# Create and export the main graph
hierarchical_agent_graph = create_hierarchical_graph()
