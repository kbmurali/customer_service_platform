"""
Central Supervisor - Routes queries to appropriate service teams.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agents.core.state import SupervisorState, CentralRouting
from agents.security import log_audit
from agents.teams.member_services import member_services_team
from agents.teams.claim_services import claim_services_team
from agents.teams.pa_services import pa_services_team
from agents.teams.provider_services import provider_services_team


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
        - member_services_team: Handle member lookup, eligibility, and coverage questions
        - claim_services_team: Handle claim lookup and status inquiries
        - pa_services_team: Handle prior authorization lookup and requirements
        - provider_services_team: Handle provider search and network verification
        
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
        """Central supervisor node."""
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
        
        # Log routing decision
        log_audit(
            user_id=state.get("user_id", "unknown"),
            action="central_routing",
            resource="central_supervisor",
            details={
                "next_team": next_team,
                "reasoning": result.get("reasoning", "")
            }
        )
        
        return {
            "next": next_team,
            "team": next_team,
            "execution_path": execution_path
        }


def create_hierarchical_graph():
    """Create the complete hierarchical agent graph."""
    workflow = StateGraph(SupervisorState)
    
    # Add central supervisor
    central = CentralSupervisor()
    workflow.add_node("central_supervisor", central.node)
    
    # Add service teams as subgraphs
    workflow.add_node("member_services_team", member_services_team.invoke)
    workflow.add_node("claim_services_team", claim_services_team.invoke)
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
