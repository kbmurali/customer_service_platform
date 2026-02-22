"""
Base classes and utilities for agent implementation.
"""

from typing import Callable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.core.state import AgentState, SupervisorState
from agents.security import check_permission, log_audit


class BaseAgent:
    """Base class for all worker agents."""
    
    def __init__(self, name: str, description: str, tools: list, system_prompt: str):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def create_agent(self):
        """Create a LangGraph agent with tools."""
        return create_react_agent(
            self.llm,
            tools=self.tools,
            state_modifier=self.system_prompt
        )
    
    def node(self, state: AgentState) -> AgentState:
        """Agent node function for LangGraph."""
        # Security check
        user_role = state.get("user_role", "CSR_TIER1")
        
        # Log agent execution
        log_audit(
            user_id=state.get("user_id", "unknown"),
            action="agent_execution",
            resource=self.name,
            details={"session_id": state.get("session_id")}
        )
        
        # Execute agent
        agent = self.create_agent()
        result = agent.invoke(state)
        
        return {
            "messages": result["messages"],
            "next": "supervisor"  # Return to supervisor
        }


class BaseSupervisor:
    """Base class for supervisor agents."""
    
    def __init__(self, name: str, members: list[str], system_prompt: str):
        self.name = name
        self.members = members
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def create_supervisor_chain(self, routing_type):
        """Create supervisor routing chain."""
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        
        options = self.members + ["FINISH"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{messages}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        return chain
    
    def node(self, state: SupervisorState, routing_type) -> SupervisorState:
        """Supervisor node function."""
        chain = self.create_supervisor_chain(routing_type)
        
        # Get routing decision
        result = chain.invoke({
            "messages": state["messages"],
            "options": self.members + ["FINISH"]
        })
        
        next_agent = result.get("next", "FINISH")
        
        # Update execution path
        execution_path = state.get("execution_path", [])
        execution_path.append(f"{self.name} -> {next_agent}")
        
        # Log routing decision
        log_audit(
            user_id=state.get("user_id", "unknown"),
            action="supervisor_routing",
            resource=self.name,
            details={
                "next_agent": next_agent,
                "reasoning": result.get("reasoning", "")
            }
        )
        
        return {
            "next": next_agent,
            "execution_path": execution_path
        }


def create_team_graph(supervisor: BaseSupervisor, workers: dict[str, BaseAgent], routing_type):
    """Create a team graph with supervisor and workers."""
    from langgraph.graph import StateGraph, END
    from agents.core.state import SupervisorState
    
    workflow = StateGraph(SupervisorState)
    
    # Add supervisor node
    workflow.add_node("supervisor", lambda state: supervisor.node(state, routing_type))
    
    # Add worker nodes
    for name, agent in workers.items():
        workflow.add_node(name, agent.node)
    
    # Add edges from workers back to supervisor
    for name in workers.keys():
        workflow.add_edge(name, "supervisor")
    
    # Add conditional edges from supervisor
    def route_supervisor(state: SupervisorState):
        return state["next"]
    
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {**{name: name for name in workers.keys()}, "FINISH": END}
    )
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
