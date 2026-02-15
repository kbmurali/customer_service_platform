"""
Magentic Orchestration.

This implementation strictly adheres to the Magentic-One architecture:
1. Two Ledgers: 
   - TaskLedger (Strategic: Plan, Facts, Goal)
   - ProgressLedger (Tactical: History, Stall Count)
2. Dual-Loop Control:
   - Outer Loop (Planner): Generates/Refines the plan based on stalls or new facts.
   - Inner Loop (Manager): Executes the plan, delegates to workers, detects stalls.
3. Stall Detection: Explicit logic to break execution loops and trigger re-planning.
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_member_policy_id, get_policy_details, get_policy_metadata, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import re
import json
import logging
from typing import Dict, Optional, TypedDict, Literal, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

# -----------------------------
# 0) Basic Setup
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOOLS: Dict[str, Tool] = { 
    t.name: wrap_tool(t) 
    for t in [get_member_id, get_member_policy_id, get_policy_details, calculator, get_policy_metadata] 
}

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

# -----------------------------
# 1) The Two Ledgers & State
# -----------------------------

class TaskLedger(TypedDict, total=False):
    """
    STRATEGIC MEMORY: Held by the Outer Loop.
    Contains the high-level plan and consolidated facts.
    """
    goal: str
    plan: List[str]          # The high-level strategy steps
    facts: Dict[str, Any]    # Consolidated known truths (e.g., member_id found)
    is_complete: bool        # Flag to signal overall task completion

class ProgressLedger(TypedDict, total=False):
    """
    TACTICAL MEMORY: Held by the Inner Loop.
    Tracks immediate execution history to detect stalls.
    """
    history: List[str]       # Log of recent worker outputs
    stall_count: int         # How many times have we failed to progress?
    last_worker: Optional[str]

class MagenticState(TypedDict, total=False):
    # Inputs
    user_question: str
    
    # The Dual Ledgers
    task_ledger: TaskLedger
    progress_ledger: ProgressLedger
    
    # Worker Sandbox (Scratchpad for tools)
    current_worker_output: Optional[str]
    
    # Final Result
    final_answer: Optional[str]
    
# -----------------------------
# 2) Worker Agents/Nodes
# -----------------------------

#%%
# Worker 1: Member ID lookup (ReAct agent using the interactive tool)
member_id_lookup_agent_prompt = (
    "You are MemberIDLookupAgent. Your ONLY job is to produce the member_id string.\n"
    "If member id is present in the user's question, extract it.\n"
    "Otherwise, call the tool to ask the user.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', output that error.\n"
    "Return ONLY the member_id and nothing else."
)

member_id_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_member_id"]],
    prompt=member_id_lookup_agent_prompt,
)

def member_id_lookup_agent_node(state: MagenticState, config: RunnableConfig ) -> MagenticState:
    if state.get('member_id'):
        return state

    user_question = state.get( 'user_question' )
    
    human_message = HumanMessage(content=f"User Question: {user_question}")
    
    result = member_id_lookup_agent.invoke(
        {"messages": [ human_message ]},
        config=config
    )
    
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    
    if err:
        return {**state, "current_worker_output": err}

    return {**state, "current_worker_output": text}

# -- Worker 2: Policy Lookup --
def policy_lookup_node(state: MagenticState) -> MagenticState:
    """Worker: Finds Policy ID using Member ID."""
    facts = state.get( "task_ledger" )["facts"]
    
    member_id = facts.get("member_id")
    
    if not member_id:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} Missing member_id in facts."}
    
    try:
        res = TOOLS["get_member_policy_id"].invoke({"member_id": member_id})
        return {**state, "current_worker_output": str(res)}
    except Exception as e:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} {str(e)}"}

# -- Worker 3: Policy Details --
def policy_details_node(state: MagenticState) -> MagenticState:
    """Worker: Fetches text details."""
    facts = state.get( "task_ledger" )["facts"]
    
    policy_id = facts.get("policy_id")
    
    if not policy_id:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} Missing policy_id in facts."}
    
    try:
        res = TOOLS["get_policy_details"].invoke({"policy_id": policy_id})
        return {**state, "current_worker_output": str(res)}
    except Exception as e:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} {str(e)}"}

# -- Worker 4: Metadata --
def metadata_node(state: MagenticState) -> MagenticState:
    """Worker: Fetches numeric metadata."""
    facts = state.get( "task_ledger" )["facts"]
    
    policy_id = facts.get("policy_id")
    
    if not policy_id:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} Missing policy_id in facts."}
    
    try:
        res = TOOLS["get_policy_metadata"].invoke({"policy_id": policy_id})
        # If dict, simplify to string for the LLM
        return {**state, "current_worker_output": json.dumps(res) if isinstance(res, dict) else str(res)}
    except Exception as e:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} {str(e)}"}

# -- Worker 5: Calculator --
def calculator_node(state: MagenticState) -> MagenticState:
    """Worker: Performs math."""
    # The Inner Manager should have put the expression in the plan or passed it via context, 
    # but to keep it simple, we extract from the last plan instruction or facts.
    # In a robust implementation, the Manager passes arguments explicitly. 
    # Here we assume the Manager wrote the expression into the `facts` for the worker to find.
    expr = state.get( "task_ledger" )["facts"].get("calc_expression")
    
    if not expr:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} No calc_expression found in facts."}
    try:
        res = TOOLS["calculator"].invoke({"expression": expr})
        return {**state, "current_worker_output": f"Result: {res}"}
    except Exception as e:
        return {**state, "current_worker_output": f"{ERROR_PREFIX} {str(e)}"}
    
def response_writer_node(state: MagenticState, config: RunnableConfig ) -> MagenticState:
    system_message = SystemMessage(
        content=(
            "You are a friendly insurance policy assistant.\n"
            "Use ONLY information present in the shared facts:\n"
            "- policy_details (text evidence)\n"
            "- policy_metadata (numbers)\n"
            "- calc_expression + calc_result (if present)\n"
            "Be concise and explain the breakdown when you provide totals.\n"
        )
    )
    
    facts = state.get( "task_ledger" )["facts"]
    
    prompt = (
        f"User Question: {state.get('user_question','')}\n"
        f"Facts Collected: {json.dumps(facts, indent=2)}\n"
    )
    
    human_message = HumanMessage(content=prompt)
    
    res = llm.invoke(
                        [system_message, human_message],
                        config=config
                    ).content.strip()
    
    return {**state, "final_answer": res, "current_worker_output": "Drafted final answer."}

# -----------------------------
# 3) Orchestration Logic
# -----------------------------

# --- OUTER LOOP: PLANNER ---
def outer_planner_node(state: MagenticState, config: RunnableConfig) -> MagenticState:
    """
    Refines the TaskLedger (Plan & Facts).
    Triggered at start OR when Inner Loop stalls.
    """
    init_task_ledger : TaskLedger = { 
                                        "goal": state.get( "user_question", "" ),
                                        "plan": [], 
                                        "facts": {}, 
                                        "is_complete": False
                                     }
    
    task_ledger = state.get( "task_ledger", init_task_ledger)
    
    init_progress_ledger: ProgressLedger = {
                                                "history": [], 
                                                "stall_count": 0, 
                                                "last_worker": None
                                            }
    
    progress_ledger = state.get("progress_ledger", init_progress_ledger)
    
    # If this is a re-plan due to stall, the prompt changes
    context = (
        f"Goal: {task_ledger.get( 'goal' )}\n"
        f"Current Plan: {task_ledger.get( 'plan' )}\n"
        f"Known Facts: {json.dumps(task_ledger.get( 'facts' ))}\n"
        f"Recent Execution History: {progress_ledger.get( 'history', [] )[-5:]}\n" # See last 5 steps
    )
    
    system_prompt = (
        "You are the Lead Strategist. You manage the Task Ledger.\n"
        "1. Update the 'plan' (list of high-level steps) based on what we know and what is still missing.\n"
        "2. If execution history shows errors/stalls, change the plan to try a different approach.\n"
        "3. Ensure the plan covers gathering Member ID, Policy ID, Policy Details, Metadata, Calculation if needed, and Drafting of response.\n"
        "You MUST output ONLY valid JSON (no code fences).\n"
        "Schema:\n"
        "{ \"plan\": [str], \"facts_update\": {key: value} }"
    )

    try:
        raw = llm.invoke(
                            [SystemMessage(content=system_prompt), HumanMessage(content=context)],
                            config=config
                        ).content.strip()
        
        
        
        cleaned = re.sub(r"```json|```", "", raw).strip()
        
        data = json.loads(cleaned)
        
        # Update Task Ledger
        task_ledger["plan"] = data.get( 'plan', task_ledger.get( 'plan' ))
        
        task_ledger["facts"].update( data.get( 'facts_update', {}) )
        
        # Reset Progress Ledger Stall Count on Re-plan
        progress_ledger["stall_count"] = 0
    except Exception as e:
        logger.error(f"Planner failed: {e}")
    
    return {**state, "task_ledger": task_ledger, "progress_ledger": progress_ledger}

# --- INNER LOOP: MANAGER ---
def inner_manager_node(state: MagenticState, config: RunnableConfig) -> MagenticState:
    """
    Executes the current plan step.
    Decides: Which worker next? OR Is task done? OR Are we stalled?
    """
    task_ledger = state.get( 'task_ledger' )
    progress_ledger = state.get( 'progress_ledger' )
    
    # Check for previous worker output and log it
    if state.get( 'current_worker_output' ):
        log_entry = f"Worker {progress_ledger.get('last_worker')} output: {state.get( 'current_worker_output') }"
        progress_ledger["history"].append(log_entry)
        
        # Basic Fact Extraction (Simulated)
        # In a real system, a separate 'FactScraper' node might run here.
        # We'll just let the Manager LLM decide to update facts in the next turn or via direct update.
        output = state.get( 'current_worker_output' ) or ""
        
        if "member_id" in output and "Found" in output: # heuristic
             pass 

    # Manager Decision
    system_prompt = (
        "You are the Tactical Manager. Execute the plan provided by the Strategist.\n"
        "Available Workers: [member_id_lookup, policy_lookup, policy_details_lookup, metadata_lookup, calculator, writer]\n"
        "Rules:\n"
        "1. Look at the FIRST unsatisfied step in the Plan.\n"
        "2. Choose the best worker.\n"
        "3. If an error persists (stalls), output decision='STALL'.\n"
        "4. If plan is done and writer worker output is drafted, output decision='FINALIZE'.\n"
        "5. If you need to record a fact (like member id, policy id, policy details), put it in 'facts_update'.\n"
        "6. If you need to perform a calc, put the expression in 'calc_expression' in 'facts_update'.\n\n"
        "IMPORTANT: You MUST output ONLY valid JSON (no code fences).\n"
        "Schema:\n"
        "{ \"decision\": \"<worker_name>\"|\"STALL\"|\"FINALIZE\", \"facts_update\": {...} }"
    )

    user_prompt = (
        f"Plan: {task_ledger.get( 'plan' )}\n"
        f"Facts: {json.dumps(task_ledger.get( 'facts' ))}\n"
        f"History: {progress_ledger.get( 'history', [])[-3:]}\n" # Context window management
    )

    try:
        raw = llm.invoke(
                            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                            config=config
                        ).content.strip()
        
        cleaned = re.sub(r"```json|```", "", raw).strip()
        
        decision_data = json.loads(cleaned)
        
        decision = decision_data.get("decision")
        
        # Update Facts immediately (Tactical update)
        if "facts_update" in decision_data:
            task_ledger["facts"].update( decision_data.get( 'facts_update', {}) )
        
        # Update Progress
        progress_ledger["last_worker"] = decision

        # Stall Detection Logic (Simple)
        if decision == "STALL":
            progress_ledger["stall_count"] += 1
            
        # Update State
        return {**state, "task_ledger": task_ledger, "progress_ledger": progress_ledger, "current_worker_output": None}

    except Exception:
        # Fallback
        return {**state, "progress_ledger": {**progress_ledger, "last_worker": "STALL"}}

# -----------------------------
# 4) Graph Construction (Routing)
# -----------------------------

def router(state: MagenticState) -> Literal["outer_planner", "member_id_lookup", "policy_lookup", "policy_details_lookup", "metadata_lookup", "calculator", "writer", "end"]:
    """
    Decides where to go based on Inner Manager's decision.
    """
    progress_ledger = state.get( 'progress_ledger' )
    decision = progress_ledger.get("last_worker")

    # 1. Stall / Re-plan Trigger
    if decision == "STALL" or progress_ledger.get( 'stall_count', 0 ) > 2:
        return "outer_planner" # Go back to Outer Loop

    # 2. Completion
    if decision == "FINALIZE":
        return "end"

    # 3. Delegation
    valid_workers = {
        "member_id_lookup", "policy_lookup", "policy_details_lookup", 
        "metadata_lookup", "calculator", "writer"
    }
    
    if decision in valid_workers:
        return decision # Go to specific worker
    
    return "outer_planner" # Fallback to re-plan if unsure

def build_magentic_graph():
    # Define the Graph
    g = StateGraph(MagenticState)

    # Nodes
    g.add_node("outer_planner", outer_planner_node)
    g.add_node("inner_manager", inner_manager_node)

    g.add_node("member_id_lookup", member_id_lookup_agent_node)
    g.add_node("policy_lookup", policy_lookup_node)
    g.add_node("policy_details_lookup", policy_details_node)
    g.add_node("metadata_lookup", metadata_node)
    g.add_node("calculator", calculator_node)
    g.add_node("writer", response_writer_node)

    # Edges

    # Start -> Outer Loop (Initialize Plan)
    g.add_edge(START, "outer_planner")

    # Outer Loop -> Inner Loop (Start Executing)
    g.add_edge("outer_planner", "inner_manager")

    # Inner Loop -> Router (Worker | Stall | End)
    g.add_conditional_edges(
        "inner_manager",
        router,
        {
            "outer_planner": "outer_planner", # STALL detected, re-plan
            "member_id_lookup": "member_id_lookup",
            "policy_lookup": "policy_lookup",
            "policy_details_lookup": "policy_details_lookup",
            "metadata_lookup": "metadata_lookup",
            "calculator": "calculator",
            "writer": "writer",
            "end": END
        }
    )

    # Workers -> Back to Inner Loop (Report results)
    for worker in [ 
                    "member_id_lookup", 
                    "policy_lookup", 
                    "policy_details_lookup",
                    "metadata_lookup", 
                    "calculator", 
                    "writer"
                  ]:
        g.add_edge( worker, "inner_manager" )

    checkpointer=InMemorySaver()
    return g.compile(checkpointer=InMemorySaver())