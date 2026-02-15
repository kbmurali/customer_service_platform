"""
Tree-of-Thoughts (ToT) demo using LangChain + LangGraph.

Scenario: Member service agent answering "How much will I pay for a doctor visit?"
We intentionally allow ambiguity (e.g., "total payment" could mean member OOP vs total billed),
and ToT explores multiple candidate reasoning branches, scores them, then executes the best one.

Requirements (typical):
  pip install langchain langchain-openai langgraph langchain-community chromadb python-dotenv numexpr

Environment:
  export OPENAI_API_KEY="..."
  (optional) export LANGSMITH_API_KEY="..." etc if you want tracing
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_member_policy_id, get_policy_details, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import logging
from typing import Dict, Optional, TypedDict, List, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.runnables import RunnableConfig

import json
from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
# -----------------------------
# 0) Basic SetUp
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TOOLS: Dict[str, Tool] = { 
                            t.name : wrap_tool( t ) 
                            for t in [ 
                                      get_member_id,
                                      get_member_policy_id,
                                      get_policy_details,
                                      calculator 
                                    ] 
                        }

#%%
# -----------------------------
# 1) Tree-of-Thoughts State
# -----------------------------
class Thought(TypedDict):
    title: str
    reasoning: str
    expression: Optional[str]  # math expression to compute (if needed)
    expected_output: str       # what the thought intends to answer
    score: Optional[float]     # filled during evaluation


class ToTState(TypedDict, total=False):
    user_question: str
    member_id: str
    policy_id: str
    policy_details: str

    thoughts: List[Thought]
    chosen_thought: Thought

    error: str
    computed_value: str
    final_answer: str

#%%
# -----------------------------
# 2) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a member service representative. "
        "You must base answers strictly on the provided policy details. "
        "If you compute a number, you must provide a correct math expression. "
        "In this app, we use Tree-of-Thoughts: generate multiple candidate solution plans, score them, "
        "choose the best, then execute it."
    )
)

#%%
# -----------------------------
# 3) ToT Graph Nodes
# -----------------------------

#%%
# Agent A: Intake / ID Lookup Agent
# A small ReAct agent is used even though a Tool only agent could be sufficient 
# to demonstrate that other full-fledged agents can be integrated.
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

def member_id_lookup_agent_node(state: ToTState, config: RunnableConfig ) -> ToTState:
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
        return {**state, "error": err}

    return {**state, "member_id": text}

#%%
# Policy id look up node - Tool only
def policy_lookup_node(state: ToTState) -> ToTState:
    if state.get('policy_id'):
        return state
    member_id = state.get('member_id')
    if not member_id:
        return {**state, "error": "Missing member_id before policy lookup."}

    try:
        policy_id = TOOLS["get_member_policy_id"].invoke({"member_id": member_id})
        err = _tool_error_guard(policy_id)
        if err:
            return {**state, "error": err}
        return {**state, "policy_id": str(policy_id).strip()}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in get_member_policy_id: {e}"}

#%%
# Policy details look up node - Tool only
def policy_details_node(state: ToTState) -> ToTState[str, Any]:
    if state.get('policy_details'):
        return state
    policy_id = state.get('policy_id')
    if not policy_id:
        return {**state, "error": "Missing policy_id before policy details retrieval."}

    try:
        details = TOOLS["get_policy_details"].invoke({"policy_id": policy_id})
        err = _tool_error_guard(details)
        if err:
            return {**state, "error": err}
        return {**state, "policy_details": str(details)}
    except Exception as e:
        return {**state, "error": f"{ERROR_PREFIX} in get_policy_details: {e}"}

#%%
# Generate thoughts node - No Tools - LLM only
def generate_thoughts_node(state: ToTState) -> Dict[str, Any]:
    """
    Create multiple candidate "thoughts" (branches).
    Each thought includes a plan and optionally a math expression.
    """
    prompt = (
        "Given:\n"
        f"- User question: {state.get( 'user_question' )}\n"
        f"- Policy details: {state.get( 'policy_details' )}\n\n"
        "Generate 3-5 candidate solution thoughts (different interpretations/plans). "
        "Some may be wrong; that's okay. "
        "Return STRICT JSON array, each item with keys:\n"
        "title, reasoning, expression (or null), expected_output.\n"
        "Do not prepend json qualifier to the JSON.\n"
        "Important: If a thought claims a numeric result, include a math expression for calculator.\n"
        "Important: Do not include currency signs or other non-math signs in a math expression for calculator.\n"
    )

    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
    raw = msg.content.strip()

    try:
        thoughts = json.loads(raw)
        assert isinstance(thoughts, list)
    except Exception as e:
        raise ValueError( f"No thoughts were generated for {state.get( 'member_id' )} : {state.get( 'user_question' )}" ) from e

    # Ensure score key exists
    for t in thoughts:
        t["score"] = None

    logger.info( f"LLM Generated -------> {len(thoughts)} thoughts to address the user question." )
    
    return {"thoughts": thoughts}

def evaluate_thoughts_node(state: ToTState) -> ToTState:
    """
    Score each thought. Higher score means better alignment with policy + question.
    """
    thoughts = state.get( "thoughts" )
    
    prompt = (
        "Score each candidate thought from 0.0 to 10.0 based on:\n"
        "- Matches policy wording (especially definitions)\n"
        "- Correctness of computation plan\n"
        "- Directly answers the user's question\n\n"
        f"Policy details:\n{state.get( 'policy_details' )}\n\n"
        f"User question:\n{state.get( 'user_question' )}\n\n"
        "Return STRICT JSON array of numbers, same length/order as the thoughts.\n"
        "Do not prepend json qualifier to the JSON.\n"
        f"Thoughts:\n{json.dumps(thoughts, indent=2)}\n"
    )

    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
    raw = msg.content.strip()

    try:
        scores = json.loads(raw)
        assert isinstance(scores, list) and len(scores) == len(thoughts)
        scores = [float(s) for s in scores]
    except Exception as e:
        raise ValueError( f"Thoughts could not be scored for {state.get( 'member_id' )} : {state.get( 'user_question' )}" ) from e

    for t, s in zip(thoughts, scores):
        t["score"] = s

    return {"thoughts": thoughts}

#%%
# Choose best thought node - Simple impl
def choose_best_thought_node(state: ToTState) -> ToTState:
    thoughts = state.get( "thoughts" )
    chosen_thought = max(thoughts, key=lambda t: float(t.get("score") or 0.0))
    return {"chosen_thought": chosen_thought}

#%%
# Maybe Compute node - Tool only
def maybe_compute_node(state: ToTState) -> ToTState:
    """
    If the chosen thought includes an expression, compute it.
    We also allow the model to *correct* the expression if needed.
    """
    chosen_thought: Thought = state.get( "chosen_thought" )
    expr = chosen_thought.get("expression")

    if not expr:
        return {"computed_value": ""}

    # Ask LLM to ensure expression matches the policy details (guards against hallucinated numbers).
    correction_prompt = (
        "Check whether this expression matches the policy details. "
        "If it uses wrong numbers, rewrite it correctly.\n\n"
        f"Policy details:\n{state.get( 'policy_details' )}\n\n"
        f"Proposed expression:\n{expr}\n\n"
        "Return STRICT JSON: {\"expression\": \"...\"}\n"
        "Do not prepend json qualifier to the JSON.\n"
    )
    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=correction_prompt)])
    raw = msg.content.strip()

    try:
        expr_obj = json.loads(raw)
        expr = expr_obj["expression"]
    except Exception:
        pass

    computed_value = TOOLS["calculator"].invoke({"expression": expr})
    err = _tool_error_guard( computed_value if isinstance( computed_value, str ) else "" )
    if err:
        return {"error": err}
        
    return {"computed_value": computed_value, "chosen_thought": {**chosen_thought, "expression": expr}}

#%%
# Final response node - No Tools - LLM only
def final_response_node(state: ToTState) -> Dict[str, Any]:
    prompt = (
        "Write the final answer to the user. Requirements:\n"
        "- Base strictly on policy details.\n"
        "- Be clear and beginner-friendly.\n"
        "- If you computed a number, show the formula and the result.\n\n"
    )
    
    if state.get('error'):
        prompt += (
            f"User question: {state.get( 'user_question' )}\n"
            f"Error:\n{state.get('error')}"
        )
    else:
        chosen = state.get( 'chosen_thought' )
        computed_value = state.get( 'computed_value', "")
        
        prompt += (
            f"User question: {state.get( 'user_question' )}\n"
            f"Policy details: {state.get( 'policy_details' )}\n"
            f"Chosen thought: {json.dumps(chosen, indent=2)}\n"
            f"Computed value (if any): {computed_value}\n"
        )
        
    msg = llm.invoke([SYSTEM_PROMPT, HumanMessage(content=prompt)])
    
    return {"final_answer": msg.content}

#%%
# -----------------------------
# 4) Build the LangGraph ToT workflow
# -----------------------------
def stop_if_error(state: ToTState) -> str:
    return "stop" if state.get('error') else "continue"

def build_tot_graph():
    g = StateGraph(ToTState)

    g.add_node("fetch_member_id", member_id_lookup_agent_node)
    g.add_node("fetch_policy_id", policy_lookup_node)
    g.add_node("fetch_policy_details", policy_details_node)

    g.add_node("generate_thoughts", generate_thoughts_node)
    g.add_node("evaluate_thoughts", evaluate_thoughts_node)
    g.add_node("choose_best", choose_best_thought_node)
    g.add_node("maybe_compute", maybe_compute_node)
    g.add_node("compose_answer", final_response_node)

    # Linear info retrieval
    g.set_entry_point("fetch_member_id")
    g.add_conditional_edges("fetch_member_id", stop_if_error, {"stop": "compose_answer", "continue": "fetch_policy_id"})
    g.add_conditional_edges("fetch_policy_id", stop_if_error, {"stop": "compose_answer", "continue": "fetch_policy_details"})
    g.add_conditional_edges("fetch_policy_details", stop_if_error, {"stop": "compose_answer", "continue": "generate_thoughts"})

    # ToT steps
    g.add_edge("generate_thoughts", "evaluate_thoughts")
    g.add_edge("evaluate_thoughts", "choose_best")
    g.add_edge("choose_best", "maybe_compute")
    g.add_edge("maybe_compute", "compose_answer")
    g.add_edge("compose_answer", END)

    # Memory/checkpointing
    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)
