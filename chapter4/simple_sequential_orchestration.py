"""
Simple Sequential Orchestration (LangGraph) Example.

Pattern:
- A linear pipeline of specialized agents working in a strict sequence.
- Data flows from Agent A -> Agent B -> Agent C.
- Ideal for workflows with predictable stages.

Scenario Flow:
1. Member Id Lookup: Extracts Member ID from user text.
2. Policy Id Lookup: Fetches the policy id for the member id.
3. Policy Details Lookup: Fetches the policy details for the member's policy id.
4. User Query Evaluator: Evaluates the user query using the policy details and determines any calculations.
5. Calculator: Computes any mathematical expressions evaluated based on user query.
6. Summarize Response: Drafts the final response.

Libraries:
- LangChain: LLM + Prompts
- LangGraph: State orchestration
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_member_policy_id, get_policy_details, get_policy_metadata, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import re
import logging
from typing import Dict, Optional, TypedDict, Any, Annotated
from operator import add

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
                                      calculator,
                                      get_policy_metadata 
                                    ] 
                        }

#%%
# -----------------------------
# 1) Shared State
# -----------------------------
class SharedState(TypedDict, total=False):
    # conversation input
    user_question: str

    # artifacts passed step -> step
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]

    # evaluation
    evaluated_reason: str
    calc_expression: str 
    calc_result: str

    # errors / final
    errors: Annotated[list[str], add]
    final_answer: str

#%%
# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

#%%
# -----------------------------
# 3) SSO Nodes
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

def member_id_lookup_agent_node(state: SharedState, config: RunnableConfig ) -> SharedState:
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
        return {**state, "errors": [err]}

    return {**state, "member_id": text}

#%%
# Agent B (Policy lookup): tool-only, no LLM text-as-artifact
def policy_lookup_node(state: SharedState, config: RunnableConfig) -> SharedState:
    if state.get('policy_id'):
        return state
    
    member_id = state.get('member_id')
    if not member_id:
        return {**state, "errors": ["Missing member_id before policy lookup."]}

    try:
        policy_id = TOOLS["get_member_policy_id"].invoke({"member_id": member_id})
        err = _tool_error_guard(policy_id)
        if err:
            return {**state, "errors": [err]}
        return {**state, "policy_id": str(policy_id).strip()}
    except Exception as e:
        return {**state, "errors": [f"{ERROR_PREFIX} in get_member_policy_id: {e}"]}
    
#%%
# Agent C (Policy details): tool-only
def policy_details_node(state: SharedState, config: RunnableConfig) -> SharedState:
    if state.get('policy_details'):
        return state
    
    policy_id = state.get('policy_id')
    if not policy_id:
        return {**state, "errors": ["Missing policy_id before policy details retrieval."]}

    try:
        details = TOOLS["get_policy_details"].invoke({"policy_id": policy_id})
        err = _tool_error_guard(details)
        if err:
            return {**state, "errors": [err]}
        return {**state, "policy_details": str(details)}
    except Exception as e:
        return {**state, "errors": [f"{ERROR_PREFIX} in get_policy_details: {e}"]}

#%%
# Agent D (Query Evaluator): Evaluates user query and determines any calculations needed.
def query_evaluator_node(state: SharedState, config: RunnableConfig) -> SharedState:
    """
    Evaluate user query:
      - reason: can answer directly from evidence (no math needed)
      - calculate: compute total payment using parsed fields (numeric expression only)
      - clarify: missing info (e.g., member id, evidence, or required numbers)
    """

    # Hard safety: if evidence missing, clarify rather than guess
    if not state.get('policy_details'):
        return {
            **state,
            "evaluated_reason": "I couldn't retrieve your policy details yet. Can you confirm your member id or what you want to check?",
        }

    system_message = SystemMessage(
        content=(
            "You are a helpful insurance policy assistant.\n"
            "You MUST only use the provided policy evidence.\n"
            "Return ONLY valid JSON (no code fences, no extra text).\n"
            "Schema:\n"
            "  {\"calc_expression\"?: string,\n"
            "   \"evaluated_reason\"?: string }\n" 
            "Rules / guardrails:\n"
            "- If errors exist: finalize by evaluated_reason explaining the error.\n"
            "- Else If user question is about payment or billed amounts, do the following:\n"
            "  - Ensure policy details exists.\n"
            "  - Determine the math involved by applying the logic based on the policy details.\n"
            "  - Create calc_expression based on math involved.\n" 
            "  - calc_expression MUST be numeric-only (no variable names, no $ signs).\n"
            "  - Do not compute yourself but return calc_expression in the json.\n"
            "  - Provide the evaluated reason for the math expression and return as evaluated_reason in the json.\n"
            "- Else evaluate correct answer to user question and return as evaluated_reason in the json.\n"
            "- Never invent policy details; use policy details.\n\n"
        )
    )

    evidence = state.get('policy_details', "")
    user_question = state.get('user_question', "")
    errors = "\n".join( [ err for err in state.get( 'errors', [] )] )
    
    human_message = HumanMessage(
        content=(
            f"User Question:\n{user_question}\n\n"
            f"Policy Details:\n{evidence}\n\n"
            f"Errors:\n{errors}\n\n"
        )
    )

    raw = llm.invoke([system_message, human_message]).content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        query_eval: Dict[str, Any] = json.loads(raw)
    except Exception:
        # If planner output can't be parsed, do a safe clarify instead of guessing.
        return {
            **state,
            "evaluated_reason": "I couldn't determine the next step from the policy evidence. What exactly are you trying to find (policy id, copay, total payment, etc.)?",
        }

    # next_step == "calculate"
    expr = (query_eval.get("calc_expression") or "").strip()

    

    # Enforce numeric-only expression (no identifiers)
    if expr:
        if re.search(r"[A-Za-z_]\w*", expr):
            return {
                **state,
                "evaluated_reason": "To calculate, I need numeric values only. What is the billed amount or the missing number referenced in your request?",
            }

    evaluated_reason = query_eval.get(
                                        "evaluated_reason",
                                        "I can help, but I need a bit more detail about what you want to know.",
                                    )
    
    return {**state, "calc_expression": expr, "evaluated_reason": evaluated_reason}

#%%
# Agent E (Tool Only Calculator): compute numeric expression only
def calculator_node(state: SharedState, config: RunnableConfig) -> SharedState:
    expr = (state.get('calc_expression') or "").strip()
    if not expr:
        return state

    try:
        result = TOOLS["calculator"].invoke({"expression": expr})
        err = _tool_error_guard(result)
        if err:
            return {**state, "errors": [err]}
        return {**state, "calc_result": str(result)}
    except Exception as e:
        return {**state, "errors": [f"{ERROR_PREFIX} in calculator: {e}"]}
    
#%%
# Agent F (Final response writer): plain LLM call (no ReAct “agent” with empty tools)
def final_response_node(state: SharedState, config: RunnableConfig) -> SharedState:
    system_message = SystemMessage(
        content=(
            "You are a friendly insurance policy assistant.\n"
            "Use ONLY the policy evidence and computed results already present.\n"
            "If calc_result is present, explain the breakdown succinctly.\n"
            "If evaluated_reason is present, return it and cite the evidence contextually.\n"
            "If there are errors, summarize them and suggest next steps.\n"
        )
    )

    errors = "\n".join( [ err for err in state.get( 'errors', [] )] )
    
    parts = [
        f"User Question: {state.get('user_question')}",
        f"Member ID: {state.get('member_id')}",
        f"Policy ID: {state.get('policy_id')}",
        f"Policy Details: {state.get('policy_details')}",
        f"Calculation Expression: {state.get('calc_expression')}",
        f"Calculation Result: {state.get('calc_result')}",
        f"Evaluated Reason:\n{state.get('evaluated_reason')}",
        f"Errors: {errors}",
    ]
    
    content = "\n\n".join([p for p in parts if p and "None" not in p])
    
    human_message = HumanMessage(content=content)

    final = llm.invoke([system_message, human_message]).content.strip()
    
    return {**state, "final_answer": final}

#%%
# -----------------------------
# 4) Build the Graph
# -----------------------------
def build_sequential_graph():
    g = StateGraph( SharedState )

    # Add Nodes
    g.add_node("intake", member_id_lookup_agent_node)
    g.add_node("policy_lookup", policy_lookup_node)
    g.add_node("fetch_policy_details", policy_details_node)
    g.add_node("query_evaluator", query_evaluator_node)
    g.add_node("calculator", calculator_node)
    g.add_node("final_response", final_response_node)

    # Add Edges (Linear Sequence)
    # intake -> policy_lookup -> fetch_policy_details -> query_evaluator -> calculator -> final_response -> END
    g.set_entry_point("intake")
    g.add_edge("intake", "policy_lookup")
    g.add_edge("policy_lookup", "fetch_policy_details")
    g.add_edge("fetch_policy_details", "query_evaluator")
    g.add_edge("query_evaluator", "calculator")
    g.add_edge("calculator", "final_response")
    g.add_edge("final_response", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)