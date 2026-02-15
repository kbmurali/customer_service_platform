"""
Supervisor-Driven Network (LangGraph) demo.

Supervisor-Driven Network: a central Supervisor decides which specialized worker 
agent should act next, based on the current state of the conversation.

Scenario:
- User asks questions about an insurance policy.
- The system may need to:
    1) obtain member_id
    2) look up policy_id
    3) fetch policy details
    4) (optionally) compute an estimate for a visit or procedure
    5) explain the result / provide a clear final answer

Key difference:
- Instead of a hard-coded linear chain, the Supervisor dynamically routes to workers.

Requirements:
- LangChain + LangGraph
- OpenAI API key via environment variables
- Optional LangSmith for tracing

Notes for beginners:
- The "Supervisor" is just an LLM prompt that returns a small JSON decision:
    {"next": "<worker_name>", "reason": "..."}
- Each worker is a small ReAct agent with a limited toolset.
- The graph loops: Supervisor -> Worker -> Supervisor ... until Supervisor chooses "final".
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_enhanced_member_policy_id, get_enhanced_policy_details, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import re
import json
import logging
from typing import Dict, Optional, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.runnables import RunnableConfig

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
                                      get_enhanced_member_policy_id,
                                      get_enhanced_policy_details,
                                      calculator ] 
                        }

# ---------------------------------------------------------------------------
# 1) Shared graph state
# ---------------------------------------------------------------------------
WorkerName = Literal[
    "member_id_worker",
    "policy_lookup_worker",
    "policy_details_worker",
    "estimator_worker",
    "explainer_worker",
    "final",
]

class SupervisorState(TypedDict, total=False):
    # conversation
    user_question: str

    # stored state
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]

    # estimation fields
    visit_type: Optional[str]  # "primary" or "specialist"
    calc_expression: Optional[str]
    calc_result: Optional[str]

    # supervisor decision
    next: WorkerName
    routing_reason: str

    # output
    final_answer: str

    # error
    error: str
    
#%%
# -----------------------------
# 2) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

# -------------------------------
# 3) Worker agents (specialized)
# -------------------------------
#%%
# Agent A: Intake / ID Lookup Agent (A small ReAct agent that only knows member id lookup tool)
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

def member_id_lookup_agent_node(state: SupervisorState, config: RunnableConfig ) -> SupervisorState:
    if state.get("member_id"):
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
# Agent B: Policy Lookup Agent (A small ReAct agent that only knows policy lookup tool)
policy_lookup_agent_prompt = (
    "You are PolicyLookupAgent. Your ONLY job is to return the policy_id for the given member_id.\n"
    "Use the tool provided.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', output that error.\n"
    "Return ONLY the policy_id string."
)

policy_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_enhanced_member_policy_id"]],
    prompt=policy_lookup_agent_prompt,
)

def policy_lookup_agent_node( state: SupervisorState, config: RunnableConfig ) -> SupervisorState:
    member_id = state.get("member_id")
    if not member_id:
        return {**state, "error": "Missing member_id before policy lookup."}
    
    if state.get( 'policy_id' ):
        return state

    result = policy_lookup_agent.invoke(
        {"messages": [HumanMessage(content=f"member_id={member_id}. Get the policy_id.")]},
        config=config
    )
    
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}

    return {**state, "policy_id": text}

#%%
# Agent C: Policy Details Retrieval Agent (A small ReAct agent that only knows policy details lookup tool)
policy_details_agent_prompt = (
    "You are PolicyDetailsAgent. Your ONLY job is to fetch policy details text for a policy_id.\n"
    "Use the tool provided.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', output that error.\n"
    "Return ONLY the raw policy details text."
)

policy_details_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_enhanced_policy_details"]],
    prompt=policy_details_agent_prompt,
)

def policy_details_agent_node( state: SupervisorState, config: RunnableConfig ) -> SupervisorState:
    policy_id = state.get("policy_id")
    if not policy_id:
        return {**state, "error": "Missing policy_id before policy details retrieval."}
    
    if state.get( 'policy_details' ):
        return state

    result = policy_details_agent.invoke(
        {"messages": [HumanMessage(content=f"policy_id={policy_id}. Fetch policy details.")]},
        config=config
    )
    text = result["messages"][-1].content.strip()

    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}

    return {**state, "policy_details": text}

#%%
# Agent D: Estimator: (1) decide visit_type, (2) build expression, (3) calculate
estimator_agent_prompt = (
    "You are EstimatorAgent. You estimate what the member pays for a visit.\n"
    "You will be given a user question and policy details.\n\n"
    "Pick visit_type as either 'primary' or 'specialist'. If unclear, default to 'primary'.\n"
    "Then produce a calculator expression that uses ONLY numbers, no $ signs.\n"
    "Example Expression format:\n"
    "  copay + (member_portion_pct * visit_fee) + deductible_if_applicable\n\n"
    "- If the visit is specialist AND the policy says deductible applies to specialist visits,\n"
    "  add the deductible amount.\n\n"
    "Important: Do not include currency signs or other non-math signs in a math expression for calculator.\n"
    "Important: Return ONLY JSON with keys: visit_type, calc_expression.\n"
    "Important: Do not prepend json qualifier to the JSON.\n"
)
estimator_agent = create_react_agent(model=llm, tools=[], prompt=estimator_agent_prompt)

def estimator_agent_node( state: SupervisorState, config: RunnableConfig ) -> SupervisorState:
    if state.get("calc_result"):
        return state
    
    if not state.get("policy_details"):
        return {**state, "error": "Missing policy_details before estimation."}

    human_message = HumanMessage(
        content=(
            f"User question:\n{state.get( 'user_question', '' )}\n\n"
            f"Policy details:\n{state.get( 'policy_details','' )}\n"
        )
    )
    result = estimator_agent.invoke({"messages": [human_message]}, config=config)
    
    raw = result["messages"][-1].content.strip()
    
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(raw)
    except Exception as e:
        return {**state, "error": f"EstimatorAgent returned invalid JSON: {e}. Raw: {raw[:200]}"}

    visit_type = (payload.get("visit_type") or "primary").strip().lower()
    
    expr = (payload.get("calc_expression") or "").strip()
    
    if not expr:
        return {**state, "error": "EstimatorAgent did not provide calc_expression."}

    calc_tool = TOOLS["calculator"]
    
    calc_result = calc_tool.invoke( expr )
    
    err = _tool_error_guard( calc_result )
    
    if err:
        return {**state, "error": err}

    return {**state, "visit_type": visit_type, "calc_expression": expr, "calc_result": calc_result}

#%%
# Agent E: ExplainerAgent: Generates final response
explainer_agent_prompt = (
    "You are ExplainerWorker. Explain the answer clearly.\n"
    "Use ONLY provided policy text and computed values. Do not invent policy terms.\n"
    "Do not perform any calculations.\n"
    "If calculations are already pereformed, use their explanations to summarize.\n"
    "If something is missing, say what is missing.\n"
)

explainer_agent = create_react_agent(
    model=llm, 
    tools=[], 
    prompt=explainer_agent_prompt
)

def explainer_agent_node( state: SupervisorState, config: RunnableConfig ) -> SupervisorState:
    content = (
        f"User question:\n{state.get( 'user_question','' )}\n\n"
        f"Member ID: {state.get( 'member_id' )}\n"
        f"Policy ID: {state.get( 'policy_id' )}\n\n"
        f"Policy evidence:\n{state.get( 'policy_details','' )}\n\n"
    )
    
    if state.get( 'calc_result' ):
        content += (
            "Computation:\n"
            f"- visit_type: {state.get( 'visit_type' )}\n"
            f"- expression: {state.get( 'calc_expression' )}\n"
            f"- result: {state.get( 'calc_result' )}\n\n"
        )
        
    if state.get("error"):
        content += f"Error:\n{state.get( 'error' )}\n\n"

    result = explainer_agent.invoke(
        {"messages": [HumanMessage(content=content)]}, 
        config=config
    )
    
    answer = result["messages"][-1].content.strip()
    
    return {**state, "final_answer": answer}

#%%
# Agent F: Network Supervisor Agent (router)
# ---------------------------------------------------------------------------
supervisor_agent_system_prompt = """
You are the Supervisor in a Supervisor-Driven Network.

Choose which worker should run next based on the current state.

Workers:
- member_id_worker: obtain member_id
- policy_lookup_worker: obtain policy_id (requires member_id)
- policy_details_worker: obtain policy_details (requires policy_id)
- estimator_worker: compute an estimate (requires policy_details)
- explainer_worker: write a clear final response
- final: stop the network

Rules:
1) If error exists -> next = explainer_worker
2) If member_id missing -> next = member_id_worker
3) If policy_id missing -> next = policy_lookup_worker
4) If policy_details missing -> next = policy_details_worker
5) If user asks for cost/total/payment/how much/breakdown -> next = estimator_worker
6) Else -> next = explainer_worker

Return ONLY JSON:
{"next": "<worker_name>", "reason": "<short reason>"}\n

"Important: Do not prepend json qualifier to the JSON.\n"
"""

def supervisor_agent_node(state: SupervisorState, config: RunnableConfig ) -> SupervisorState:
    if state.get("final_answer"):
        next_route = "final"
        reason = "final_answer already present."
        return {**state, "next": next_route, "routing_reason": reason}

    system_message = SystemMessage(content=supervisor_agent_system_prompt)

    snapshot = {
        "user_question": state.get( 'user_question' ),
        "member_id": state.get( 'member_id' ),
        "policy_id": state.get( 'policy_id' ),
        "policy_details_present": bool(state.get( 'policy_details' )),
        "calc_result_present": bool(state.get( 'calc_result' )),
        "error": state.get( 'error' ),
    }
    
    human_message = HumanMessage( content=json.dumps( snapshot, indent=2 ))

    raw = llm.invoke( [system_message, human_message], config=config ).content.strip()
    
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"next": "explainer_worker", "reason": "Supervisor JSON parse failed; falling back."}

    next_route = decision.get( 'next', "explainer_worker" )
    reason = decision.get( 'reason', "")

    allowed = {
        "member_id_worker",
        "policy_lookup_worker",
        "policy_details_worker",
        "estimator_worker",
        "explainer_worker",
        "final",
    }
    
    if next_route not in allowed:
        next_route = "explainer_worker"
        reason = f"Invalid next; clamped. Raw: {decision}"

    return {**state, "next": next_route, "routing_reason": reason}


def route_from_supervisor( state: SupervisorState ) -> str:
    return state.get("next", "explainer_worker")

#%%
# -------------------------------------
# 4) Build the LangGraph (the network)
# -------------------------------------
def build_supervisor_network():
    g = StateGraph(SupervisorState)

    g.add_node( "supervisor", supervisor_agent_node )
    g.add_node( "member_id_worker", member_id_lookup_agent_node )
    g.add_node( "policy_lookup_worker", policy_lookup_agent_node )
    g.add_node( "policy_details_worker", policy_details_agent_node )
    g.add_node( "estimator_worker", estimator_agent_node )
    g.add_node( "explainer_worker", explainer_agent_node )

    g.set_entry_point( "supervisor" )

    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "member_id_worker": "member_id_worker",
            "policy_lookup_worker": "policy_lookup_worker",
            "policy_details_worker": "policy_details_worker",
            "estimator_worker": "estimator_worker",
            "explainer_worker": "explainer_worker",
            "final": END,
        },
    )

    # Each worker returns control to the supervisor
    for worker in [
        "member_id_worker",
        "policy_lookup_worker",
        "policy_details_worker",
        "estimator_worker",
        "explainer_worker",
    ]:
        g.add_edge( worker, "supervisor" )

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)