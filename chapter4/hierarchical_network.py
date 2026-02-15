"""
Hierarchical Network (LangGraph) demo example.

Pattern:
- Root Supervisor routes to "teams" (subgraphs).
- Each team is itself a mini supervisor-driven network:
    TeamSupervisor -> Worker -> TeamSupervisor ... until TeamSupervisor outputs "done".
- Root supervisor loops until it decides we have enough information, then calls final explainer.

Scenario:
- User asks insurance questions that might involve:
  * Policy info (policy id, policy details)
  * Claim status / claim history
  * Cost estimate (based on policy + optionally claim context)
- Root decides which team(s) to run.

Libraries used:
- LangChain: LLM + tools + vector store
- LangGraph: graph/state machine to orchestrate the network
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_enhanced_member_policy_id, get_enhanced_policy_details, get_latest_claim, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import re
import json
import logging
from typing import Optional, Dict, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

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
                                      get_latest_claim,
                                      calculator, 
                                    ] 
                        }

#%%
# -----------------------------------------------------------------------------
# 1) Shared Root State
# -----------------------------------------------------------------------------
TeamName = Literal["policy_team", "claims_team", "final_explainer", "final"]

class RootState(TypedDict, total=False):
    # conversation input
    user_question: str

    # common identifiers
    member_id: Optional[str]
    policy_id: Optional[str]

    # team outputs (collected)
    policy_details: Optional[str]
    latest_claim: Optional[str]

    # optional estimate
    visit_type: Optional[str]
    calc_expression: Optional[str]
    calc_result: Optional[str]

    # routing
    next: TeamName
    routing_reason: str

    # output
    final_answer: str

    # error
    error: str

#%%
# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

# -----------------------------------------------------------------------------
# 3) POLICY TEAM (subgraph): TeamSupervisor -> workers -> done
# -----------------------------------------------------------------------------
PolicyNext = Literal["member_id_worker", "policy_id_worker", "policy_details_worker", "done"]

class PolicyTeamState(TypedDict, total=False):
    user_question: str
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]
    next: PolicyNext
    routing_reason: str
    error: str

#%%
# Policy Worker: Member ID Lookup Worker
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

def member_id_lookup_agent_node(state: PolicyTeamState, config: RunnableConfig ) -> PolicyTeamState:
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
# Policy Worker: Policy ID Lookup Worker
policy_id_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_enhanced_member_policy_id"]],
    prompt=(
        "You are PolicyTeam.PolicyIDWorker.\n"
        "Given member_id, call the tool and return ONLY the policy_id.\n"
        f"If tool returns '{ERROR_PREFIX}...', output that error."
    ),
)

def policy_id_lookup_agent_node(state: PolicyTeamState, config: RunnableConfig) -> PolicyTeamState:
    if state.get("policy_id"):
        return state
    
    mid = state.get("member_id")
    if not mid:
        return {**state, "error": "Missing member_id in PolicyTeam before policy id lookup."}
    
    result = policy_id_lookup_agent.invoke({"messages": [HumanMessage(content=f"member_id={mid}")]}, config=config)
    
    text = result["messages"][-1].content.strip()
    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}
    
    return {**state, "policy_id": text}

#%% 
# Policy Worker: Policy Details Lookup Worker
policy_details_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_enhanced_policy_details"]],
    prompt=(
        "You are PolicyTeam.PolicyDetailsWorker.\n"
        "Given policy_id, call the tool and return ONLY the policy details text.\n"
        f"If tool returns '{ERROR_PREFIX}...', output that error."
    ),
)

def policy_details_lookup_agent_node(state: PolicyTeamState, config: RunnableConfig) -> PolicyTeamState:
    if state.get("policy_details"):
        return state
    policy_id = state.get("policy_id")
    if not policy_id:
        return {**state, "error": "Missing policy_id in PolicyTeam before policy details fetch."}
    
    result = policy_details_lookup_agent.invoke({"messages": [HumanMessage(content=f"policy_id={policy_id}")]}, config=config)
    text = result["messages"][-1].content.strip()
    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}
    
    return {**state, "policy_details": text}

#%%
# Policy Team Supervisor
policy_team_supervisor_prompt = """
You are PolicyTeamSupervisor. Decide which policy worker to run next.

Workers:
- member_id_worker
- policy_id_worker (needs member_id)
- policy_details_worker (needs policy_id)
- done

Rules:
1) If error exists -> done
2) If member_id missing -> member_id_worker
3) If policy_id missing -> policy_id_worker
4) If policy_details missing -> policy_details_worker
5) Else -> done

- Provide reason for determining the next worker.
Return ONLY JSON (no code fences, no extra text): {"next": "...", "reason": "..."}
"""

def policy_team_supervisor_node(state: PolicyTeamState, config: RunnableConfig) -> PolicyTeamState:
    snapshot = {
        "member_id": state.get("member_id"),
        "policy_id": state.get("policy_id"),
        "policy_details": state.get("policy_details"),
        "error": state.get("error"),
    }
    raw = llm.invoke(
        [SystemMessage(content=policy_team_supervisor_prompt), HumanMessage(content=json.dumps(snapshot))],
        config=config,
    ).content.strip()
    
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"next": "done", "reason": "JSON parse failed; stopping team."}

    next_worker = decision.get("next", "done")
    
    if next_worker not in {"member_id_worker", "policy_id_worker", "policy_details_worker", "done"}:
        next_worker = "done"
        
    return {**state, "next": next_worker, "routing_reason": decision.get("reason", "")}

#%%
# -----------------------------
# 4) PolicyTeam LangGraph
# -----------------------------
def route_policy_team(state: PolicyTeamState) -> str:
    return state.get("next", "done")

def build_policy_team_graph():
    g = StateGraph(PolicyTeamState)
    g.add_node("team_supervisor", policy_team_supervisor_node)
    g.add_node("member_id_worker", member_id_lookup_agent_node)
    g.add_node("policy_id_worker", policy_id_lookup_agent_node)
    g.add_node("policy_details_worker", policy_details_lookup_agent_node)

    g.set_entry_point("team_supervisor")

    g.add_conditional_edges(
        "team_supervisor",
        route_policy_team,
        {
            "member_id_worker": "member_id_worker",
            "policy_id_worker": "policy_id_worker",
            "policy_details_worker": "policy_details_worker",
            "done": END,
        },
    )

    for worker in ["member_id_worker", "policy_id_worker", "policy_details_worker"]:
        g.add_edge(worker, "team_supervisor")

    checkpointer=InMemorySaver()
    return g.compile(checkpointer=checkpointer)


policy_team_app = build_policy_team_graph()

#%%
# -----------------------------------------------------------------------------
# 5) CLAIMS TEAM (subgraph): TeamSupervisor -> workers -> done
# -----------------------------------------------------------------------------
ClaimsNext = Literal["member_id_worker", "latest_claim_worker", "done"]

class ClaimsTeamState(TypedDict, total=False):
    user_question: str
    member_id: Optional[str]
    latest_claim: Optional[str]
    next: ClaimsNext
    routing_reason: str
    error: str
    
#%%
# Claims Worker: Member ID Lookup Worker
# A small ReAct agent is used even though a Tool only agent could be sufficient 
# to demonstrate that other full-fledged agents can be integrated.

def claims_member_id_lookup_agent_node(state: ClaimsTeamState, config: RunnableConfig ) -> ClaimsTeamState:
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
# Claims Worker: Latest Claim Lookup Worker
claims_latest_lookup_agent = create_react_agent(
    model=llm,
    tools=[TOOLS["get_latest_claim"]],
    prompt=(
        "You are ClaimsTeam.LatestClaimWorker.\n"
        "Given member_id, call the tool and return ONLY the claim text.\n"
        f"If tool returns '{ERROR_PREFIX}...', output that error."
    ),
)

def claims_latest_lookup_agent_node(state: ClaimsTeamState, config: RunnableConfig) -> ClaimsTeamState:
    if state.get("latest_claim"):
        return state
    mid = state.get("member_id")
    if not mid:
        return {**state, "error": "Missing member_id in ClaimsTeam before claim lookup."}
    result = claims_latest_lookup_agent.invoke({"messages": [HumanMessage(content=f"member_id={mid}")]}, config=config)
    text = result["messages"][-1].content.strip()
    err = _tool_error_guard(text)
    if err:
        return {**state, "error": err}
    return {**state, "latest_claim": text}

#%%
# Claims Team Supervisor
claims_team_supervisor_prompt = """
You are ClaimsTeamSupervisor. Decide which claims worker to run next.

Workers:
- member_id_worker
- latest_claim_worker (needs member_id)
- done

Rules:
1) If error exists -> done
2) If member_id missing -> member_id_worker
3) If latest_claim missing -> latest_claim_worker
4) Else -> done

- Provide reason for determining the next worker.
Return ONLY JSON (no code fences, no extra text): {"next": "...", "reason": "..."}
"""

def claims_team_supervisor_node(state: ClaimsTeamState, config: RunnableConfig) -> ClaimsTeamState:
    snapshot = {
        "member_id": state.get("member_id"),
        "latest_claim_present": bool(state.get("latest_claim")),
        "error": state.get("error"),
    }
    raw = llm.invoke(
        [SystemMessage(content=claims_team_supervisor_prompt), HumanMessage(content=json.dumps(snapshot))],
        config=config,
    ).content.strip()
    
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    
    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"next": "done", "reason": "JSON parse failed; stopping team."}

    next_worker = decision.get("next", "done")
    if next_worker not in {"member_id_worker", "latest_claim_worker", "done"}:
        next_worker = "done"
        
    return {**state, "next": next_worker, "routing_reason": decision.get("reason", "")}

#%%
# -----------------------------
# 6) ClaimsTeam LangGraph
# -----------------------------
def route_claims_team(state: ClaimsTeamState) -> str:
    return state.get("next", "done")

def build_claims_team_graph():
    g = StateGraph(ClaimsTeamState)
    g.add_node("team_supervisor", claims_team_supervisor_node)
    g.add_node("member_id_worker", claims_member_id_lookup_agent_node)
    g.add_node("latest_claim_worker", claims_latest_lookup_agent_node)

    g.set_entry_point("team_supervisor")

    g.add_conditional_edges(
        "team_supervisor",
        route_claims_team,
        {
            "member_id_worker": "member_id_worker", 
            "latest_claim_worker": "latest_claim_worker", 
            "done": END
        },
    )

    for worker in ["member_id_worker", "latest_claim_worker"]:
        g.add_edge(worker, "team_supervisor")

    checkpointer=InMemorySaver()
    return g.compile(checkpointer=checkpointer)


claims_team_app = build_claims_team_graph()

#%%
# -----------------------------------------------------------------------------
# 7) Root-level Workers
# -----------------------------------------------------------------------------
#%%
# Root-level Estimator: (1) decide visit_type, (2) build expression, (3) calculate
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

def estimator_agent_node(state: RootState, config: RunnableConfig) -> RootState:
    if state.get("calc_result"):
        return state
    if not state.get("policy_details"):
        return {**state, "error": "Missing policy_details before estimation."}

    human = HumanMessage(
        content=(
            f"User question:\n{state.get('user_question','')}\n\n"
            f"Policy details:\n{state.get('policy_details','')}\n\n"
            f"Latest claim (optional):\n{state.get('latest_claim','')}\n"
        )
    )
    result = estimator_agent.invoke({"messages": [human]}, config=config)
    raw = result["messages"][-1].content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(raw)
    except Exception as e:
        return {**state, "error": f"Estimator returned invalid JSON: {e}. Raw={raw[:200]}"}

    visit_type = (payload.get("visit_type") or "primary").strip().lower()
    expr = (payload.get("calc_expression") or "").strip()
    if not expr:
        return {**state, "error": "Estimator did not provide calc_expression."}

    calc_result = TOOLS["calculator"].invoke(expr)
    err = _tool_error_guard(calc_result)
    if err:
        return {**state, "error": err}

    return {**state, "visit_type": visit_type, "calc_expression": expr, "calc_result": calc_result}

#%%
# Root-level Explainer: Generates final response
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

def explainer_agent_node(state: RootState, config: RunnableConfig) -> RootState:
    parts = [
        f"User question:\n{state.get('user_question','')}",
        f"Member ID: {state.get('member_id')}",
        f"Policy ID: {state.get('policy_id')}",
        f"Policy details:\n{state.get('policy_details','')}",
        f"Latest claim:\n{state.get('latest_claim','')}",
        f"Estimated visit_type: {state.get('visit_type')}",
        f"Estimated expression: {state.get('calc_expression')}",
        f"Estimated result: {state.get('calc_result')}",
        f"Error: {state.get('error')}"
    ]
    
    human_message = HumanMessage(content="\n\n".join([p for p in parts if p and "None" not in p]))
    
    result = explainer_agent.invoke({"messages": [human_message]}, config=config)
    answer = result["messages"][-1].content.strip()
    return {**state, "final_answer": answer}

#%%
# -----------------------------------------------------------------------------
# 8) Root Supervisor (routes to teams and finalization)
# -----------------------------------------------------------------------------
root_supervisor_prompt = """
You are the RootSupervisor of a Hierarchical Network.

You can route to:
- policy_team: when policy_id/details are needed OR user asks about coverage/fees/copays/deductible.
- claims_team: when user asks about claim status, claims history, or mentions a claim id.
- estimator: when user asks about payment or coverage costs, and policy details are already present
- final_explainer: when we have enough info to answer, or if error exists.

Rules (high-level):
1) If error exists -> next=final_explainer
2) If the user asks about claims/status/paid/pending/claim -> If latest claim is present, next=final_explainer, ELSE next=claims_team (needs latest_claim)
3) If the user asks about coverage/fees/copay/deductible/policy id OR cost estimate -> If policy details are present, next=final_explainer, ELSE next=policy_team (needs policy_details)
4) If the user asks about payment or coverage costs -> If policy details are present, next=final_explainer, ELSE next=policy_team
5) Otherwise -> final_explainer

Return ONLY JSON: {"next":"...","reason":"..."}.
Return ONLY valid JSON (no code fences, no extra text).
"""

def root_supervisor_node(state: RootState, config: RunnableConfig) -> RootState:
    if state.get("final_answer"):
        return {**state, "next": "final", "routing_reason": "final_answer already present"}

    have_estimate = bool(state.get("calc_result"))

    snapshot = {
        "user_question": state.get("user_question"),
        "member_id": state.get("member_id"),
        "policy_id": state.get("policy_id"),
        "policy_details_present": bool(state.get("policy_details")),
        "latest_claim_present": bool(state.get("latest_claim")),
        "have_estimate": have_estimate,
        "error": state.get("error"),
    }

    raw = llm.invoke(
        [SystemMessage(content=root_supervisor_prompt), HumanMessage(content=json.dumps(snapshot, indent=2))],
        config=config,
    ).content.strip()

    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"next": "final_explainer", "reason": "JSON parse failed; defaulting to final_explainer"}

    next_route = decision.get("next", "final_explainer")
    if next_route not in {"policy_team", "claims_team", "estimator", "final_explainer", "final"}:
        next_route = "final_explainer"
        reason = f"Invalid next; clamped. decision={decision}"
    else:
        reason = decision.get("reason", "")

    return {**state, "next": next_route, "routing_reason": reason}

#%%
# -----------------------------------------------------------------------------
# 9) Root Graph Assembly (Hierarchy!)
# -----------------------------------------------------------------------------
def policy_team_node(state: RootState, config: RunnableConfig) -> RootState:
    """Run policy subgraph, then merge its results into RootState."""
    team_in: PolicyTeamState = {
        "user_question": state.get("user_question", ""),
        "member_id": state.get("member_id"),
        "policy_id": state.get("policy_id"),
        "policy_details": state.get("policy_details"),
    }
    
    team_out = policy_team_app.invoke(team_in, config=config)
    
    merged: RootState = {
        **state,
        "member_id": team_out.get("member_id", state.get("member_id")),
        "policy_id": team_out.get("policy_id", state.get("policy_id")),
        "policy_details": team_out.get("policy_details", state.get("policy_details")),
    }
    
    if team_out.get("error"):
        merged["error"] = team_out["error"]
        
    return merged

def claims_team_node(state: RootState, config: RunnableConfig) -> RootState:
    """Run claims subgraph, then merge its results into RootState."""
    team_in: ClaimsTeamState = {
        "user_question": state.get("user_question", ""),
        "member_id": state.get("member_id"),
        "latest_claim": state.get("latest_claim"),
    }
    
    team_out = claims_team_app.invoke(team_in, config=config)
    
    merged: RootState = {
        **state,
        "member_id": team_out.get("member_id", state.get("member_id")),
        "latest_claim": team_out.get("latest_claim", state.get("latest_claim")),
    }
    
    if team_out.get("error"):
        merged["error"] = team_out["error"]
        
    return merged

def route_root(state: RootState) -> str:
    return state.get("next", "final_explainer")

def build_root_sup_graph():
    g = StateGraph(RootState)

    g.add_node("root_supervisor", root_supervisor_node)
    g.add_node("policy_team", policy_team_node)
    g.add_node("claims_team", claims_team_node)
    g.add_node("estimator", estimator_agent_node)

    g.add_node("final_explainer", explainer_agent_node)

    g.set_entry_point("root_supervisor")

    g.add_conditional_edges(
        "root_supervisor",
        route_root,
        {
            "policy_team": "policy_team",
            "claims_team": "claims_team",
            "estimator": "estimator",
            "final_explainer": "final_explainer",
            "final": END,
        },
    )

    # After team runs, go to maybe_estimate then back to root supervisor
    g.add_edge("policy_team", "root_supervisor")
    g.add_edge("claims_team", "root_supervisor")
    g.add_edge("estimator", "root_supervisor")

    # After final explainer, go back to root supervisor which will terminate
    g.add_edge("final_explainer", "root_supervisor")

    checkpointer=InMemorySaver()
    return g.compile(checkpointer=checkpointer)