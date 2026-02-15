"""
Generic Hand-Off Orchestration agent for "Ask anything about my insurance policy".

Uses:
- LangChain (LLMs, tools)
- LangGraph (orchestration + explicit hand-offs)
- LangSmith (tracing)

How it works:
1) SupervisorAgent (LLM) routes the conversation:
   - identity -> if member_id missing
   - policy   -> if policy not loaded
   - qa       -> if question is informational (policy related, coverage, copay, deductible, etc.)
   - math     -> if question needs arithmetic (estimated payment, % calculations, etc.)
   - done     -> if answer already produced
2) Each specialized agent does one job then hands back to supervisor.

Beginner-friendly notes:
- This demo stores small "policy docs" in Chroma (vector DB).
- For a real system you'd load policies from your database + docs pipeline.
"""
#%%
from __future__ import annotations

from common_utils import get_member_id, get_member_policy_id, get_policy_details, search_policy_knowledge, calculator
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import json
import logging
from typing import Optional, Literal, TypedDict

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_core.tools import tool

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


TOOLS_IDENTITY = [wrap_tool(get_member_id)]
TOOLS_POLICY = [
    wrap_tool(get_member_policy_id),
    wrap_tool(get_policy_details),
]
TOOLS_RETRIEVAL = [wrap_tool(search_policy_knowledge)]
TOOLS_MATH = [wrap_tool(calculator)]

# ----------------------------
# 1) Handoff State
# ----------------------------
class HandoffState(TypedDict, total=False):
    user_question: str
    member_id: Optional[str]
    policy_id: Optional[str]
    policy_details: Optional[str]
    
    # routing + outputs
    next_handoff: Optional[Literal["identity", "policy", "math", "summary"]]
    answer: Optional[str]
    error: Optional[str]
    final_answer: Optional[str]

#%%
# -----------------------------
# 2) LLM (you can swap model)
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

#%%
# -----------------------------
# 3) Handoff Agents & Graph Nodes
# -----------------------------
#%%
# Agent A: Handoff Supervisor (LLM triage) decides next hand-off
class RouteDecision(BaseModel):
    next_handoff: Literal["identity", "policy", "math", "summary"] = Field(description="Which specialist should handle the next step.")
    rationale: str = Field(description="Short reason for this routing decision.")

handoff_sup_agent_system_message = """
You are a supervisor for an insurance assistant that uses hand-offs to specialists.

Given:
- user's question
- available policy details and policy chunks

Specialists:
- member_id_agent: obtain member_id
- policy_lookup_agent: obtain policy_id (requires member_id)
- policy_details_agent: obtain policy_details (requires policy_id)
- math_agent: Build math expressions from policy_details required to answer user question (requires policy_details)
- response_summarizer_agent: write a clear final response

Rules:
1) If error exists -> next_handoff = summary
2) If member_id missing -> next_handoff = identity
3) If policy_id missing -> next_handoff = policy
4) If policy_details missing -> next_handoff = policy
5) If user asks for cost/total/payment/how much/breakdown -> next = math
6) Else -> next_handoff = summary

"Important: Do not prepend json qualifier to the JSON.\n"

Decide routing:
- Ensure member id is known.
- "policy" for policy/coverage/benefits/explanations, definitions, copay/deductible details, etc.
- "math" for estimated payments, applying deductible/copay/coinsurance, computing totals.
- If both are needed, choose "policy" first if retrieval is needed, otherwise "math".
- "summary" only if the assistant already has an answer to user question.
- Important: Carefully inspect if required amounts and totals involving deductible/copay/coinsurance are already computed.
"""

handoff_sup_agent_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=handoff_sup_agent_system_message),
        SystemMessage(
            content=(
                "STATE SNAPSHOT:\n"
                "user_question={user_question}\n"
                "member_id={member_id}\n"
                "policy_id={policy_id}\n"
                "policy_details={policy_details}\n"
                "has_error={has_error}\n"
                "answer={answer}\n"
            )
        )
    ]
)

def handoff_sup_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    system_message = SystemMessage(content=handoff_sup_agent_system_message)

    snapshot = {
            "user_question": state.get( 'user_question' ),
            "member_id": state.get( 'member_id' ),
            "policy_id": state.get( 'policy_id' ),
            "policy_details": state.get( 'policy_details' ),
            "has_error": bool(state.get( 'error' )),
            "answer": state.get( 'answer' ) or "Not evaluated yet.",
    }
    
    human_message = HumanMessage( content=json.dumps( snapshot, indent=2 ))
    
    decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke( [system_message, human_message], config=config)

    nxt = decision.next_handoff
        
    logger.info("Handoff supervisor routing -> %s | %s", nxt, decision.rationale)
    return {"next_handoff": nxt}

#%%
# Agent B: ID Lookup Agent (A small ReAct agent that only knows member id lookup tool)
member_id_lookup_agent_prompt = (
    "You are MemberIDLookupAgent. Your job is ONLY to get user's member id.\n"
    "You can inspect user's question to extract user's member id.\n"
    "If member id is not specified, get the member id by invoking the tool.\n"
    f"If the tool returns a message starting with '{ERROR_PREFIX}', stop and output that error.\n"
    "Return ONLY the member_id string and nothing else."
)

member_id_lookup_agent = create_react_agent(
    model=llm,
    tools=TOOLS_IDENTITY,
    prompt=member_id_lookup_agent_prompt,
)

def member_id_lookup_agent_node(state: HandoffState, config: RunnableConfig ) -> HandoffState:
    if state.get( 'member_id' ):
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
        return {"error": err}
    
    return {"member_id": text}
#%%
# Agent C: PolicyAgent: load policy_id + policy details
def policy_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    member_id = state.get( 'member_id' )
    
    if not member_id:
        return {**state, "error": "Missing member_id before policy lookup."}

    policy_id = None
    
    t_policy_id, t_details = TOOLS_POLICY

    try:
        policy_id = t_policy_id.invoke( { "member_id": member_id } )
        err = _tool_error_guard( policy_id )
        
        if err:
            return { "error": err }
    except Exception as e:
        return { "error": t_policy_id.handle_tool_error( e ) }

    policy_details = None
    
    try:
        policy_details = t_details.invoke( { "policy_id": policy_id } )
        err = _tool_error_guard( policy_details )
        
        if err:
            return { "error": err }
    except Exception as e:
        return { "error": t_details.handle_tool_error( e ) }

    return {
        "policy_id": policy_id,
        "policy_details": policy_details
    }

#%%
# Agent E: MathAgent: LLM plans -> calculator tool executes
class MathPlan(BaseModel):
    expression: str = Field(description="A pure numeric expression to compute total payment.")
    explanation: str = Field(description="Short explanation of the expression and assumptions.")
    result_units: str = Field(description="Units, usually 'USD'.")

def math_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    user_question = state.get( 'user_question' )
    
    policy_details = state.get( 'policy_details', "" )
    
    policy_chunks = state.get( 'policy_chunks', [] )

    context_blocks = [f"[Policy Chunk {i}]\n{p}" for i, p in enumerate(policy_chunks, start=1)]

    prompt = f"""
You are a math specialist for insurance cost estimates.

Goal:
- Build a numeric expression to estimate amounts to answer user's question.
- Use ONLY policy_data facts and any relevant details in policy_text.
- If the question lacks needed numbers (like billed amount, allowed amount, service price),
  choose a reasonable "fee reference" from policy_data if available and clearly state the assumption.
- Important: Do not include currency signs or other non-math signs in a math expressions.
- Important: Use arithmetic and math operations only.

User question:
{user_question}

Policy Details (structured facts):
{policy_details}

Retrieved Policy Chunks:
{chr(10).join(context_blocks)}
"""
    plan: MathPlan = llm.with_structured_output(MathPlan).invoke(prompt, config=config)

    t_calc = TOOLS_MATH[0]
    try:
        result = t_calc.invoke({"expression": plan.expression})
        err = _tool_error_guard(result)
        if err:
            return {"error": err}
    except Exception as e:
        return {"error": t_calc.handle_tool_error(e)}

    try:
        val = float(result)
    except Exception:
        return {"error": f"Could not parse calculator result: {result}"}

    answer = (
        f"Calculated Amount: {val:.2f} ({plan.result_units}).\n\n"
        f"How I computed it:\n{plan.explanation}\n\n"
        f"Expression: {plan.expression} = {val:.2f}"
    )

    return { "answer": answer }
#%%
# Agent F: Response Summarizer Agent (no tools)
response_summarizer_agent_prompt = (
    "You are a friendly insurance policy assistant.\n"
    "Summarize the following information in easy-to-understand format.\n"
    "Use ONLY the evidence snippets to justify the answer.\n"
    "Do not perform any calculations.\n"
    "If calculations are already performed, use their explanations to summarize.\n"
    "If a policy detail is missing, say so and ask what to check next.\n"
)

response_summarizer_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=response_summarizer_agent_prompt,
)

def response_summarizer_agent_node(state: HandoffState, config: RunnableConfig) -> HandoffState:
    """
    Produce the final answer grounded in evidence.
    If there's a calc_result, include a short breakdown.
    """
    answer = state.get('answer')
    
    if state.get( 'error' ):
        answer = state.get( 'error' )
        
    messages: list[AnyMessage] = [SystemMessage(
                    content=(
                        "STATE:\n"
                        f"user_question={state.get('user_question')}\n"
                        f"member_id={state.get('member_id')}\n"
                        f"policy_id={state.get('policy_id')}\n"
                        f"policy_details={state.get('policy_details')}\n"
                        f"response_to_summarize={answer}\n"
                    )
                )]

    result = response_summarizer_agent.invoke({"messages": messages}, config=config)
    final_answer = result["messages"][-1].content.strip()
    return {"final_answer": final_answer}

#%%
# -----------------------------
# 4) Build the LangGraph
# -----------------------------
def build_ho_graph():
    g = StateGraph(HandoffState)

    g.add_node("handoff_supervisor", handoff_sup_agent_node)
    g.add_node("identity", member_id_lookup_agent_node)
    g.add_node("policy", policy_agent_node)
    g.add_node("math", math_agent_node)
    g.add_node("summary", response_summarizer_agent_node)

    g.set_entry_point("handoff_supervisor")

    def route(state: HandoffState) -> str:
        nxt = state.get("next_handoff")
        if nxt in ("identity", "policy", "math", "summary"):
            return nxt
        return "summary"

    g.add_conditional_edges(
        "handoff_supervisor",
        route,
        {
            "identity": "identity",
            "policy": "policy",
            "math": "math",
            "summary": "summary",
        },
    )

    # After specialists, return to handoff_supervisor to decide next hand-off
    g.add_edge("identity", "policy")
    g.add_edge("policy", "handoff_supervisor")
    g.add_edge("math", "handoff_supervisor")
    g.add_edge("summary", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)