"""
Beginner-friendly RAG 2.0 (Dynamic Context Retrieval) for Health Insurance Member Support

Install:
  pip install -U langgraph langchain langchain-openai langchain-chroma chromadb pydantic

Env:
  export OPENAI_API_KEY="..."

What makes this RAG 2.0?
- The agent can retrieve AGAIN if it detects missing info (gap-check loop).
"""
#%%
from __future__ import annotations

from typing import TypedDict, List, Dict, Literal, Annotated
import operator
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#%%
# ----------------------------
# 1) Build a small Chroma KB
# ----------------------------

docs = [
    # PLAN / BENEFITS
    Document(page_content="SilverPlus PPO: Urgent care copay is $50.", metadata={"lane": "plan"}),
    Document(page_content="SilverPlus PPO: ER copay is $250 after deductible.", metadata={"lane": "plan"}),
    Document(page_content="Prior authorization is required for MRI/CT imaging on SilverPlus PPO.", metadata={"lane": "plan"}),

    # CLAIMS
    Document(page_content="Common denial reasons: missing prior authorization, out-of-network provider, service not covered.", metadata={"lane": "claims"}),
    Document(page_content="Appeals must be filed within 180 days of a denial; include denial letter and supporting docs.", metadata={"lane": "claims"}),

    # NETWORK
    Document(page_content="To confirm in-network: verify provider NPI and location in the provider directory.", metadata={"lane": "network"}),
    Document(page_content="Out-of-network services may cost more; emergencies are typically covered differently.", metadata={"lane": "network"}),
]

embeddings = OpenAIEmbeddings()  # uses OpenAI embedding model defaults
if "vectorstore" not in globals():
    vectorstore = Chroma(
        collection_name="health_member_support",
        embedding_function=embeddings,
    )
    vectorstore.add_documents(docs)
    print(">>> Vectorstore initialised with", vectorstore._collection.count(), "docs")

# Three filtered retrievers (one per “lane”)
plan_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"lane": "plan"}})
claims_retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"lane": "claims"}})
network_retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"lane": "network"}})


#%%
# ----------------------------
# 2) LLM + structured outputs
# ----------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class IntentOut(BaseModel):
    intent: Literal["coverage_or_cost", "prior_auth", "claim_denial_or_appeal", "provider_network", "other"] = Field(
        description="Member's primary intent"
    )

class GapOut(BaseModel):
    need_more: bool = Field(
        description=(
            "True ONLY if there is a clear, specific gap in the context that prevents "
            "answering the member's question. False if the draft already sufficiently "
            "addresses all parts of the question — do NOT set True just to be thorough."
        )
    )
    # Optional refined queries by lane (empty string means “don’t retrieve from that lane”)
    plan_query: str = ""
    claims_query: str = ""
    network_query: str = ""


intent_llm = llm.with_structured_output(IntentOut)
gap_llm = llm.with_structured_output(GapOut)

#%%
# ----------------------------
# 3) LangGraph State
# ----------------------------

class State(TypedDict):
    member_message: str
    intent: str

    # Dynamic context — uses operator.add reducer so lists accumulate across nodes
    plan_ctx: Annotated[List[str], operator.add]
    claims_ctx: Annotated[List[str], operator.add]
    network_ctx: Annotated[List[str], operator.add]

    draft: str
    final: str

    # RAG 2.0 loop control
    need_more: bool
    loop_count: int
    max_loops: int

    # Refined retrieval queries from gap_check
    _plan_query: str
    _claims_query: str
    _network_query: str

#%%
# ----------------------------
# 4) Graph nodes
# ----------------------------

def classify_intent(state: State) -> State:
    """Use structured output to classify intent."""
    result = intent_llm.invoke(
        f"Classify the intent of this health insurance member message:\n\n{state['member_message']}"
    )
    state["intent"] = result.intent
    print( f">>> Completed classify intent: {result.intent}\n" )
    return state


def retrieve_initial(state: State) -> State:
    """First retrieval pass based on intent (keep it simple)."""
    q = state["member_message"]

    # RAG 2.0: intentionally retrieve only plan context on first pass.
    # The gap_check loop is responsible for pulling claims/network if needed.
    # This ensures at least one gap-check iteration fires for multi-lane questions.
    state["plan_ctx"] = [d.page_content for d in plan_retriever.invoke(q)]

    print( f">>> Completed retrieve initial\n" )
    
    return state


def draft_answer(state: State) -> State:
    """Draft an answer using whatever context we currently have."""
    plan_ctx   = list(dict.fromkeys(state["plan_ctx"]))
    claims_ctx = list(dict.fromkeys(state["claims_ctx"]))
    network_ctx = list(dict.fromkeys(state["network_ctx"]))

    context_parts = []
    if plan_ctx:
        context_parts.append("PLAN:\n- " + "\n- ".join(plan_ctx))
    if claims_ctx:
        context_parts.append("CLAIMS:\n- " + "\n- ".join(claims_ctx))
    if network_ctx:
        context_parts.append("NETWORK:\n- " + "\n- ".join(network_ctx))

    context = "\n\n".join(context_parts) if context_parts else "(no context)"

    prompt = f"""
You are a health insurance member support assistant.
You MUST answer using ONLY the facts explicitly stated in the context below.
Do NOT use general knowledge, assumptions, or common sense to fill gaps.
If the context does not contain a specific fact needed to answer, explicitly say "I don't have that information in the current context."
Do NOT give medical advice.

Member message:
{state['member_message']}

Context:
{context}

Write a short draft response using ONLY the context above:
"""
    state["draft"] = llm.invoke(prompt).content
    print( f">>> Completed draft answer\n" )
    return state


def gap_check(state: State) -> State:
    """
    RAG 2.0: decide if we need more context,
    and if yes, produce refined retrieval queries (structured output).
    """
    context = "\n".join(dict.fromkeys(state["plan_ctx"] + state["claims_ctx"] + state["network_ctx"]))
    print(f">>> gap_check sees context:\n{context}\n")
    prompt = f"""
Evaluate whether the current context is sufficient to fully answer the member's question.

- If the context adequately covers ALL parts of the question, set need_more=False and leave all queries empty.
- ONLY set need_more=True if there is a clear, specific gap. Provide targeted queries ONLY for lanes missing info.
- Do not request more context just to be thorough — if the draft answers the question, that is enough.
- For claims lane, use queries like "appeal process steps denial" or "appeal filing deadline documents required".

Member message:
{state['member_message']}

Current context:
{context}

Current draft:
{state['draft']}
"""
    result = gap_llm.invoke(prompt)
    state["need_more"] = result.need_more

    # Store refined queries temporarily inside the state dict (simple approach)
    state["_plan_query"] = result.plan_query
    state["_claims_query"] = result.claims_query
    state["_network_query"] = result.network_query
    
    print(f">>> gap_check queries — plan: '{result.plan_query}' | claims: '{result.claims_query}' | network: '{result.network_query}'")
    print( f">>> Completed gap check with need more: {result.need_more}\n" )
    return state


def retrieve_more(state: State) -> State:
    """If needed, retrieve additional context using refined queries."""
    if state["loop_count"] >= state["max_loops"]:
        state["need_more"] = False
        return state

    pq = state.get("_plan_query", "").strip()
    cq = state.get("_claims_query", "").strip()
    nq = state.get("_network_query", "").strip()

    if pq:
        print( f">>> In retrieve more: {pq}\n" )
        new_plan = [d.page_content for d in plan_retriever.invoke(pq) if d.page_content not in state["plan_ctx"]]
        state["plan_ctx"] = new_plan

    if cq:
        print( f">>> In retrieve more: {cq}\n" )
        raw_claims = claims_retriever.invoke(cq)
        print(f">>> claims_retriever returned {len(raw_claims)} docs: {[d.page_content for d in raw_claims]}")
        new_claims = [d.page_content for d in raw_claims if d.page_content not in state["claims_ctx"]]
        state["claims_ctx"] = new_claims
        print(f">>> new claims docs being added: {new_claims}")

    if nq:
        print( f">>> In retrieve more: {nq}\n" )
        new_network = [d.page_content for d in network_retriever.invoke(nq) if d.page_content not in state["network_ctx"]]
        state["network_ctx"] = new_network

    state["loop_count"] += 1
    print( f">>> Completed retrieve more with loop count at: {state[ 'loop_count' ]}\n" )
    return state


def finalize(state: State) -> State:
    """Final response after we’re satisfied with context."""
    plan_ctx   = list(dict.fromkeys(state["plan_ctx"]))
    claims_ctx = list(dict.fromkeys(state["claims_ctx"]))
    network_ctx = list(dict.fromkeys(state["network_ctx"]))

    context_parts = []
    if plan_ctx:
        context_parts.append("PLAN:\n- " + "\n- ".join(plan_ctx))
    if claims_ctx:
        context_parts.append("CLAIMS:\n- " + "\n- ".join(claims_ctx))
    if network_ctx:
        context_parts.append("NETWORK:\n- " + "\n- ".join(network_ctx))

    context = "\n\n".join(context_parts) if context_parts else "(no context)"

    prompt = f"""
You are a health insurance member support assistant.
Use ONLY the provided context. Do NOT give medical advice.
Write a concise final response and list any key info you still need.

Member message:
{state['member_message']}

Context:
{context}

Final response:
"""
    state["final"] = llm.invoke(prompt).content
    return state


def route_after_gap(state: State) -> str:
    """Loop if we need more context and have loops left; otherwise finalize."""
    if state["need_more"] and state["loop_count"] < state["max_loops"]:
        return "retrieve_more"
    return "finalize"

#%%
# ----------------------------
# 5) Build the LangGraph
# ----------------------------

g = StateGraph(State)

g.add_node("classify_intent", classify_intent)
g.add_node("retrieve_initial", retrieve_initial)
g.add_node("draft_answer", draft_answer)
g.add_node("gap_check", gap_check)
g.add_node("retrieve_more", retrieve_more)
g.add_node("finalize", finalize)

g.set_entry_point("classify_intent")
g.add_edge("classify_intent", "retrieve_initial")
g.add_edge("retrieve_initial", "draft_answer")
g.add_edge("draft_answer", "gap_check")

g.add_conditional_edges("gap_check", route_after_gap, {
    "retrieve_more": "retrieve_more",
    "finalize": "finalize",
})

# RAG 2.0 loop: after retrieving more, draft again and re-check gaps
g.add_edge("retrieve_more", "draft_answer")
g.add_edge("finalize", END)

app = g.compile()

#%%
# ----------------------------
# 6) Try it
# ----------------------------

state: State = {
    "member_message": "I was never told I needed prior auth and now my claim got denied. What are the exact steps to file an appeal and what documents do I need to submit?",
    "intent": "other",
    "plan_ctx": [],
    "claims_ctx": [],
    "network_ctx": [],
    "draft": "",
    "final": "",
    "need_more": False,
    "loop_count": 0,
    "max_loops": 2,
    "_plan_query": "",
    "_claims_query": "",
    "_network_query": "",
}

#%%
out = app.invoke(state)
print(out["final"])