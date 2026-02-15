"""
Semantic Consensus (LangGraph) pattern.

What this file does:
- Imports the example agent implementations (WITHOUT re-implementing them) and treats them
  as "solution providers".
- Runs all providers on the same user question to produce candidate answers.
- Uses a stronger "Judge" LLM to pick the best answer via semantic appropriateness + majority voting.

Requirements:
- langchain, langchain-openai, langgraph
- OpenAI API key in env (OPENAI_API_KEY)
- (Optional) LangSmith env vars, as used by the example files.
"""
#%%
from __future__ import annotations

from common_utils import get_member_id
from common_utils import wrap_tool, _tool_error_guard, ERROR_PREFIX

import os
import json
import re
from dataclasses import dataclass
import logging
from typing import List, Dict, TypedDict, Callable

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
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
                            "get_member_id" : wrap_tool( get_member_id ) 
                        }

#%%
# --------------------------------------------------------------------------------------
# 1) Provider wrapper : Tream each underlying pattern impl as a provider
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Provider:
    name: str
    solve: Callable[[str, str], str]
    
# --------------------------------------------------------------------------------------
# 2) Semantic Consensus (Judge + majority voting)
# --------------------------------------------------------------------------------------
def _sanitize_llm_json(raw: str) -> str:
    """Remove ```json fences (common LLM formatting) so json.loads can parse."""
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    return raw

class Candidate(TypedDict, total=False):
    provider: str
    answer: str
    error: str


class JudgeVote(TypedDict, total=False):
    judge: str
    winner_index: int
    scores: List[int]
    rationale: str


class ConsensusState(TypedDict, total=False):
    user_question: str
    member_id: str
    candidates: List[Candidate]
    votes: list[JudgeVote]
    winning_index: int
    error: str
    final_answer: str

def _majority_vote(votes: List[JudgeVote], num_candidates: int) -> int:
    counts = [0] * num_candidates
    score_sums = [0] * num_candidates

    for v in votes:
        wi = v.get( "winner_index" )
        
        if wi and 0 <= wi < num_candidates:
            counts[wi] += 1
            
        scores = v.get("scores") or []
        
        for i in range(min(num_candidates, len(scores))):
            score_sums[i] += int(scores[i])

    # Choose by (vote count, total score) to break ties.
    best = 0
    
    for i in range(1, num_candidates):
        if counts[i] > counts[best]:
            best = i
        elif counts[i] == counts[best] and score_sums[i] > score_sums[best]:
            best = i
            
    return best

#%%
# Agent A: Intake / ID Lookup Agent
# A small ReAct agent is used even though a Tool only agent could be sufficient 
# to demonstrate that other full-fledged agents can be integrated.
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), max_tokens=2000)

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

def member_id_lookup_agent_node(state: ConsensusState, config: RunnableConfig ) -> ConsensusState:
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
def build_semantic_consensus_graph( providers: List[Provider], judge_llm ) :
    """
    Build a LangGraph app that:
    1) Collects candidate answers from provider agents
    2) Runs three judge passes (different judging styles)
    3) Majority-votes to pick the final answer
    """
    
    def collect_candidates_node(state: ConsensusState) -> ConsensusState:
        user_question = state.get("user_question", "")
        
        member_id: str = state.get( "member_id" ) or ""
        
        if not member_id:
            return {**state, "error": "Missing member_id before policy lookup."}
        
        candidates: List[Candidate] = []
        
        for provider in providers:
            try:
                ans = provider.solve(user_question, member_id)
                candidates.append({"provider": provider.name, "answer": ans})
            except Exception as e:
                candidates.append({"provider": provider.name, "answer": "", "error": str(e)})
        return {**state, "candidates": candidates, "votes": []}

    def make_judge_node(judge_name: str, style_instructions: str):
        def judge_node(state: ConsensusState, config: RunnableConfig) -> ConsensusState:
            user_question = state.get("user_question", "")
            
            candidates = state.get("candidates", [])
            if not candidates:
                return {**state, "final_answer": "No candidate answers were produced."}
            
            votes = state.get("votes", [])
            logger.info( f"judge_node {judge_name}:: number of votes = {len(votes)}" )
        
            # Build a compact candidate list
            formatted = []
            for i, c in enumerate(candidates):
                provider = c.get("provider", f"provider_{i}")
                ans = c.get("answer", "")
                err = c.get("error")
                if err:
                    ans = f"[ERROR from {provider}] {err}"
                formatted.append(f"Candidate {i} ({provider}):\n{ans}")

            system_message = SystemMessage(
                content=(
                    "You are a strict judge that selects the best answer among multiple candidates.\n"
                    "You must judge on: correctness, relevance to the user's question, clarity, and completeness.\n"
                    "You must NOT invent facts.\n"
                    "Return ONLY valid JSON (no code fences).\n\n"
                    "JSON schema:\n"
                    "{\n"
                    '  "winner_index": <int>,\n'
                    '  "scores": [<int 1-10 for each candidate>],\n'
                    '  "rationale": "<short explanation>"\n'
                    "}\n\n"
                    f"Judging style:\n{style_instructions}\n"
                )
            )

            human_message = HumanMessage(
                content="User question:\n"
                        f"{user_question}\n\n"
                        "Candidates:\n\n" + "\n\n---\n\n".join(formatted)
            )

            raw = judge_llm.invoke( [system_message, human_message], config=config ).content
            raw = _sanitize_llm_json(raw)

            try:
                payload = json.loads(raw)
            except Exception:
                # Fail safe: pick the first non-error candidate.
                winner = 0
                for i, c in enumerate(candidates):
                    if c.get("answer") and not c.get("error"):
                        winner = i
                        break
                payload = {"winner_index": winner, "scores": [5] * len(candidates), "rationale": "Judge JSON parse failed; fallback selection."}

            # Clamp and normalize
            num_candidates = len(candidates)
            
            winner_index = int(payload.get("winner_index", 0))
            
            if winner_index < 0 or winner_index >= num_candidates:
                winner_index = 0

            scores = payload.get("scores") or []
            if not isinstance(scores, list):
                scores = []
                
            # Ensure we have a score for each candidate
            norm_scores: List[int] = []
            
            for i in range(num_candidates):
                try:
                    s = int(scores[i]) if i < len(scores) else 5
                except Exception:
                    s = 5
                s = max(1, min(10, s))
                norm_scores.append(s)

            vote: JudgeVote = {
                "judge": judge_name,
                "winner_index": winner_index,
                "scores": norm_scores,
                "rationale": str(payload.get("rationale", "")),
            }
            
            votes = votes + [vote]
            
            return {**state, "votes": votes}
        return judge_node
    
    judge_strict = make_judge_node(
        "strict_correctness",
        "Prefer the candidate that is most factually grounded and least speculative.",
    )
    
    judge_helpful = make_judge_node(
        "helpful_explainer",
        "Prefer the candidate that is easiest for a beginner to understand, while still correct.",
    )
    
    judge_skeptical = make_judge_node(
        "skeptical_safety",
        "Penalize candidates that make assumptions or omit important caveats.",
    )
    
    def aggregate_node(state: ConsensusState) -> ConsensusState:
        candidates = state.get("candidates", [])
        votes = state.get("votes", [])
        logger.info( f"aggregate_node:: number of votes = {len(votes)}" )
        
        if not candidates:
            return {**state, "final_answer": "No candidates available to choose from."}

        winning_index = _majority_vote(votes, num_candidates=len(candidates))
        
        best = candidates[winning_index]
        
        final_answer = best.get("answer") or ""
        
        error = best.get("error") or ""

        return {**state, "winning_index": winning_index, "final_answer": final_answer, "error": error}
    
    def stop_if_error(state: ConsensusState) -> str:
        return "stop" if state.get('error') else "continue"

    g = StateGraph(ConsensusState)
    g.add_node("intake", member_id_lookup_agent_node)
    g.add_node("collect", collect_candidates_node)
    g.add_node("judge1", judge_strict)
    g.add_node("judge2", judge_helpful)
    g.add_node("judge3", judge_skeptical)
    g.add_node("aggregate", aggregate_node)

    g.set_entry_point("intake")
    g.add_conditional_edges("intake", stop_if_error, {"stop": END, "continue": "collect"})
    g.add_conditional_edges("collect", stop_if_error, {"stop": END, "continue": "judge1"})
    
    g.add_edge("judge1", "judge2")
    g.add_edge("judge2", "judge3")
    g.add_edge("judge3", "aggregate")
    g.add_edge("aggregate", END)

    checkpointer=InMemorySaver()
    return g.compile(checkpointer=checkpointer)
