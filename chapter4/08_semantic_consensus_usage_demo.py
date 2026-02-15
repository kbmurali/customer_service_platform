#%%
from chain_of_agents import build_coa_graph, ChainState
from magentic_orchestration import build_magentic_graph, MagenticState
from simple_sequential_orchestration import build_sequential_graph, SharedState
from semantic_consensus import build_semantic_consensus_graph, ConsensusState, Provider
from common_utils import make_langsmith_config

import os
import uuid
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

# --------------------------------------------------------------------------------------
# 1) Provider wrappers (treat each imported example as a "solution provider agent")
# --------------------------------------------------------------------------------------
def build_providers() -> List[Provider]:
    # Each example uses make_langsmith_config(thread_id) and returns final_answer in the state.
    def run_coa(question: str, member_id: str ) -> str:
        thread_id = str(uuid.uuid4())
        config: RunnableConfig = make_langsmith_config(thread_id=thread_id)
        app = build_coa_graph()  # build a fresh compiled graph
        state: ChainState = {"user_question": question, "member_id": member_id}
        out = app.invoke(state, config=config)
        return out.get("final_answer", "")

    def run_magentic(question: str, member_id: str) -> str:
        thread_id = str(uuid.uuid4())
        config: RunnableConfig = make_langsmith_config(thread_id=thread_id)
        app = build_magentic_graph()
        user_question = f"{question}. Member Id: {member_id}"
        state: MagenticState = {"user_question": user_question}
        out = app.invoke(state, config=config)
        return out.get("final_answer", "")

    def run_sso(question: str, member_id: str ) -> str:
        thread_id = str(uuid.uuid4())
        config: RunnableConfig = make_langsmith_config(thread_id=thread_id)
        app = build_sequential_graph()
        state: SharedState = {"user_question": question, "member_id": member_id}
        out = app.invoke(state, config=config)
        return out.get("final_answer", "")

    return [
        Provider(name="Chain-of-Agents", solve=run_coa),
        Provider(name="Simple Sequential Orchestration", solve=run_sso),
        Provider(name="Magentic Orchestration", solve=run_magentic),
    ]

#%%
# --------------------------------------------------------------------------------------
# 2) Invoke Consensus App
# --------------------------------------------------------------------------------------
providers = build_providers()

judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")
judge_llm = ChatOpenAI(model=judge_model, max_tokens=2000, temperature=0.2)

app = build_semantic_consensus_graph( providers=providers, judge_llm=judge_llm )

def invoke_app(thread_id: str, question: str):
    runnable_config = make_langsmith_config(thread_id=thread_id)
    
    state: ConsensusState = {"user_question": question}

    final_state = app.invoke(state, config=runnable_config)
    
    if final_state.get("error"):
        print("\n[FINAL ERROR]")
        print(final_state["error"])
        print("\n")
    
    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n")
    
    print("\n[VOTES]")
    print(len(final_state.get( "votes", [] )))
    print("\n")
    
    print("\n\n[BEST PROVIDER]")
    wi = final_state.get( "winning_index" )
    print( f"---->Winning Index: {wi}" )
    if wi:
        print( f"---->Winning Provider: {providers[wi].name}" )
    print("\n") 

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a primary doctor visit?"

invoke_app( thread_id=thread_id_1, question=question_1 )
# %%
