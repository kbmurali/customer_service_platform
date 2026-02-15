#%%
from chain_of_agents import build_coa_graph, ChainState
from common_utils import make_langsmith_config
import uuid

# %%
# -----------------------------
# 1) COA App
# -----------------------------
app = build_coa_graph()

#%%
# -----------------------------
# 2) Invoke COA App
# -----------------------------
def invoke_app(thread_id: str, question: str):
    runnable_config = make_langsmith_config(thread_id=thread_id)
    state: ChainState = {"user_question": question}

    final_state = app.invoke(state, config=runnable_config)

    if final_state.get("error"):
        print("\n[FINAL ERROR]")
        print(final_state["error"])
        print("\n")

    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n")

    print("\n" + "=" * 70)
    print("\n[DEBUG STATE]")
    print("\n" + "=" * 70)
    print("\n")
    print(
        {
            "member_id": final_state.get("member_id"),
            "policy_id": final_state.get("policy_id"),
            "doctor_visit_fee": final_state.get("doctor_visit_fee"),
            "copay": final_state.get("copay"),
            "member_portion_pct": final_state.get("member_portion_pct"),
            "next_step": final_state.get("next_step"),
            "clarification_question": final_state.get("clarification_question"),
            "evaluated_answer": final_state.get("evaluated_answer"),
            "calc_expression": final_state.get("calc_expression"),
            "calc_result": final_state.get("calc_result"),
            "error": final_state.get("error"),
        }
    )

#%%
thread_id = str(uuid.uuid4())
question = "What is my policy id?"

invoke_app( thread_id=thread_id, question=question )

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a primary doctor visit?"

invoke_app( thread_id=thread_id_1, question=question_1 )

# %%
question_2 = "Can you breakdown how you evaluated the total payment?"

invoke_app( thread_id=thread_id_1, question=question_2 )

# %%
