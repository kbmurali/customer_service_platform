#%%
from simple_sequential_orchestration import build_sequential_graph, SharedState
from common_utils import make_langsmith_config
import uuid

#%%
# -----------------------------
# 1) SSO App
# -----------------------------
app = build_sequential_graph()

#%%
# -----------------------------
# 2) Invoke SSO App
# -----------------------------
def invoke_app(thread_id: str, question: str):
    runnable_config = make_langsmith_config(thread_id=thread_id)
    state: SharedState = {"user_question": question}

    final_state = app.invoke(state, config=runnable_config)
    
    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n")
    
    if final_state.get("errors"):
        print("\n[FINAL ERRORS]")
        for error in final_state.get( 'errors', [] ):
            print( error )
            print("\n")

    print("\n" + "=" * 70)
    print("\n[DEBUG STATE]")
    print("\n" + "=" * 70)
    print("\n")
    print(
        {
            "member_id": final_state.get("member_id"),
            "policy_id": final_state.get("policy_id"),
            "policy_details": final_state.get("policy_details"),
            "evaluated_reason": final_state.get("evaluated_reason"),
            "calc_expression": final_state.get("calc_expression"),
            "calc_result": final_state.get("calc_result"),
            "errors": final_state.get("errors"),
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
