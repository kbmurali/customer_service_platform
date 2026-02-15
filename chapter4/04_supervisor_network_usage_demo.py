#%%
from supervisor_network import build_supervisor_network, SupervisorState
from common_utils import make_langsmith_config
import uuid
import json

#%%
# ---------------------------------
# 1) Supervisor-Driven Network App
# ---------------------------------
app = build_supervisor_network()

#%%
# -----------------------------
# 2) Invoke App
# -----------------------------
def invoke_app( thread_id: str, question: str ):
    config = make_langsmith_config(thread_id=thread_id)
    initial_state: SupervisorState = {"user_question": question}
    final_state = app.invoke( initial_state, config=config )

    print("\n" + "=" * 80)
    print(f"THREAD: {thread_id}")
    print(f"QUESTION: {question}")
    print("\n" + "=" * 80)

    if final_state.get( 'error' ):
        print("[ERROR]")
        print(final_state.get( 'error' ))

    print("\n[FINAL ANSWER]")
    print(final_state.get( 'final_answer', "(no final_answer)"))

    print("\n\n" + "=" * 80)
    print("\n[DEBUG SNAPSHOT]")
    print("\n" + "=" * 80)
    print(
        json.dumps(
            {
                "member_id": final_state.get("member_id"),
                "policy_id": final_state.get("policy_id"),
                "visit_type": final_state.get("visit_type"),
                "calc_expression": final_state.get("calc_expression"),
                "calc_result": final_state.get("calc_result"),
                "next": final_state.get("next"),
                "routing_reason": final_state.get("routing_reason"),
                "error": final_state.get("error"),
            },
            indent=2,
        )
    )
    print("\n" + "=" * 80 + "\n")
    return final_state

#%%
# Turn 1: cost question (forces supervisor to gather IDs + policy + estimate)
thread_id = str(uuid.uuid4())
invoke_app( thread_id, "What would be my total payment for a primary visit?" )

#%%
# Turn 2: follow-up question on same thread_id (shows how supervisor can re-route)
invoke_app( thread_id, "Can you break down how you calculated that?" )

#%%
thread_id_2 = str(uuid.uuid4())
question_2 = "What is my policy id?"

invoke_app( thread_id=thread_id_2, question=question_2 )
# %%
