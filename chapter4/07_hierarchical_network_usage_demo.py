#%%
from hierarchical_network import build_root_sup_graph, RootState
from common_utils import make_langsmith_config
import uuid
import json

#%%
# ---------------------------------
# 1) Hierarchical Network App
# ---------------------------------
app = build_root_sup_graph()

#%%
# -----------------------------------------------------------------------------
# 2) Invoke App
# -----------------------------------------------------------------------------
def invoke(thread_id: str, question: str):
    config = make_langsmith_config(thread_id)
    initial: RootState = {"user_question": question}
    final_state = app.invoke(initial, config=config)

    print("\n" + "=" * 80)
    print(f"THREAD: {thread_id}")
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", "(no final_answer)"))

    if final_state.get("error"):
        print("\n[ERROR]")
        print(final_state["error"])

    print("\n[DEBUG STATE]")
    debug_fields = {
        "member_id": final_state.get("member_id"),
        "policy_id": final_state.get("policy_id"),
        "policy_details_present": bool(final_state.get("policy_details")),
        "latest_claim_present": bool(final_state.get("latest_claim")),
        "visit_type": final_state.get("visit_type"),
        "calc_expression": final_state.get("calc_expression"),
        "calc_result": final_state.get("calc_result"),
        "next": final_state.get("next"),
        "routing_reason": final_state.get("routing_reason"),
        "error": final_state.get("error"),
    }
    print(json.dumps(debug_fields, indent=2))
    print("=" * 80 + "\n")
    
#%%
thread_id1 = str(uuid.uuid4())

# Example 1: pure policy / estimate question (policy_team + estimate)
invoke(thread_id1, "What would be my total payment for a specialist visit?")

#%%
thread_id2 = str(uuid.uuid4())

# Example 2: claims question (claims_team)
invoke(thread_id2, "What is the status of my latest claim?")

#%%
thread_id3 = str(uuid.uuid4())

# Example 3: combined question (root may call both teams, then explain)
invoke(thread_id3, "My claim seems pending—does my policy require a deductible for that specialist visit, and what would I pay?")