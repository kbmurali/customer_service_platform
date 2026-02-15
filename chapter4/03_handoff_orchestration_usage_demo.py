#%%
from handoff_orchestration import build_ho_graph, HandoffState
from common_utils import make_langsmith_config
from langchain_core.messages import HumanMessage
import uuid

#%%
# -----------------------------
# 1) Handoff App
# -----------------------------
app = build_ho_graph()

#%%
# -----------------------------
# 2) Invoke HO App
# -----------------------------
def invoke_app(thread_id: str, question: str):
    runnable_config = make_langsmith_config(thread_id=thread_id)

    state: HandoffState = {"user_question": question}

    final_state = app.invoke(state, config=runnable_config)

    if final_state.get("error"):
        print("\n[FINAL ERROR]")
        print(final_state.get("error"))
        print("\n")

    print("\n[FINAL ANSWER]")
    print(final_state.get("final_answer", ""))
    print("\n")

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a doctor visit?"

invoke_app(thread_id=thread_id_1, question=question_1)

# %%
question_2 = "Can you breakdown how you evaluated the total payment?"

invoke_app(thread_id=thread_id_1, question=question_2)

#%%
thread_id_2 = str(uuid.uuid4())
question_2 = "What is my policy id?"

invoke_app(thread_id=thread_id_2, question=question_2)
# %%
