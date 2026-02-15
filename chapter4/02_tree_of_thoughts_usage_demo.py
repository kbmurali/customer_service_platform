#%%
from tree_of_thoughts import build_tot_graph, ToTState
from common_utils import make_langsmith_config
import uuid

#%%
# -----------------------------
# 1) ToT App
# -----------------------------
app = build_tot_graph()

#%%
# -----------------------------
# 2) Invoke ToT App
# -----------------------------
def invoke_app( thread_id : str, question: str ):
    runnable_config = make_langsmith_config( thread_id=thread_id )
    
    state: ToTState = {"user_question": question}

    final_state = app.invoke(
        state,
        config=runnable_config
    )
    
    print("\n--- FINAL ANSWER ---\n")
    print(final_state["final_answer"])
    
    print("\n" + "=" * 70)
    print("\n[DEBUG STATE]")
    print("\n" + "=" * 70)
    print("\n")
    
    for t in final_state.get("thoughts", []):
        print(f"- {t['title']} | score={t.get('score')} | expr={t.get('expression')}")


#%%
thread_id = str(uuid.uuid4())
question = "What is my policy id? my member id is abc123"

invoke_app( thread_id=thread_id, question=question )

#%%
thread_id_1 = str(uuid.uuid4())
question_1 = "What would be my total payment for a doctor visit?"

invoke_app( thread_id=thread_id_1, question=question_1 )
# %%
question_2 = "Can you breakdown how you evaluated the total payment?"

invoke_app( thread_id=thread_id_1, question=question_2 )
# %%
