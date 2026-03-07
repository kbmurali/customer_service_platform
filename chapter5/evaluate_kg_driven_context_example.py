#%%
from langsmith import Client

from langchain_openai import ChatOpenAI
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

import pandas as pd
import time

from cypher_driven_context_example import query_knowledge_graph

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#%%
client = Client()

# 1. Create a Dataset in LangSmith
dataset_name = "Insurance-Graph-QA-v2"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name, description="GraphRAG complex queries")

    # 2. Add Examples (Ground Truth)
    client.create_examples(
        inputs=[
            {"question": "What is the total amount of all pending claims of 'Alice Smith' and her family?"},
            {"question": "Which policy covers 'Bob Smith' and what is the coverage limit?"},
        ],
        outputs=[
            {
                "answer": "$1,500",
                "required_nodes": ["Alice Smith", "Bob Smith", "CLM-101"]
            },
            {
                "answer": "Policy POL-GOLD-001 (PPO) with a limit of $10,000",
                "required_nodes": ["POL-GOLD-001", "Bob Smith"]
            }
        ],
        dataset_id=dataset.id,
    )

#%%
# 2. Define the faithfulness evaluator
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def graph_faithfulness_evaluator(run: Run, example: Example) -> list[dict]:
    """
    Evaluates if the run answer matches the ground truth AND if it used graph tools.
    """
    # 1. Extract Run Data
    student_answer = run.outputs.get("final_answer", "")
    ground_truth = example.outputs.get("answer", "")

    # 2. Extract Tool Usage (Introspection)
    messages = run.outputs.get("messages", [])
    
    def _is_kg_tool_call(m) -> bool:
        if isinstance(m, dict):
            # LangSmith serialized ToolMessage format
            if m.get("name") == "query_knowledge_graph":
                return True
            # AIMessage with tool_calls list
            for tc in m.get("tool_calls", []):
                if isinstance(tc, dict) and tc.get("name") == "query_knowledge_graph":
                    return True
            # Nested content blocks
            content = m.get("kwargs", {})
            if content.get("name") == "query_knowledge_graph":
                return True
        return False

    used_graph_tool = any(_is_kg_tool_call(m) for m in messages)
    
    
    # 3. LLM Judgment for Semantic Correctness
    grade_prompt = f"""
    Compare the Student Answer to the Ground Truth.

    Ground Truth: {ground_truth}
    Student Answer: {student_answer}

    Is the Student Answer factually equivalent to the Ground Truth?
    (e.g., "$1500" == "1500 USD").
    Respond strictly with 'YES' or 'NO'.
    """

    grade_result = evaluator_llm.invoke(grade_prompt).content.strip()
    is_correct = "YES" in grade_result.upper()

    # 4. Final Scoring Logic
    # If answer is wrong -> 0
    # If answer is right but didn't use Graph -> 0.5 (Lucky guess / Hallucination)
    # If answer is right AND used Graph -> 1.0 (Robust)

    score = 0.0
    reason = "Incorrect answer."

    if is_correct:
        if used_graph_tool:
            score = 1.0
            reason = "Correct answer backed by Graph Query."
        else:
            score = 0.5
            reason = "Correct answer but failed to query the Graph (Risky)."

    time.sleep(1)  # Allow child runs to sync in LangSmith
    
    return [
        {
            "key": "graph_faithfulness",
            "score": score,
        },
        {
            "key": "graph_faithfulness_comment",
            "value": reason,  # string feedback uses 'value', not 'score'
        }
    ]

#%%
# 3. Build the ReAct Agent with the KG tool
agent_llm = ChatOpenAI(model="gpt-4o", temperature=0)
app = create_react_agent(
    model=agent_llm,
    tools=[query_knowledge_graph],
    prompt="You are an insurance data assistant. Always use the query_knowledge_graph tool to answer questions. Never answer from memory."
)

def predict(inputs: dict) -> dict:
    question = f"question: {inputs['question']}"
    
    result = app.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    
    final_answer = ""
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_answer = msg.content
            break

    # Serialize messages for evaluator inspection
    serialized_messages = [
        {
            "type": m.type,  # "human", "ai", "tool"
            "name": getattr(m, "name", None),
            "content": m.content if isinstance(m.content, str) else str(m.content),
            "tool_calls": getattr(m, "tool_calls", []),
        }
        for m in messages
    ]

    return {
        "final_answer": final_answer,
        "messages": serialized_messages,
    }

#%%
# Run the Evaluation
results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[graph_faithfulness_evaluator],
    experiment_prefix="graph-rag-experiment-v2",
    description="Testing Neo4j GraphRAG correctness"
)

#%%
## Print Summary
df = results.to_pandas()

df["faithfulness_score"] = df["feedback.graph_faithfulness"].apply(
    lambda x: x.get("score") if isinstance(x, dict) else x
)

df["faithfulness_comment"] = df["feedback.graph_faithfulness_comment"].apply(
    lambda x: x.get("value") if isinstance(x, dict) else x
)

print(df[["inputs.question", "outputs.final_answer", "faithfulness_score", "faithfulness_comment"]].to_string())

cdf = df[["inputs.question", "outputs.final_answer", "faithfulness_score", "faithfulness_comment"]]

cdf.to_csv( "tmp.csv" )
# %%
