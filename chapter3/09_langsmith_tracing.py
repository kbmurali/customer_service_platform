#%%
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, Tool
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

import numexpr
import math
import os

from langgraph.checkpoint.memory import InMemorySaver   # short-term (thread) memory
from langgraph.store.memory import InMemoryStore        # optional long-term memory
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
import uuid

# --- new imports for LangSmith ---
from langsmith import Client
from langchain_core.tracers import LangChainTracer

# --- new imports for Logging ---
import logging

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
# Embeddings for Chroma
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Sample member → policy mapping
sample_member_policy_docs = [
    Document(
        page_content="Member ID abc123 is associated with policy Policy_abc123_1.",
        metadata={"member_id": "abc123", "policy_id": "Policy_abc123_1"}
    ),
    Document(
        page_content="Member ID xyz789 is associated with policy Policy_xyz789_1.",
        metadata={"member_id": "xyz789", "policy_id": "Policy_xyz789_1"}
    ),
]

# Sample policy → policy details
sample_policy_detail_docs = [
    Document(
        page_content=(
            "Policy ID Policy_abc123_1. "
            "Doctor visit fee is $100 and Co-Pay is $5. "
            "Member pays 10% of the doctor visit fee. "
            "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
        ),
        metadata={
            "policy_id": "Policy_abc123_1",
            "doctor_visit_fee": 100,
            "copay": 5,
            "member_portion_pct": 0.10,
        }
    ),
    Document(
        page_content=(
            "Policy ID Policy_xyz789_1. "
            "Doctor visit fee is $150 and Co-Pay is $10. "
            "Member pays 20% of the doctor visit fee. "
            "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
        ),
        metadata={
            "policy_id": "Policy_xyz789_1",
            "doctor_visit_fee": 150,
            "copay": 10,
            "member_portion_pct": 0.20,
        }
    ),
]

# Optional: set a persistent directory if you want to reuse the store between runs
persist_dir_member_policy = os.path.join("chroma_store", "member_policy")
persist_dir_policy_details = os.path.join("chroma_store", "policy_details")

member_policy_db = Chroma.from_documents(
    documents=sample_member_policy_docs,
    embedding=embeddings,
    collection_name="member_policy",
    persist_directory=persist_dir_member_policy,
)

policy_details_db = Chroma.from_documents(
    documents=sample_policy_detail_docs,
    embedding=embeddings,
    collection_name="policy_details",
    persist_directory=persist_dir_policy_details,
)
    
#%%
@tool
def get_member_id() -> str:
    """
    Gets a member id when user's member id is required_summary_

    Returns:
        str: member id
    """
    print( f"Get Member Id Invoked\n" )
    member_id = input( "What is your member id?")
    return member_id.strip()

@tool
def get_member_policy_id(member_id: str) -> str:
    """
    Gets policy id of a member given member id by retrieving from a Chroma vector database.

    Args:
        member_id (str): Unique member id used to access member information

    Returns:
        str: policy id of a member
    """
    print(f"Get Member Policy Id Invoked For {member_id}\n")

    # Use the member_id as the query into Chroma
    docs = member_policy_db.similarity_search( member_id, k=1 )

    if not docs or docs[0].metadata.get("member_id") != member_id:
        raise ValueError( f"No policy found for member id: {member_id}" )

    # We expect policy_id to be in metadata
    policy_id = docs[0].metadata.get("policy_id")
    
    if not policy_id:
        raise ValueError( f"Policy id missing in vector store metadata for member id: {member_id}" )

    return policy_id

@tool
def get_policy_details(policy_id: str) -> str:
    """
    Gets policy details containing fee, co-pay, and member payment information given policy id
    by retrieving from a Chroma vector database.

    Args:
        policy_id (str): Unique policy id used to access policy information

    Returns:
        str: Policy details
    """
    print(f"Get Policy Details Invoked For {policy_id}\n")

    # Use the policy_id as the query into Chroma
    docs = policy_details_db.similarity_search(policy_id, k=1)

    if not docs or docs[0].metadata.get( "policy_id" ) != policy_id:
        raise ValueError( f"No policy details found for policy id: {policy_id}" )

    # You can either return doc.page_content directly,
    # or format it using metadata if you prefer more structure.
    policy_details = docs[0].page_content

    return policy_details

@tool
def calculator( expression: str ) -> str:
    """
    Calculates a mathematical expression comprising of terms 
    with mathematical operations such as arthmetic, exponential, 
    power of, and logarthmic. Example: 0.01 * 100 + 5 + 4**2"

    Args:
        expression (str): Mathematic expression that needs to be evaluated and calculated

    Returns:
        str: Caluculated Value
    """
    print( f"Calculator Invoked With Expression: {expression}\n" )
    
    math_constants = { "pi" : math.pi, "i": 1j, "e":math.exp }
    
    result = numexpr.evaluate( expression.strip(), local_dict=math_constants )
    
    return str( result )

#%%
ERROR_PREFIX = "TOOL_ERROR"

tools = [ 
         Tool(
                name="get_member_id",
                func=get_member_id,
                description=get_member_id.description,
                args_schema={},
                handle_tool_error=lambda e: f"{ERROR_PREFIX} in get_member_id: {e}"
            ),
         Tool(
                name="get_member_policy_id",
                func=get_member_policy_id,
                description=get_member_policy_id.description,
                args_schema=get_member_policy_id.args_schema,
                handle_tool_error=lambda e: f"{ERROR_PREFIX} in get_member_policy_id: {e}"
            ),
         Tool(
                name="get_policy_details",
                func=get_policy_details,
                description=get_policy_details.description,
                args_schema=get_policy_details.args_schema,
                handle_tool_error=lambda e: f"{ERROR_PREFIX} in get_policy_details: {e}"
            ),
         Tool(
                name="calculator",
                func=calculator,
                description=calculator.description,
                args_schema=calculator.args_schema,
                handle_tool_error=lambda e: f"{ERROR_PREFIX} in calculator: {e}"
            )
]

#%%
system_message = (
    "You are a member service representative and can answer questions on member insurance policy."
    "If member id is not specified, always get the member id without asking the user for it."
    "Always use user specified member id to get member's policy id."
    "Using the policy id you can get policy details."
    "All your answers must be based on policy details."
    "Questions involving numbers, you must form correct mathematical expressions based on policy details."
    "You can use a calculator to evaluate mathematical expressions."
    f"If any tool returns a message starting with '{ERROR_PREFIX}', you must NOT call any more tools, "
    "must NOT continue execution, and must instead respond once with a brief error explanation and stop."
)

llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=12000
)

#%%
# --- SHORT-TERM MEMORY: checkpointer ---
# This will keep per-thread conversation state (messages, tool calls, etc.)
checkpointer = InMemorySaver()

# --- OPTIONAL LONG-TERM MEMORY: store ---
# For demo purposes we use in-memory; in production you’d use Postgres, Redis, etc.
memory_store = InMemoryStore()

# --- MESSAGE TRIMMING HOOK (memory management) ---
def pre_model_hook(state: dict) -> dict:
    """
    Called before the LLM node is run.
    We trim the conversation so the model only sees the last N tokens,
    but the *full* history is still saved in the checkpointer.
    """
    trimmed = trim_messages(
        state["messages"],
        strategy="last",                         # keep last messages
        token_counter=count_tokens_approximately,
        max_tokens=384,                          # tune based on your model
        start_on="human",
        end_on=("human", "tool"),
    )
    # We keep full history in state["messages"], but only pass trimmed context
    # to the model via `llm_input_messages`.
    return {"llm_input_messages": trimmed}

#%%
# --- CREATE AGENT WITH MEMORY (REUSES YOUR create_react_agent APPROACH) ---
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message,       # same system_message as your example
    checkpointer=checkpointer,   # short-term memory: per-thread history
    store=memory_store,          # optional: long-term memory backend
    pre_model_hook=pre_model_hook,
)

#%% 
# --- CREATING A LOGGER :: Can be configured to central logging in production---
logger = logging.getLogger(__name__)

# %%
# --- INITIATING LANGSMITH CLIENT ---
langsmith_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT")
)

langsmith_tracer = LangChainTracer(
    client=langsmith_client,
    project_name=os.getenv("LANGSMITH_PROJECT", "langchain-primer-demo")
)

# --- USING THE AGENT WITH THREAD-SCOPED MEMORY AND LANGSMITH TRACING ---
trace_id = str(uuid.uuid4())

# --- INITIATING CONFIG WITH THREAD ID, TRACE ID, AND COMMON LOGGER
thread_config = RunnableConfig(
    configurable= {
                        "thread_id": trace_id,  # any stable ID for this conversation
    },
    callbacks=[langsmith_tracer],
    metadata={ 
                "app": os.getenv("LANGSMITH_PROJECT", "langchain-primer-demo"), 
                "env": "demo",
                
                #Same trace id for conversational memory, LangSmith tracing, and other Logging
                "request_id" : trace_id 
             },
    tags=["member-service-demo", "langchain-primer"],
)

#%%
# First conversation:
question_1 = HumanMessage( content="What would be my total payment for a doctor visit?" )

logger.info( "Invoking Agent to Answer Question 1", extra={"request_id": trace_id })

result_1 = agent.invoke( 
                            {"messages": [question_1]},
                            config=thread_config,
)

print("Conversation 1:", result_1["messages"][-1].content)

#%%
# Second conversation:
question_2 = HumanMessage( content="Can you remind me how you calculated that total and break it down?" )

logger.info( "Invoking Agent to Answer Question 2", extra={"request_id": trace_id })

result_2 = agent.invoke( 
                            {"messages": [question_2]},
                            config=thread_config,
)

print("Conversation 2:", result_2["messages"][-1].content)
# %%
