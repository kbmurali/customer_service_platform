#%%
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
import numexpr
import math

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

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
def get_member_policy_id( member_id : str ) -> str:
    """
    Gets policy id of a member given member id

    Args:
        member_id (str): Unique member id used to access member information

    Returns:
        str: policy id of a member
    """
    print( f"Get Member Policy Id Invoked For {member_id}\n" )
    return f"Policy_{member_id}_1"

@tool
def get_policy_details( policy_id : str ) -> str:
    """
    Gets policy details containing fee, co-pay, and member payment information given policy idry_

    Args:
        policy_id (str): Unique policy id used to access policy information_

    Returns:
        str: Policy details
    """
    print( f"Get Policy Details Invoked For {policy_id}\n" )
    
    policy_details = ( 
        "Doctor visit fee is $100 and Co-Pay is $5." 
        "Member pays 10% of the doctor visit fee."
        "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
    )
    
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
tools = [ get_member_id, get_member_policy_id, get_policy_details, calculator ]

#%%
system_message = (
    "You are a member service representative and can answer questions on member insurance policy."
    "If member id is not specified, always get the member id without asking the user for it."
    "Always use user specified member id to get member's policy id."
    "Using the policy id you can get policy details."
    "All your answers must be based on policy details."
    "Questions involving numbers, you must form correct mathematical expressions based on policy details."
    "You can use a calculator to evaluate mathematical expressions."
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage( content=system_message ),
    MessagesPlaceholder( variable_name="messages" )
])

#%%
llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=12000
)

tools_bound_llm = llm.bind( prompt=prompt ).bind_tools( tools )

#%%
def invoke_llm( graph_state: MessagesState ) -> dict:
    response_message = tools_bound_llm.invoke( graph_state["messages"] )
    return { "messages" : [response_message] }

#%%
graph_builder = StateGraph( MessagesState )

graph_builder.add_node( "invoke_llm", invoke_llm )
graph_builder.add_node( "tools", ToolNode( tools=tools ) )

graph_builder.add_edge( START, "invoke_llm" )
graph_builder.add_conditional_edges( "invoke_llm", tools_condition )
graph_builder.add_edge( "tools", "invoke_llm" )

react_graph = graph_builder.compile()

#%%
question = "What would be my total payment for a doctor visit?"

human_message = HumanMessage( content=question )

graph_state = react_graph.invoke( { "messages": [human_message] } )

llm_response = graph_state["messages"][-1].content

print( f"\nLLM Response:\n\n{llm_response}\n")
# %%
