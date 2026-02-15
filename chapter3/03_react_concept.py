#%%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langgraph.graph import StateGraph, START, END, MessagesState
import numexpr
import math

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
def get_member_id() -> str:
    print( f"Get Member Id Invoked\n" )
    member_id = input( "What is your member id?")
    return member_id.strip()

def get_member_policy_id( member_id : str ) -> str:
    print( f"Get Member Policy Id Invoked For {member_id}\n" )
    return f"Policy_{member_id}_1"

def get_policy_details( policy_id : str ) -> str:
    print( f"Get Policy Details Invoked For {policy_id}\n" )
    
    policy_details = ( 
        "Doctor visit fee is $100 and Co-Pay is $5." 
        "Member pays 10% of the doctor visit fee."
        "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
    )
    
    return policy_details

def calculator( expression: str ) -> str:
    print( f"CALCULATOR INVOKED WITH EXPR: {expression}\n" )
    
    math_constants = { "pi" : math.pi, "i": 1j, "e":math.exp }
    
    result = numexpr.evaluate( expression.strip(), local_dict=math_constants )
    
    return str( result )

#%%
member_id_tool = {
    "type": "object",
    "title": "get_member_id",
    "description": "Gets a member id when user's member id is required",
    "properties" : {},
    "required": []
}

member_policy_id_tool = {
    "type": "object",
    "title": "get_member_policy_id",
    "description": "Gets policy id of a member given member id",
    "properties" : {
        "member_id" : {
            "type" : "string",
            "title" : "member_id",
            "description": "Unique member id used to access member information"
        }
    },
    "required": [ "member_id" ]
}

policy_details_tool = {
    "type": "object",
    "title": "get_policy_details",
    "description": "Gets policy details containing fee, co-pay, and member payment information given policy id",
    "properties" : {
        "policy_id" : {
            "type" : "string",
            "title" : "policy_id",
            "description": "Unique policy id used to access policy information"
        }
    },
    "required": [ "policy_id" ]
}

calculator_tool = {
    "type": "object",
    "title": "calculator",
    "description": "Calculates a mathematical expression comprising of \
                    terms with mathematical operations such as \
                    arthmetic, exponential, power of, and logarthmic.\
                    Example: 0.01 * 100 + 5 + 4**2",
    "properties" : {
        "expression" : {
            "type" : "string",
            "title" : "expression",
            "description": "Mathematic expression that needs to be evaluated and calculated"
        }
    },
    "required": [ "expression" ]
}

tool_specs = [ member_id_tool, member_policy_id_tool, policy_details_tool, calculator_tool ]

tool_func_map = {
    "get_member_id" : get_member_id,
    "get_member_policy_id" : get_member_policy_id,
    "get_policy_details" : get_policy_details,
    "calculator" : calculator
}
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

tools_bound_llm = llm.bind( prompt=prompt ).bind_tools( tool_specs )

#%%
def invoke_llm( graph_state: MessagesState ) -> dict:
    response_message = tools_bound_llm.invoke( graph_state["messages"] )
    return { "messages" : [response_message] }

def decide_to_call_tools( graph_state: MessagesState ):
    if graph_state["messages"][-1].tool_calls:
        return "tools"
    
    return END

def call_tools( graph_state: MessagesState ) -> dict:
    tool_calls = graph_state["messages"][-1].tool_calls
    
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call[ "name" ]
        
        if tool_name in tool_func_map.keys():
            tool = tool_func_map[ tool_name ]
            tool_result = tool( **tool_call[ "args" ] )
            tool_message = ToolMessage( content=tool_result, tool_call_id=tool_call[ "id" ] )
            tool_messages.append( tool_message )
        else:
            raise ValueError( f"Undefined tool: {tool_name}" )
        
    return { "messages" : tool_messages }

#%%
graph_builder = StateGraph( MessagesState )

graph_builder.add_node( "invoke_llm", invoke_llm )
graph_builder.add_node( "tools", call_tools )

graph_builder.add_edge( START, "invoke_llm" )
graph_builder.add_conditional_edges( "invoke_llm", decide_to_call_tools )
graph_builder.add_edge( "tools", "invoke_llm" )

react_graph = graph_builder.compile()

#%%
question = "What would be my total payment for a doctor visit?"

human_message = HumanMessage( content=question )

graph_state = react_graph.invoke( { "messages": [human_message] } )

llm_response = graph_state["messages"][-1].content

print( f"\nLLM Response:\n\n{llm_response}\n")
# %%
