#%%
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, BaseTool, StructuredTool

from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables import RunnableConfig

import numexpr
import math
import os

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
mcp_client = MultiServerMCPClient(
    {
        "member-policy-mcp": {
            "transport": "streamable_http",        # HTTP-based MCP
            "url": "http://localhost:8000/mcp",    # external URL of your MCP endpoint
            "headers": {
                "Accept": "application/json,text/event-stream",
            },
        }
    }
)

async def load_mcp_tools():
    # Get all tools from the member-policy-mcp server
    tools = await mcp_client.get_tools( server_name="member-policy-mcp" )
    
    return tools

def make_sync_tool( async_tool: BaseTool ) -> BaseTool:
    async_fn = async_tool.coroutine  # the async implementation

    def sync_fn(**kwargs):
        return asyncio.run(async_fn(**kwargs))

    # Rebuild as a Sync Tool with same name/schema
    return StructuredTool.from_function(
        sync_fn,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )
    
#%%
async_mcp_tools = asyncio.run( load_mcp_tools() )

sync_mcp_tools = [ make_sync_tool(t) for t in async_mcp_tools ]

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

# %%
tools = [ get_member_id, calculator, *sync_mcp_tools ]

langsmith_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT")
)

langsmith_tracer = LangChainTracer(
    client=langsmith_client,
    project_name=os.getenv("LANGSMITH_PROJECT", "langchain-primer-demo")
)

trace_config = RunnableConfig(
    callbacks=[langsmith_tracer],
    metadata={ 
                "app": os.getenv("LANGSMITH_PROJECT", "langchain-primer-demo"), 
                "env": "demo" 
             },
    tags=["member-service-demo", "langchain-primer"],
)

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

llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=12000
)
#%%
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message
)

# %%
question = "What would be my total payment for a doctor visit?"

human_message = HumanMessage( content=question )

# %%
result = agent.invoke( { "messages" : [human_message] }, config=trace_config )

print( result["messages"][-1].content )