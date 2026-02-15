#%%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
def get_membership_status( member_id : str ) -> str:
    print( f"Function get_membership_status invoked for member id: {member_id}\n" )
    return "Platinum"

#%%
openapi_tool_spec = {
    "type" : "object",
    "title": "get_membership_status",
    "description" : "Given member id, return the member's membership status",
    "properties" : {
        "member_id" : {
            "title": "member_id",
            "description" : "Unique membership id of a member",
            "type": "string"
        }
    },
    "required": [ "member_id" ]
}

#%%
system_message = (
    "You are a member service representative."
    "Your task is to get membership status based on user specified member id."
    "Once you get membership status, return the status to the user."
    "Always use the tool specified to get the membership status."
)

prompt = ChatPromptTemplate.from_messages( [
    SystemMessage( content=system_message ),
    MessagesPlaceholder( variable_name="messages" )
])

#%%
llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=12000
)

tool_bound_llm = llm.bind( prompt=prompt ).bind_tools( [openapi_tool_spec] )

#%%
mem_id = "123abc"
human_message = HumanMessage( content=mem_id )

llm_response = tool_bound_llm.invoke( [human_message] )

#%%
print( f"LLM Response Content: {llm_response.content}\n" )
print( f"LLM Response Tool Calls Message \n" )
print( f"{llm_response.tool_calls}" )
# %%
tool_func_map = {
    "get_membership_status" : get_membership_status
}

tool_call = llm_response.tool_calls[0]
tool_name = tool_call[ "name" ]
tool = tool_func_map[ tool_name ]
tool_result = tool( **tool_call[ "args" ] )

tool_message = ToolMessage( content=tool_result, tool_call_id=tool_call["id"] )

llm_response = tool_bound_llm.invoke( [
    human_message, llm_response, tool_message  
] )

#%%
print( f"LLM Response Content: {llm_response.content}\n" )
print( f"LLM Response Tool Calls Message \n" )
print( f"{llm_response.tool_calls}" )
# %%
