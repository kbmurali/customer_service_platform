#%%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
prompt_message = (
    "You are a member service representative."
    "Your task is to return membership status based on user specified member id."
    "If you do not know the membership status, respond with GET STATUS: <member id>."
    "If you know the membership status, respond with STATUS: <final message>."
    "HUMAN: {member_id}\n"
)

prompt = PromptTemplate.from_template( prompt_message )

#%%
llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=12000
)

#%%
mem_id = "123abc"

llm_response = (prompt | llm).invoke( {"member_id": mem_id} )

#%%
print( f"LLM Response Content:\n" )
print( f"{llm_response.content}\n" )
# %%
new_prompt_message = (
    "You are a member service representative."
    "Your task is to return membership status based on user specified member id."
    "If you do not know the membership status, respond with GET STATUS: <member id>."
    "If you know the membership status, respond with STATUS: <final message>."
    "HUMAN: {member_id}\n"
    "AI: {query}\n"
    "QUERY RESULT : {result}\n"
)

new_prompt = PromptTemplate.from_template( new_prompt_message )

llm_response2 = (new_prompt | llm).invoke( {
    "member_id": mem_id,
    "query" : llm_response.content,
    "result" : "Platinum"
} )

#%%
print( f"LLM Response2 Content:\n" )
print( f"{llm_response2.content}\n" )
# %%
