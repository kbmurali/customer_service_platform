#%%
from neo4j import GraphDatabase
from graphql import graphql_sync, GraphQLObjectType, build_ast_schema, parse as parse_sdl
from neo4j_graphql_shim import neo4j_graphql

from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os
import json
import logging

from typing import List, Optional
from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
if not os.getenv( "NEO4J_URI", None ):
    raise ValueError( "ERROR: Neo4j BOLT URI not set in the runtime environment!!" )

if not os.getenv( "NEO4J_USER", None ):
    raise ValueError( "ERROR: Neo4j USERNAME not set in the runtime environment!!" )

if not os.getenv( "NEO4J_PASSWORD", None ):
    raise ValueError( "ERROR: Neo4j PASSWORD not set in the runtime environment!!" )

URI = os.getenv( "NEO4J_URI" )
AUTH = ( os.getenv( "NEO4J_USER" ), os.getenv( "NEO4J_PASSWORD" ) )
driver = GraphDatabase.driver( URI, auth=AUTH )

#%%
# Configure logging to capture output from the library
logging.basicConfig()
logging.getLogger("neo4j_graphql_py").setLevel(logging.DEBUG)  # shim uses same logger name

#%%
def make_executable_schema(type_defs):
    return build_ast_schema(parse_sdl(type_defs))
    
#%%
# --- The GraphQL Schema ---
# This strict schema defines exactly what the LLM is allowed to query.
type_defs = """
directive @relation(
    name: String, 
    direction: String, 
    from: String, 
    to: String
) on FIELD_DEFINITION | OBJECT

type Member {
  member_id: ID!
  name: String
  
  # Allow nested filtering on policies
  policies(type: String): [Policy] 
    @relation(name: "INSURED_BY", direction: OUT)
    
  # Allow nested filtering on claims
  claims(status: String, service: String, amount: Float): [Claim] 
    @relation(name: "FILED", direction: OUT)
}

type Policy {
  policy_id: ID!
  type: String
  limit: Float
  
  # Allow filtering members by name inside a policy
  members(name: String): [Member] 
    @relation(name: "INSURED_BY", direction: IN)
}

type Claim {
  claim_id: ID!
  amount: Float
  status: String # Status values are Pending, Paid, Denied
  service: String
  diagnosis(description: String): Diagnosis @relation(name: "FOR_DIAGNOSIS", direction: OUT)
  filed_by: Member @relation(name: "FILED", direction: IN)
}

type Diagnosis {
  code: ID!
  description: String
}

type Query {
  # Entry points for the LLM
  members(name: String, member_id: ID): [Member]
  policies(policy_id: ID): [Policy]
  claims(status: String, service: String): [Claim]
}
"""

#%%
# Create the executable schema
schema = make_executable_schema( type_defs )

#%%
# --- The "Magic" Resolver ---
# Instead of writing Cypher manually, we just call neo4j_graphql()
# It inspects the requested fields and builds the optimal Cypher query.
def generic_resolver(object, info, **kwargs):
    return neo4j_graphql(object, info.context, info, **kwargs)

query_type: GraphQLObjectType = schema.type_map["Query"]

for field_name in query_type.fields:
    query_type.fields[field_name].resolve = generic_resolver

#%%
@tool
def query_knowledge_graph(question: str) -> str:
    """
    Useful for answering questions about relationships between Members, Policies, Claims, and Diagnoses.
    Input should be a natural language question.
    """
    # Chain to generate GraphQL from Question
    # We inject the actual GraphQL Schema into the prompt
    graphql_generation_template = """
    Task: You are a Query Specialist. Convert the user's question into a GraphQL query matching the following schema.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    
    Schema:
    {schema}
    
    Business Logic:
    - "Family" means all members connected to the same policy.
    - To find a family's total, fetch the claims for the user AND all members on their policies.
    Example:
    
    Question: "How much has the family of John Doe claimed for Hypertension?"
    Logic: 
      - Find Member 'John Doe'
      - Find Member's Policy
      - Find ALL Members on that Policy (the family)
      - Find their Claims for Diagnosis 'Hypertension'
      - Sum the amount
      
    Example:
    Question: "What claims has John Doe filed?"
    GraphQL:
    query {{
    members(name: "John Doe") {{
        name
        claims {{
        amount
        status
        service
        }}
    }}
    }}
    
    

          
    The question is:
    {question}
    
    Respond ONLY with the GraphQL query string.
    """
    
    graphql_prompt = ChatPromptTemplate.from_template(graphql_generation_template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Generate the query
    chain = graphql_prompt | llm
    response = chain.invoke({"schema": type_defs, "question": question})
    
    generated_graphql = response.content.replace("```graphql", "").replace("```", "").strip()
    
    return generated_graphql

#%%
#q = "Find all pending claims of 'Alice Smith' and her family?"

#q = "What is the total amount of all claims of 'Alice Smith' and her family?"

#q = "How much has the family of Charlie Jones claimed for Hypertension?"

q = "Find all claims of 'Alice Smith' and her family for Hypertension?"
graphql_query = query_knowledge_graph(q)

print( graphql_query )


#%%
"""
Executes a GraphQL query against the database.
Input must be a valid GraphQL query string based on the defined schema.
"""
try:
    # We pass the Neo4j driver into the context so the library can use it
    context = {"driver": driver}
    
    # Execute using standard GraphQL engine
    result = graphql_sync(schema, graphql_query, context_value=context)
    
    if result.errors:
        print( f"GraphQL Error: {result.errors}" )
        
    print(  json.dumps(result.data, indent=2) )
except Exception as e:
    print( f"Execution Error: {str(e)}" )
# %%
