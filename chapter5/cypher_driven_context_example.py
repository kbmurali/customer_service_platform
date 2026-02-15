#%%
from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

import os
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

#%%
graph = Neo4jGraph(
    url = os.getenv( "NEO4J_URI" ), 
    username = os.getenv( "NEO4J_USER" ), 
    password = os.getenv( "NEO4J_PASSWORD" )
)

# Refresh schema to ensure LLM has latest context
graph.refresh_schema()

#%%
@tool
def query_knowledge_graph(question: str) -> str:
    """
    Useful for answering questions about relationships between Members, Policies, Claims, and Diagnoses.
    Input should be a natural language question.
    """
    # 1. Chain to generate Cypher from Question
    # We inject the actual Graph Schema into the prompt
    cypher_generation_template = """
    Task: Generate Cypher statement to query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    
    Schema:
    {schema}
    
    Business Logic & Rules:
    1. **Family/Dependents**: Defined as distinct Members who are insured by (`:INSURED_BY`) the same Policy.
    2. **Total Claims**: When asked for a "total" or "sum", use the Cypher `sum()` function on the `amount` property.
    3. **Case Sensitivity**: String matching (like names) is case-sensitive. Use `CONTAINS` for flexibility.
    
    Examples:
    
    Question: "How much has the family of John Doe claimed for Hypertension?"
    Cypher Logic: 
      - Find Member 'John Doe'
      - Find Member's Policy
      - Find ALL Members on that Policy (the family)
      - Find their Claims for Diagnosis 'Hypertension'
      - Sum the amount
    Query:
      MATCH (m:Member {{name: "John Doe"}})-[:INSURED_BY]->(p:Policy)
      MATCH (p)<-[:INSURED_BY]-(family:Member)
      MATCH (family)-[:FILED]->(c:Claim)-[:FOR_DIAGNOSIS]->(d:Diagnosis)
      WHERE d.description CONTAINS "Hypertension"
      RETURN sum(c.amount) as total_amount
    
    Question: "Find all policies covering Bob."
    Query:
      MATCH (m:Member)-[:INSURED_BY]->(p:Policy)
      WHERE m.name CONTAINS "Bob"
      RETURN p.policy_id as policy_id, p.type as policy_type, p.limit as policy_limit

    Question: "Find all pending claims of John Doe."
    Query:
      MATCH (m:Member {{name: "John Doe"}})-[:FILED]->(c:Claim)
      WHERE c.status="Pending"
      RETURN c.claim_id as claim_id, c.service as service, c.date as date, c.amount as amount, c.status as status
          
    The question is:
    {question}
    
    Cypher Query (Return ONLY the query, no markdown):
    """
    
    cypher_prompt = ChatPromptTemplate.from_template(cypher_generation_template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 2. Generate the query
    chain = cypher_prompt | llm
    response = chain.invoke({"schema": graph.get_schema, "question": question})
    
    print( response.content )
    
    generated_cypher = response.content.replace("```cypher", "").replace("```", "").strip()
    
    #print(f"DEBUG: Generated Cypher: {generated_cypher}")

    # 3. Execute against the Graph
    try:
        context_data = graph.query(generated_cypher)
        return json.dumps(context_data, indent=2)
    except Exception as e:
        return f"Error executing graph query: {e}"

#%%
q = "How much total money has the family of 'Alice Smith' claimed?"

print( query_knowledge_graph(q) )
# %%
q = "What is the total amount of all pending claims of 'Alice Smith' and her family?"

print( query_knowledge_graph(q) )
# %%
q = "Find all claims of 'Alice Smith' and her family?"

print( query_knowledge_graph(q) )
# %%
q = "Find all paid claims of 'Charlie Jones' and his family?"

print( query_knowledge_graph(q) )
# %%
