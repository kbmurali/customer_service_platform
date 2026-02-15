from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

import os

from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

# Initialize MCP server
mcp = FastMCP( 
                name="member-policy-mcp",
                host="0.0.0.0",                                              # bind to all interfaces or "127.0.0.1" if only local
                port=int( os.getenv( "MEMBER_POLICY_MCP_PORT", "8000" )  ),  # default 8000
                stateless_http=True
             )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

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

sample_policy_detail_docs = [
    Document(
        page_content=(
            "Policy ID Policy_abc123_1. "
            "Doctor visit fee is $100 and Co-Pay is $5. "
            "Member pays 10% of the doctor visit fee. "
            "Total amount member pays is the sum of the co-pay and "
            "member portion of the doctor visit fee."
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
            "Total amount member pays is the sum of the co-pay and "
            "member portion of the doctor visit fee."
        ),
        metadata={
            "policy_id": "Policy_xyz789_1",
            "doctor_visit_fee": 150,
            "copay": 10,
            "member_portion_pct": 0.20,
        }
    ),
]

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

@mcp.tool()
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

@mcp.tool()
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


if __name__ == "__main__":
    mcp.run( transport="streamable-http" )
