from __future__ import annotations

import os
import math
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool, Tool

from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables import RunnableConfig

import numexpr
from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
ERROR_PREFIX = "TOOL_ERROR"

def make_langsmith_config(thread_id: str, ls_default_project: str = "agentic-patterns" ) -> RunnableConfig:
    langsmith_client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_ENDPOINT"),
    )
    tracer = LangChainTracer(
        client=langsmith_client,
        project_name=os.getenv("LANGSMITH_PROJECT", ls_default_project ),
    )
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[tracer],
        metadata={
            "app": os.getenv("LANGSMITH_PROJECT", ls_default_project ),
            "env": os.getenv("ENV", "demo"),
            "request_id": thread_id,
        },
        tags=[ os.getenv("LANGSMITH_PROJECT", ls_default_project ), "member-service"],
    )

def wrap_tool( t : Tool ):
    """
    Attach a consistent tool-error handler WITHOUT changing the tool's type.
    """
    try:
        t.handle_tool_error = lambda e: f"{ERROR_PREFIX} in {t.name}: {e}"
    except Exception:
        pass
    return t

def _tool_error_guard(text: str) -> Optional[str]:
    return text if text.strip().startswith(ERROR_PREFIX) else None

#%%
# -----------------------------
# 1) Data + Vector Stores
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

sample_member_policy_docs = [
    Document(
        page_content="Member ID abc123 is associated with policy Policy_abc123_1.",
        metadata={"member_id": "abc123", "policy_id": "Policy_abc123_1"},
    ),
    Document(
        page_content="Member ID xyz789 is associated with policy Policy_xyz789_1.",
        metadata={"member_id": "xyz789", "policy_id": "Policy_xyz789_1"},
    ),
]

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
        },
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
        },
    ),
]

member_policy_db = Chroma.from_documents(
    documents=sample_member_policy_docs,
    embedding=embeddings,
    collection_name="member_policy",
)

policy_details_db = Chroma.from_documents(
    documents=sample_policy_detail_docs,
    embedding=embeddings,
    collection_name="policy_details",
)

enhanced_sample_member_policy_docs = [
    Document(
        page_content="Member ID abc123 is associated with policy Policy_abc123_1.",
        metadata={"member_id": "abc123", "policy_id": "Policy_abc123_1"},
    ),
    Document(
        page_content="Member ID xyz789 is associated with policy Policy_xyz789_1.",
        metadata={"member_id": "xyz789", "policy_id": "Policy_xyz789_1"},
    ),
]

# Extended policy details to make the supervisor routing more interesting.
# We add deductible and a separate "specialist visit" fee as an extra scenario.
enhanced_sample_policy_detail_docs = [
    Document(
        page_content=(
            "Policy ID Policy_abc123_1. "
            "Primary care visit fee is $100. Specialist visit fee is $180. "
            "Co-Pay is $5 for primary care and $20 for specialist. "
            "Member pays 10% of the visit fee after copay. "
            "Annual deductible is $250 (applies to specialist visits only). "
            "Total amount member pays is copay + (member_portion_pct * visit_fee) + deductible_if_applicable."
        ),
        metadata={
            "policy_id": "Policy_abc123_1",
            "primary_fee": 100,
            "specialist_fee": 180,
            "primary_copay": 5,
            "specialist_copay": 20,
            "member_portion_pct": 0.10,
            "deductible": 250,
            "deductible_applies_to": "specialist",
        },
    ),
    Document(
        page_content=(
            "Policy ID Policy_xyz789_1. "
            "Primary care visit fee is $150. Specialist visit fee is $220. "
            "Co-Pay is $10 for primary care and $30 for specialist. "
            "Member pays 20% of the visit fee after copay. "
            "Annual deductible is $0. "
            "Total amount member pays is copay + (member_portion_pct * visit_fee)."
        ),
        metadata={
            "policy_id": "Policy_xyz789_1",
            "primary_fee": 150,
            "specialist_fee": 220,
            "primary_copay": 10,
            "specialist_copay": 30,
            "member_portion_pct": 0.20,
            "deductible": 0,
            "deductible_applies_to": "none",
        },
    ),
]

# Claim “database” (new data to illustrate hierarchy)
sample_claim_docs = [
    Document(
        page_content=(
            "Claim CLM-1001 for member abc123 is STATUS=Pending. "
            "Service: specialist visit. Billed amount: 180. Date: 2025-11-28."
        ),
        metadata={"member_id": "abc123", "claim_id": "CLM-1001", "date_submitted" : "2025-11-28", "status": "Pending", "billed": 180, "service": "specialist"},
    ),
    Document(
        page_content=(
            "Claim CLM-2002 for member xyz789 is STATUS=Paid. "
            "Service: primary visit. Billed amount: 150. Date: 2025-10-18."
        ),
        metadata={"member_id": "xyz789", "claim_id": "CLM-2002", "date_submitted" : "2025-10-18", "status": "Paid", "billed": 150, "service": "primary"},
    ),
    Document(
        page_content=(
            "Claim CLM-2003 for member xyz789 is STATUS=Pending. "
            "Service: primary visit. Billed amount: 150. Date: 2025-11-23."
        ),
        metadata={"member_id": "xyz789", "claim_id": "CLM-2003", "date_submitted" : "2025-11-23", "status": "Pending", "billed": 150, "service": "primary"},
    ),
]

enhanced_member_policy_db = Chroma.from_documents(
    documents=enhanced_sample_member_policy_docs,
    embedding=embeddings,
    collection_name="member_policy_supervisor_demo",
)

enhanced_policy_details_db = Chroma.from_documents(
    documents=enhanced_sample_policy_detail_docs,
    embedding=embeddings,
    collection_name="policy_details_supervisor_demo",
)

claims_db = Chroma.from_documents(
    documents=sample_claim_docs, 
    embedding=embeddings,
    collection_name="hn_claims"
)

#%%
# -----------------------------
# 2) Tools (Specific agents will use them)
# -----------------------------
@tool
def get_member_id() -> str:
    """
    Gets a member id when user's member id is required_summary_

    Returns:
        str: member id
    """
    member_id = input("What is your member id? ").strip()
    
    if not member_id:
        raise ValueError("Empty member id provided.")
    
    return member_id

@tool
def get_member_policy_id(member_id: str) -> str:
    """Retrieve a member's policy_id from the member_policy Chroma store."""
    docs = member_policy_db.similarity_search(member_id, k=1)
    if not docs or docs[0].metadata.get("member_id") != member_id:
        raise ValueError(f"No policy found for member id: {member_id}")
    policy_id = docs[0].metadata.get("policy_id")
    if not policy_id:
        raise ValueError(f"Policy id missing for member id: {member_id}")
    return policy_id

@tool
def get_policy_details(policy_id: str) -> str:
    """Retrieve policy details text from the policy_details Chroma store."""
    docs = policy_details_db.similarity_search(policy_id, k=1)
    if not docs or docs[0].metadata.get("policy_id") != policy_id:
        raise ValueError(f"No policy details found for policy id: {policy_id}")
    return docs[0].page_content

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression (e.g., '0.10 * 100 + 5')."""
    math_constants = {"pi": math.pi, "i": 1j, "e": math.e}
    result = numexpr.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

@tool
def get_policy_metadata(policy_id: str) -> dict:
    """Retrieve policy metadata from the policy_details Chroma store."""
    docs = policy_details_db.similarity_search(policy_id, k=1)
    if not docs or docs[0].metadata.get("policy_id") != policy_id:
        raise ValueError(f"No policy details found for policy id: {policy_id}")
    return docs[0].metadata

@tool
def search_policy_knowledge(policy_id: str, query: str) -> List[str]:
    """
    Retrieve the most relevant policy passages for the user's question,
    restricted to the current policy_id.
    """
    docs = policy_details_db.similarity_search(query, k=3, filter={"policy_id": policy_id})
    return [d.page_content for d in docs]

@tool
def get_enhanced_member_policy_id(member_id: str) -> str:
    """Retrieve a member's policy_id from the member_policy Chroma store."""
    docs = enhanced_member_policy_db.similarity_search(member_id, k=1)
    if not docs or docs[0].metadata.get("member_id") != member_id:
        raise ValueError(f"No policy found for member id: {member_id}")
    policy_id = docs[0].metadata.get("policy_id")
    if not policy_id:
        raise ValueError(f"Policy id missing for member id: {member_id}")
    return policy_id

@tool
def get_enhanced_policy_details(policy_id: str) -> str:
    """Retrieve policy details text from the policy_details Chroma store."""
    docs = enhanced_policy_details_db.similarity_search(policy_id, k=1)
    if not docs or docs[0].metadata.get("policy_id") != policy_id:
        raise ValueError(f"No policy details found for policy id: {policy_id}")
    return docs[0].page_content

@tool
def get_latest_claim(member_id: str) -> str:
    """Return the most relevant claim text for a member_id (toy: similarity search)."""
    docs = claims_db.similarity_search(member_id, k=5)
    
    if not docs:
        raise ValueError(f"No claims found for member_id={member_id}")
    
    results = [ doc for doc in docs if doc.metadata.get("member_id") == member_id ]
    
    if not results:
        raise ValueError(f"No claims found for member_id={member_id}")
    
    sorted_results = sorted( results, key=lambda doc: doc.metadata.get("date_submitted"), reverse=True )
    
    return sorted_results[0].page_content