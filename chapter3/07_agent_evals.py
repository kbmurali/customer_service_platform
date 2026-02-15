#%%
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import statistics
import numexpr
import math
import os

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
    Document(
        page_content="Member ID test001 is associated with policy Policy_test001_1.",
        metadata={"member_id": "test001", "policy_id": "Policy_test001_1"}
    )
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
    Document(
        page_content=(
            "Policy ID Policy_test001_1. "
            "Doctor visit fee is $120 and Co-Pay is $8. "
            "Member pays 15% of the doctor visit fee. "
            "Total amount member pays is the sum of the co-pay and member portion of the doctor visit fee."
        ),
        metadata={
            "policy_id": "Policy_test001_1",
            "doctor_visit_fee": 120,
            "copay": 8,
            "member_portion_pct": 0.15,
        }
    )
]

# Optional: set a persistent directory if you want to reuse the store between runs
persist_dir_member_policy = os.path.join("chroma_test_store", "member_policy")
persist_dir_policy_details = os.path.join("chroma_test_store", "policy_details")

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

#%%
# ============================================================================
# EVALUATION DATASET
# ============================================================================
@dataclass
class EvalCriteria:
    """A single evaluation example"""
    query: str
    expected_output: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    expected_values: Optional[List[str]] = None  # Values that should appear in response
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    
EVALUATION_DATASET = [
    # Simple queries
    EvalCriteria(
        query="What is the policy ID for member abc123?",
        expected_output="Policy_abc123_1",
        expected_tools=["get_member_policy_id"],
        expected_values=["Policy_abc123_1"],
        category="policy_lookup",
        difficulty="easy"
    ),
    
    EvalCriteria(
        query="Get policy details for member xyz789",
        expected_tools=["get_member_policy_id", "get_policy_details"],
        expected_values=["$150", "$10", "20%"],
        category="policy_details",
        difficulty="medium"
    ),
    
    # Complex calculation queries
    EvalCriteria(
        query="How much would member abc123 pay for a doctor visit?",
        expected_tools=["get_member_policy_id", "get_policy_details", "calculator"],
        expected_values=["15", "$15"],  # $5 copay + $10 (10% of $100)
        category="calculation",
        difficulty="hard"
    ),
    
    EvalCriteria(
        query="Calculate the total out-of-pocket cost for member xyz789 for a doctor visit",
        expected_tools=["get_member_policy_id", "get_policy_details", "calculator"],
        expected_values=["40", "$40"],  # $10 copay + $30 (20% of $150)
        category="calculation",
        difficulty="hard"
    ),
    
    # Edge cases
    EvalCriteria(
        query="What is the policy for member invalid999?",
        expected_values=["not found", "error", "unable"],
        category="error_handling",
        difficulty="medium"
    ),
]

@dataclass
class SingleCriteriaEvalMetrics:
    """Metrics for a single evaluation run"""
    query: str
    success: bool
    response: str
    latency_seconds: float
    tool_calls: List[str]
    expected_values_found: List[bool]
    accuracy_score: float  # 0.0 to 1.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all evaluations"""
    total_examples: int
    successful: int
    failed: int
    success_rate: float
    avg_latency: float
    median_latency: float
    avg_accuracy: float
    total_tool_calls: int
    avg_tool_calls_per_query: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
def eval_response_accuracy( response: str, expected_values: Optional[List[str]] = None ) -> float:
    """
    Evaluate response accuracy based on expected values.
    
    Returns accuracy score from 0.0 to 1.0
    """
    if not expected_values:
        return 1.0  # No specific expectations
    
    response_lower = response.lower()
    matches = []
    
    for expected in expected_values:
        found = expected.lower() in response_lower
        matches.append(found)
    
    # Accuracy is percentage of expected values found
    accuracy = sum(matches) / len(matches) if matches else 0.0
    
    return accuracy

def eval_single_criteria( llm_agent, eval_criteria: EvalCriteria ) -> SingleCriteriaEvalMetrics:
    """
    Run evaluation on a single example.
    
    Returns detailed metrics for this evaluation.
    """
    print(f"\n[Evaluating] {eval_criteria.query[:60]}...")
    
    start_time = time.time()
    
    try:
        result = llm_agent.invoke({ "messages": [HumanMessage(content=eval_criteria.query)] })
        
        end_time = time.time()
        
        latency = end_time - start_time
        
        response = result["messages"][-1].content
        
        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc['name'])
        
        # Evaluate accuracy
        expected_values_found = []
        if eval_criteria.expected_values:
            for val in eval_criteria.expected_values:
                found = val.lower() in response.lower()
                expected_values_found.append(found)
        
        accuracy = eval_response_accuracy(response, eval_criteria.expected_values)
        
        success = accuracy >= 0.7  # Consider success if at least 70% accuracy
        
        metrics = SingleCriteriaEvalMetrics(
            query=eval_criteria.query,
            success=success,
            response=response,
            latency_seconds=latency,
            tool_calls=tool_calls,
            expected_values_found=expected_values_found,
            accuracy_score=accuracy,
            error=None
        )
        
        print(f"Success: {success} | Accuracy: {accuracy:.2f} | Latency: {latency:.2f}s")
        
        return metrics
        
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"Error: {str(e)}")
        
        return SingleCriteriaEvalMetrics(
            query=eval_criteria.query,
            success=False,
            response="",
            latency_seconds=latency,
            tool_calls=[],
            expected_values_found=[],
            accuracy_score=0.0,
            error=str(e)
        )

#%%
all_metrics = []
    
for eval_criteria in EVALUATION_DATASET:
    metrics = eval_single_criteria( agent, eval_criteria )
    all_metrics.append( metrics )

#%%
# Aggregate metrics
successful = sum(1 for m in all_metrics if m.success)
failed = len(all_metrics) - successful
success_rate = successful / len(all_metrics) if all_metrics else 0.0

accuracies = [m.accuracy_score for m in all_metrics]
avg_accuracy = statistics.mean(accuracies) if accuracies else 0.0
    
latencies = [m.latency_seconds for m in all_metrics]
avg_latency = statistics.mean(latencies) if latencies else 0.0
median_latency = statistics.median(latencies) if latencies else 0.0

total_tool_calls = sum(len(m.tool_calls) for m in all_metrics)
avg_tool_calls = total_tool_calls / len(all_metrics) if all_metrics else 0.0

aggregated = AggregatedMetrics(
        total_examples=len(all_metrics),
        successful=successful,
        failed=failed,
        success_rate=success_rate,
        avg_latency=avg_latency,
        median_latency=median_latency,
        avg_accuracy=avg_accuracy,
        total_tool_calls=total_tool_calls,
        avg_tool_calls_per_query=avg_tool_calls
    )

#%%
print(f"\n{'=' * 80}")
print("EVALUATION REPORT")
print(f"{'=' * 80}\n")

print(f"Overall Performance:")
print(f"  Total Examples: {aggregated.total_examples}")
print(f"  Successful: {aggregated.successful}")
print(f"  Failed: {aggregated.failed}")
print(f"  Success Rate: {aggregated.success_rate * 100:.1f}%")
print(f"  Average Accuracy: {aggregated.avg_accuracy * 100:.1f}%\n")

print(f"Latency Metrics:")
print(f"  Average: {aggregated.avg_latency:.2f}s")
print(f"  Median: {aggregated.median_latency:.2f}s")

print(f"Tool Usage:")
print(f"  Total Tool Calls: {aggregated.total_tool_calls}")
print(f"  Avg per Query: {aggregated.avg_tool_calls_per_query:.1f}\n")
# %%
