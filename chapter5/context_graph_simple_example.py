"""
LangChain + Neo4j demo: clearly separating Knowledge Graph (KG) vs Context Graph (CG)

KG (stable, reusable):
  - plans, networks, providers, procedures, benefits, rules (coverage/prior-auth/network)
CG (session-specific, fast-changing):
  - session, plan context, goals, tool calls, errors, decision

CG hierarchy:
  Session -[:HAS_USER]-> User
  Session -[:HAS_MESSAGE]-> UserMessage
  Session -[:HAS_PLAN]-> Plan
      Plan -[:HAS_GOAL]-> Goal (priority 1)
          Goal -[:HAS_TOOL_CALL]-> ToolCall
                                       ToolCall -[:HAS_ERROR]-> Error  (if failed)
      Plan -[:HAS_GOAL]-> Goal (priority 2)
          ...
  Session -[:HAS_DECISION]-> Decision

This script supports TWO modes:

A) Deterministic agent (NO LLM required) [default]
   - Uses LangChain Tools (Python functions) but chooses tool calls via simple code logic.

B) LLM tool-calling agent (optional)
   - Uses a tool-calling chat model (e.g., ChatOpenAI) to decide tool calls.
   - Requires OPENAI_API_KEY in .env.

Run:
  python cg_kg_example1_refactored.py

Neo4j Browser:
  http://localhost:7474
"""

#%%
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# LangChain core/tools
from langchain_core.tools import tool

# Neo4j graph wrapper
from langchain_community.graphs import Neo4jGraph

# Agent imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

# Optional LLM (only needed in LLM mode)
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv, find_dotenv

load_dotenv( find_dotenv() )

#%%
if not os.getenv( "NEO4J_URI", None ):
    raise ValueError( "ERROR: Neo4j BOLT URI not set in the runtime environment!!" )

if not os.getenv( "NEO4J_USER", None ):
    raise ValueError( "ERROR: Neo4j USERNAME not set in the runtime environment!!" )

if not os.getenv( "NEO4J_PASSWORD", None ):
    raise ValueError( "ERROR: Neo4j PASSWORD not set in the runtime environment!!" )

#%%
GRAPH: Optional[Neo4jGraph] = None

GRAPH = Neo4jGraph(
    url      = os.getenv( "NEO4J_URI" ),
    username = os.getenv( "NEO4J_USER" ),
    password = os.getenv( "NEO4J_PASSWORD" )
)

# %%
# ---------------------------
# Domain objects
# ---------------------------
@dataclass(frozen=True)
class MemberRequest:
    member_id: str
    plan_id: str
    provider_id: str
    procedure_code: str  # CPT, e.g., "29827"
    diagnosis_code: str  # ICD-10, e.g., "M75.1"
    requested_date: str  # YYYY-MM-DD
    location_state: str

@dataclass
class AgentResult:
    covered: bool
    prior_auth_required: bool
    in_network: bool
    explanation: str
    missing_info: Tuple[str, ...]

#%%
def reset_demo_data(neo4j_graph: Neo4jGraph) -> None:
    neo4j_graph.query("MATCH (n:KG) DETACH DELETE n;")
    neo4j_graph.query("MATCH (n:CG) DETACH DELETE n;")

def seed_knowledge_graph(neo4j_graph: Neo4jGraph) -> None:
    """
    Build KG: stable facts and rules.
    Everything has :KG plus a semantic label.

    - Diagnosis nodes participate via:
        (Diagnosis)-[:JUSTIFIES]->(Procedure)
      and an example exclusion:
        (Plan)-[:EXCLUDES_DIAGNOSIS]->(Diagnosis)
    """
    neo4j_graph.query(
        """
        // Plans
        MERGE (planA:KG:Plan {plan_id: "PLAN_A", name: "Acme Gold PPO"})
        MERGE (planB:KG:Plan {plan_id: "PLAN_B", name: "Acme Basic HMO"})

        // Networks
        MERGE (ppoNet:KG:Network {network_id: "NET_PPO", name: "Acme PPO Network"})
        MERGE (hmoNet:KG:Network {network_id: "NET_HMO", name: "Acme HMO Network"})
        MERGE (planA)-[:USES_NETWORK]->(ppoNet)
        MERGE (planB)-[:USES_NETWORK]->(hmoNet)

        // Providers
        MERGE (prov1:KG:Provider {provider_id: "PROV_100", name: "Lakeside PT Clinic"})
        MERGE (prov2:KG:Provider {provider_id: "PROV_200", name: "Downtown Specialty Hospital"})

        // Network participation (stable knowledge)
        MERGE (prov1)-[:IN_NETWORK]->(ppoNet)
        MERGE (prov1)-[:IN_NETWORK]->(hmoNet)
        MERGE (prov2)-[:IN_NETWORK]->(ppoNet)
        // prov2 NOT in HMO network

        // Procedures (CPT)
        MERGE (p97110:KG:Procedure {code: "97110", code_system: "CPT", name: "Therapeutic exercises"})
        MERGE (p29827:KG:Procedure {code: "29827", code_system: "CPT", name: "Arthroscopic shoulder rotator cuff repair"})

        // Diagnoses (ICD-10)
        MERGE (dM545:KG:Diagnosis {code: "M54.5", code_system: "ICD-10", name: "Low back pain"})
        MERGE (dM751:KG:Diagnosis {code: "M75.1", code_system: "ICD-10", name: "Rotator cuff tear"})

        // Benefits
        MERGE (benefitPT:KG:Benefit {benefit_id: "BEN_PT", name: "Physical Therapy"})
        MERGE (benefitSurg:KG:Benefit {benefit_id: "BEN_SURG", name: "Outpatient Surgery"})

        // Procedure -> Benefit mapping
        MERGE (p97110)-[:MAPS_TO_BENEFIT]->(benefitPT)
        MERGE (p29827)-[:MAPS_TO_BENEFIT]->(benefitSurg)

        // Plan coverage (stable)
        MERGE (planA)-[:COVERS_BENEFIT]->(benefitPT)
        MERGE (planA)-[:COVERS_BENEFIT]->(benefitSurg)

        MERGE (planB)-[:COVERS_BENEFIT]->(benefitPT)
        MERGE (planB)-[:COVERS_BENEFIT {limited: true, note: "Only if medical necessity criteria met"}]->(benefitSurg)

        // Prior auth rules (stable)
        MERGE (rulePA1:KG:Rule {rule_id: "RULE_PA_29827_A", type: "PRIOR_AUTH",
          description: "CPT 29827 requires prior authorization for PLAN_A"})
        MERGE (rulePA2:KG:Rule {rule_id: "RULE_PA_29827_B", type: "PRIOR_AUTH",
          description: "CPT 29827 requires prior authorization for PLAN_B"})
        MERGE (planA)-[:HAS_RULE]->(rulePA1)
        MERGE (planB)-[:HAS_RULE]->(rulePA2)
        MERGE (rulePA1)-[:APPLIES_TO_PROCEDURE]->(p29827)
        MERGE (rulePA2)-[:APPLIES_TO_PROCEDURE]->(p29827)

        // Diagnosis -> Procedure medical necessity / compatibility
        MERGE (dM751)-[:JUSTIFIES]->(p29827)
        MERGE (dM545)-[:JUSTIFIES]->(p97110)

        // Example exclusion: PLAN_B excludes M75.1
        MERGE (planB)-[:EXCLUDES_DIAGNOSIS {note: "Example exclusion for demo"}]->(dM751)
        """
    )

#%%
# ---------------------------
# Context Graph (session state)
# ---------------------------

def create_context_session(neo4j_graph: Neo4jGraph, request: MemberRequest, user_text: str) -> Tuple[str, str]:
    """
    Create a new Context Graph session.

    Structure created:
        Session -[:HAS_USER]->    User
        Session -[:HAS_MESSAGE]-> UserMessage
        Session -[:HAS_PLAN]->    Plan   (CG plan node carrying session-scoped request context)

    Goals are NOT added here; they hang off the CG Plan node and are written
    in cg_add_dynamic_goals() after LLM planning.

    Returns:
        (session_id, cg_plan_id)  — both IDs are needed by downstream functions.
    """
    session_id  = f"SES_{uuid.uuid4().hex[:10].upper()}"
    msg_id      = f"MSG_{uuid.uuid4().hex[:8].upper()}"
    cg_plan_id  = f"CGPLAN_{uuid.uuid4().hex[:8].upper()}"

    neo4j_graph.query(
        """
        // Session root
        MERGE (s:CG:Session {session_id: $session_id})
          ON CREATE SET s.created_at = timestamp(),
                        s.status     = "ACTIVE",
                        s.domain     = "HealthInsuranceCustomService"

        // User node
        MERGE (u:CG:User {member_id: $member_id})
        MERGE (s)-[:HAS_USER]->(u)

        // User message
        MERGE (m:CG:UserMessage {message_id: $msg_id})
          ON CREATE SET m.text       = $user_text,
                        m.created_at = timestamp()
        MERGE (s)-[:HAS_MESSAGE]->(m)

        // CG Plan node — carries the session-scoped request context.
        // Distinct from the KG Plan node; this is ephemeral session state.
        MERGE (p:CG:Plan {cg_plan_id: $cg_plan_id})
          ON CREATE SET p.plan_id        = $plan_id,
                        p.provider_id    = $provider_id,
                        p.procedure_code = $procedure_code,
                        p.diagnosis_code = $diagnosis_code,
                        p.requested_date = $requested_date,
                        p.location_state = $location_state,
                        p.created_at     = timestamp()
        MERGE (s)-[:HAS_PLAN]->(p)
        """,
        {
            "session_id"     : session_id,
            "member_id"      : request.member_id,
            "msg_id"         : msg_id,
            "user_text"      : user_text,
            "cg_plan_id"     : cg_plan_id,
            "plan_id"        : request.plan_id,
            "provider_id"    : request.provider_id,
            "procedure_code" : request.procedure_code,
            "diagnosis_code" : request.diagnosis_code,
            "requested_date" : request.requested_date,
            "location_state" : request.location_state,
        },
    )
    return session_id, cg_plan_id


def cg_add_dynamic_goals(
    neo4j_graph: Neo4jGraph,
    cg_plan_id: str,
    goals: List[Dict[str, Any]],
) -> List[str]:
    """
    Attach LLM-planned goals to the CG Plan node.

    Structure written:
        Plan -[:HAS_GOAL]-> Goal

    Returns:
        Ordered list of goal_ids (same order as input goals, sorted by priority).
    """
    goal_ids: List[str] = []
    for goal in sorted(goals, key=lambda g: g["priority"]):
        goal_id = f"GOAL_{uuid.uuid4().hex[:8].upper()}"
        neo4j_graph.query(
            """
            MATCH (p:CG:Plan {cg_plan_id: $cg_plan_id})
            MERGE (g:CG:Goal {goal_id: $goal_id})
              ON CREATE SET g.text       = $goal_text,
                            g.priority   = $priority,
                            g.created_at = timestamp(),
                            g.status     = "PLANNED"
            MERGE (p)-[:HAS_GOAL]->(g)
            """,
            {
                "cg_plan_id" : cg_plan_id,
                "goal_id"    : goal_id,
                "goal_text"  : goal["text"],
                "priority"   : goal["priority"],
            },
        )
        goal_ids.append(goal_id)

    print(f"[CG] Added {len(goal_ids)} dynamic goals to CG plan {cg_plan_id}")
    return goal_ids


def cg_log_tool_call(
    neo4j_graph: Neo4jGraph,
    goal_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any,
) -> str:
    """
    Write a completed ToolCall node and attach it to the owning Goal.

    Structure written:
        Goal -[:HAS_TOOL_CALL]-> ToolCall

    Returns:
        tool_call_id
    """
    tool_call_id = f"TOOL_{uuid.uuid4().hex[:10].upper()}"
    neo4j_graph.query(
        """
        MATCH (g:CG:Goal {goal_id: $goal_id})
        MERGE (t:CG:ToolCall {tool_call_id: $tool_call_id})
          ON CREATE SET t.tool_name  = $tool_name,
                        t.input      = $tool_input,
                        t.output     = $tool_output,
                        t.created_at = timestamp(),
                        t.status     = "COMPLETED"
        MERGE (g)-[:HAS_TOOL_CALL]->(t)
        """,
        {
            "goal_id"      : goal_id,
            "tool_call_id" : tool_call_id,
            "tool_name"    : tool_name,
            "tool_input"   : str(tool_input),
            "tool_output"  : str(tool_output),
        },
    )
    return tool_call_id


def cg_log_tool_error(
    neo4j_graph: Neo4jGraph,
    goal_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    error: BaseException,
) -> None:
    """
    Write a failed ToolCall node with an attached Error node.

    Structure written:
        Goal -[:HAS_TOOL_CALL]-> ToolCall -[:HAS_ERROR]-> Error
    """
    tool_call_id = f"TOOL_{uuid.uuid4().hex[:10].upper()}"
    error_id     = f"ERR_{uuid.uuid4().hex[:8].upper()}"
    neo4j_graph.query(
        """
        MATCH (g:CG:Goal {goal_id: $goal_id})
        MERGE (t:CG:ToolCall {tool_call_id: $tool_call_id})
          ON CREATE SET t.tool_name  = $tool_name,
                        t.input      = $tool_input,
                        t.created_at = timestamp(),
                        t.status     = "FAILED"
        MERGE (g)-[:HAS_TOOL_CALL]->(t)

        MERGE (e:CG:Error {error_id: $error_id})
          ON CREATE SET e.error_type    = $error_type,
                        e.error_message = $error_message,
                        e.created_at    = timestamp()
        MERGE (t)-[:HAS_ERROR]->(e)
        """,
        {
            "goal_id"       : goal_id,
            "tool_call_id"  : tool_call_id,
            "tool_name"     : tool_name,
            "tool_input"    : str(tool_input),
            "error_id"      : error_id,
            "error_type"    : type(error).__name__,
            "error_message" : str(error),
        },
    )


def cg_write_decision(neo4j_graph: Neo4jGraph, session_id: str, result: AgentResult) -> None:
    """
    Write the final Decision node, linked to the Session.

    Structure written:
        Session -[:HAS_DECISION]-> Decision
        Session.status = "COMPLETED"
    """
    decision_id = f"DEC_{uuid.uuid4().hex[:10].upper()}"
    neo4j_graph.query(
        """
        MATCH (s:CG:Session {session_id: $session_id})
        MERGE (d:CG:Decision {decision_id: $decision_id})
          ON CREATE SET d.created_at          = timestamp(),
                        d.covered             = $covered,
                        d.prior_auth_required = $prior_auth_required,
                        d.in_network          = $in_network,
                        d.explanation         = $explanation,
                        d.missing_info        = $missing_info
        MERGE (s)-[:HAS_DECISION]->(d)
        SET s.status = "COMPLETED"
        """,
        {
            "session_id"          : session_id,
            "decision_id"         : decision_id,
            "covered"             : result.covered,
            "prior_auth_required" : result.prior_auth_required,
            "in_network"          : result.in_network,
            "explanation"         : result.explanation,
            "missing_info"        : list(result.missing_info),
        },
    )

#%%
# ---------------------------
# Visualization queries
# ---------------------------

def print_visualization_queries(session_id: str) -> None:
    print("\n=== Paste into Neo4j Browser ===\n")

    print("KG overview:")
    print(
        """
MATCH (n:KG)
WHERE n:Plan OR n:Benefit OR n:Procedure OR n:Provider OR n:Network OR n:Rule
RETURN n LIMIT 200;
        """.strip()
    )

    print("\nKG neighbourhood around CPT 29827:")
    print(
        """
MATCH (p:KG:Procedure {code: "29827"})-[r]-(n:KG)
RETURN p, r, n;
        """.strip()
    )

    print(f"\nFull CG for session {session_id} (depth 1-4):")
    print(
        f"""
MATCH (s:CG:Session {{session_id: "{session_id}"}})-[r*1..4]->(n:CG)
RETURN s, r, n;
        """.strip()
    )

    print(f"\nGoals + ToolCalls for session {session_id}:")
    print(
        f"""
MATCH (s:CG:Session {{session_id: "{session_id}"}})-[:HAS_PLAN]->(p:CG:Plan)
      -[:HAS_GOAL]->(g:CG:Goal)-[:HAS_TOOL_CALL]->(t:CG:ToolCall)
OPTIONAL MATCH (t)-[:HAS_ERROR]->(e:CG:Error)
RETURN g.text AS goal, g.priority AS priority,
       t.tool_name AS tool, t.status AS status,
       e.error_message AS error
ORDER BY priority;
        """.strip()
    )

#%%
# ---------------------------
# LangChain tools that query the KG (stable knowledge)
# ---------------------------
# These tools are "the agent's interface" to the KG.
# The tool calls themselves are recorded in the CG under the owning Goal.

def _kg_coverage_check(neo4j_graph: Neo4jGraph, plan_id: str, procedure_code: str) -> bool:
    rows = neo4j_graph.query(
        """
        MATCH (plan:KG:Plan {plan_id: $plan_id})-[:COVERS_BENEFIT]->(b:KG:Benefit)
        MATCH (p:KG:Procedure {code: $procedure_code})-[:MAPS_TO_BENEFIT]->(b)
        RETURN count(b) > 0 AS covered
        """,
        {"plan_id": plan_id, "procedure_code": procedure_code},
    )
    return bool(rows[0]["covered"]) if rows else False


def _kg_prior_auth_check(neo4j_graph: Neo4jGraph, plan_id: str, procedure_code: str) -> bool:
    rows = neo4j_graph.query(
        """
        MATCH (plan:KG:Plan {plan_id: $plan_id})-[:HAS_RULE]->(r:KG:Rule {type: "PRIOR_AUTH"})
        MATCH (r)-[:APPLIES_TO_PROCEDURE]->(p:KG:Procedure {code: $procedure_code})
        RETURN count(r) > 0 AS prior_auth_required
        """,
        {"plan_id": plan_id, "procedure_code": procedure_code},
    )
    return bool(rows[0]["prior_auth_required"]) if rows else False


def _kg_network_check(neo4j_graph: Neo4jGraph, plan_id: str, provider_id: str) -> bool:
    rows = neo4j_graph.query(
        """
        MATCH (plan:KG:Plan {plan_id: $plan_id})-[:USES_NETWORK]->(net:KG:Network)
        MATCH (prov:KG:Provider {provider_id: $provider_id})-[:IN_NETWORK]->(net)
        RETURN count(net) > 0 AS in_network
        """,
        {"plan_id": plan_id, "provider_id": provider_id},
    )
    return bool(rows[0]["in_network"]) if rows else False


def _validate_plan_id(plan_id: str) -> None:
    """plan_id must follow the PLAN_<SUFFIX> pattern, e.g. PLAN_A, PLAN_B."""
    if not plan_id.startswith("PLAN_"):
        raise ValueError(
            f"Invalid plan_id '{plan_id}': must start with 'PLAN_'. "
            f"Did you pass a provider_id or procedure_code by mistake?"
        )

def _validate_procedure_code(procedure_code: str) -> None:
    """CPT procedure codes are purely numeric, e.g. '29827', '97110'."""
    if not procedure_code.isdigit():
        raise ValueError(
            f"Invalid procedure_code '{procedure_code}': CPT codes contain digits only. "
            f"Did you pass a provider_id or plan_id by mistake?"
        )

def _validate_provider_id(provider_id: str) -> None:
    """provider_id must follow the PROV_<SUFFIX> pattern, e.g. PROV_100, PROV_200."""
    if not provider_id.startswith("PROV_"):
        raise ValueError(
            f"Invalid provider_id '{provider_id}': must start with 'PROV_'. "
            f"Did you pass a procedure_code or plan_id by mistake?"
        )


@tool
def coverage_check(plan_id: str, procedure_code: str) -> str:
    """Check whether a procedure is covered under a plan (KG lookup). Returns 'true' or 'false'.
    Args: plan_id (str): The plan identifier, e.g. 'PLAN_A'.
          procedure_code (str): The CPT procedure code (digits only), e.g. '29827'."""
    assert GRAPH is not None, "GRAPH not initialized"
    _validate_plan_id(plan_id)
    _validate_procedure_code(procedure_code)
    return str(_kg_coverage_check(GRAPH, plan_id, procedure_code)).lower()


@tool
def prior_auth_check(plan_id: str, procedure_code: str) -> str:
    """Check whether prior authorization is required for a procedure under a plan (KG lookup). Returns 'true' or 'false'.
    Args: plan_id (str): The plan identifier, e.g. 'PLAN_A'.
          procedure_code (str): The CPT procedure code (digits only), e.g. '29827'."""
    assert GRAPH is not None, "GRAPH not initialized"
    _validate_plan_id(plan_id)
    _validate_procedure_code(procedure_code)
    return str(_kg_prior_auth_check(GRAPH, plan_id, procedure_code)).lower()


@tool
def network_check(plan_id: str, provider_id: str) -> str:
    """Check whether a provider is in-network for a plan (KG lookup). Returns 'true' or 'false'.
    Args: plan_id (str): The plan identifier, e.g. 'PLAN_A'.
          provider_id (str): The provider identifier, e.g. 'PROV_200'."""
    assert GRAPH is not None, "GRAPH not initialized"
    _validate_plan_id(plan_id)
    _validate_provider_id(provider_id)
    return str(_kg_network_check(GRAPH, plan_id, provider_id)).lower()


TOOLS = [coverage_check, prior_auth_check, network_check]

#%%
# ---------------------------
# LLM Phase 1: Goal Planning
# ---------------------------

def llm_plan_goals(user_text: str, request: MemberRequest) -> List[Dict[str, Any]]:
    """
    Phase 1: Use LLM to analyze user intent and determine ordered goals.

    Returns:
        List of goal dicts with 'text' and 'priority' keys, sorted by priority.
    """
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    class Goal(BaseModel):
        text: str      = Field(description="Clear description of what needs to be determined")
        priority: int  = Field(description="Priority order (1=highest)")
        reasoning: str = Field(description="Why this goal is relevant to the user's query")

    class GoalPlan(BaseModel):
        goals: List[Goal] = Field(description="Ordered list of goals to achieve")

    parser = JsonOutputParser(pydantic_object=GoalPlan)

    prompt = PromptTemplate(
        template="""You are a health insurance customer service goal planner.

Analyze the user's query and the structured request data to determine what goals need to be accomplished.

USER QUERY: {user_text}

STRUCTURED REQUEST DATA:
- Member ID: {member_id}
- Plan ID: {plan_id}
- Provider ID: {provider_id}
- Procedure Code: {procedure_code}
- Diagnosis Code: {diagnosis_code}
- Requested Date: {requested_date}
- Location State: {location_state}

AVAILABLE GOALS (choose relevant ones based on user intent):
1. "Determine coverage eligibility" - Check if the procedure is covered under the plan
2. "Determine if prior authorization is required" - Check if prior auth is needed
3. "Determine if provider is in-network" - Check if provider is in the plan's network
4. "Verify medical necessity" - Check if diagnosis justifies the procedure
5. "Check benefit limits" - Check for any limitations or exclusions

INSTRUCTIONS:
- Identify which goals are relevant based on the user's explicit questions
- Order goals by priority (what the user cares about most = priority 1)
- Only include goals that are actually asked about or implied by the query
- Provide reasoning for each goal selection

{format_instructions}

Return ONLY valid JSON, no preamble or explanation.""",
        input_variables=[
            "user_text", "member_id", "plan_id", "provider_id",
            "procedure_code", "diagnosis_code", "requested_date", "location_state",
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[WARNING] No OPENAI_API_KEY found. Using default goals.")
        return [
            {"text": "Determine coverage eligibility",            "priority": 1},
            {"text": "Determine if prior authorization is required", "priority": 2},
            {"text": "Determine if provider is in-network",       "priority": 3},
        ]

    llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "user_text"      : user_text,
            "member_id"      : request.member_id,
            "plan_id"        : request.plan_id,
            "provider_id"    : request.provider_id,
            "procedure_code" : request.procedure_code,
            "diagnosis_code" : request.diagnosis_code,
            "requested_date" : request.requested_date,
            "location_state" : request.location_state,
        })

        print(f"\n[LLM PHASE 1] Goal Planning:")
        for goal in result["goals"]:
            print(f"  Priority {goal['priority']}: {goal['text']}")
            print(f"    Reasoning: {goal['reasoning']}")

        return [
            {"text": goal["text"], "priority": goal["priority"]}
            for goal in sorted(result["goals"], key=lambda g: g["priority"])
        ]

    except Exception as e:
        print(f"[ERROR] Goal planning failed: {e}")
        print("[FALLBACK] Using default goals")
        return [
            {"text": "Determine coverage eligibility",             "priority": 1},
            {"text": "Determine if prior authorization is required", "priority": 2},
            {"text": "Determine if provider is in-network",        "priority": 3},
        ]

#%%
# ---------------------------
# Callback: logs tool calls under the ACTIVE goal
# ---------------------------

class CGToolLoggingCallback(BaseCallbackHandler):
    """
    LangChain callback that writes ToolCall (and Error) nodes into the CG
    under the currently-active Goal node.

    Usage:
        callback = CGToolLoggingCallback(graph, session_id)
        callback.set_active_goal(goal_id)   # call before each per-goal agent invocation
        executor.invoke({"input": ...}, config={"callbacks": [callback]})
    """

    def __init__(self, graph: Neo4jGraph, session_id: str):
        super().__init__()
        self.graph          = graph
        self.session_id     = session_id
        self._active_goal_id: Optional[str] = None
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        print(f"[DEBUG] CGToolLoggingCallback initialized for session: {session_id}")

    def set_active_goal(self, goal_id: str) -> None:
        """Switch the active goal before each per-goal agent invocation."""
        self._active_goal_id = goal_id
        print(f"[DEBUG] Active goal set to: {goal_id}")

    # ------------------------------------------------------------------ #
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name") or serialized.get("id", ["unknown_tool"])
        if isinstance(tool_name, list):
            tool_name = tool_name[-1] if tool_name else "unknown_tool"

        print(f"[DEBUG] Tool started: {tool_name} | input: {input_str} | goal: {self._active_goal_id}")
        self._tool_runs[str(run_id)] = {"tool_name": tool_name, "input_str": input_str}

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._tool_runs.pop(str(run_id), None)
        if info is None:
            print(f"[DEBUG] Warning: on_tool_end called with no matching on_tool_start (run_id={run_id})")
            return

        tool_name  = info["tool_name"]
        tool_input = {"raw": info["input_str"]}
        print(f"[DEBUG] Tool ended: {tool_name} | output: {output}")

        if self._active_goal_id:
            cg_log_tool_call(self.graph, self._active_goal_id, tool_name, tool_input, output)
        else:
            print("[DEBUG] Warning: on_tool_end fired but no active goal is set — ToolCall not recorded.")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._tool_runs.pop(str(run_id), None)
        tool_name  = info["tool_name"]  if info else "unknown_tool"
        tool_input = {"raw": info["input_str"]} if info else {}

        print(f"[DEBUG] Tool error: {tool_name} | error: {error}")

        if self._active_goal_id:
            cg_log_tool_error(self.graph, self._active_goal_id, tool_name, tool_input, error)
        else:
            print("[DEBUG] Warning: on_tool_error fired but no active goal is set — Error not recorded.")

#%%
# ---------------------------
# Goal → tool mapping
# ---------------------------

# Maps every known goal text (from llm_plan_goals) to the single tool that
# serves it.  Goals that have no direct KG tool (e.g. "Check benefit limits")
# map to an empty list — the agent will reason from context alone.
#
# IMPORTANT: keys must match exactly the goal texts produced by llm_plan_goals
# (both the LLM-generated ones and the fallback defaults).

GOAL_TOOL_MAP: Dict[str, List[Any]] = {
    "Determine coverage eligibility"             : [coverage_check],
    "Determine if prior authorization is required": [prior_auth_check],
    "Determine if provider is in-network"        : [network_check],
    "Verify medical necessity"                   : [],   # no direct KG tool — LLM reasons from context
    "Check benefit limits"                       : [],   # no direct KG tool — LLM reasons from context
}

def _tools_for_goal(goal_text: str) -> List[Any]:
    """
    Return the tool subset for a given goal text.
    Falls back to all tools with a warning if the goal text is unrecognised
    (e.g. the LLM invented a novel goal phrasing).
    """
    if goal_text in GOAL_TOOL_MAP:
        return GOAL_TOOL_MAP[goal_text]
    # Unknown goal — use all tools rather than silently doing nothing,
    # and emit a warning so it is visible in logs.
    print(f"[WARNING] Unrecognised goal text '{goal_text}' — falling back to all tools.")
    return TOOLS


#%%
# ---------------------------
# LLM Phase 2: Goal Execution (Option A — per-goal agent invocation)
# ---------------------------

def llm_execute_goals_with_tools(
    graph: Neo4jGraph,
    session_id: str,
    user_text: str,
    goals: List[Dict[str, Any]],
    goal_ids: List[str],
    request: MemberRequest,
) -> str:
    """
    Phase 2 (Option A): Invoke a fresh tool-calling agent once per goal so that:
      - Every ToolCall node is unambiguously attached to the goal that triggered it.
      - Each agent invocation only has access to the tool(s) relevant to that goal,
        making it structurally impossible to fire irrelevant tools.
      - Exact argument values (plan_id, procedure_code, provider_id) are injected
        directly into the system prompt so the LLM never has to parse them from
        free text — eliminating wrong-argument hallucinations entirely.

    Args:
        graph:      Neo4j graph instance
        session_id: Session ID (used for the callback)
        user_text:  Original user query (shown to the agent as context)
        goals:      Ordered list of goal dicts (text + priority) — sorted by priority
        goal_ids:   Parallel list of CG goal_ids in the same priority order
        request:    Original MemberRequest — provides the exact tool argument values

    Returns:
        Aggregated natural language response addressing all goals.
    """
    assert len(goals) == len(goal_ids), "goals and goal_ids must be parallel lists of the same length"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Single callback instance — we update the active goal before each invocation.
    callback = CGToolLoggingCallback(graph, session_id)

    goal_responses: List[str] = []

    for goal, goal_id in zip(goals, goal_ids):
        goal_text     = goal["text"]
        goal_priority = goal["priority"]
        goal_tools    = _tools_for_goal(goal_text)

        print(f"\n[LLM PHASE 2] Goal (priority {goal_priority}): '{goal_text}'")
        print(f"              Restricted tools: {[t.name for t in goal_tools] or '(none — reasoning only)'}")

        # Point the callback at this goal BEFORE the agent runs.
        callback.set_active_goal(goal_id)

        if goal_tools:
            # Build a fresh agent restricted to only this goal's tools.
            # The system prompt:
            #   (a) names only the single allowed tool — no other tool can be called
            #   (b) supplies the exact argument values from the structured request,
            #       so the LLM never has to parse them from natural language text.
            tool_names_str = "\n".join(
                f"- {t.name}: {t.description}" for t in goal_tools
            )
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are a health insurance customer service agent.

YOUR CURRENT GOAL: {{goal_text}}

EXACT ARGUMENT VALUES — use these verbatim when calling the tool. Do not substitute other values:
  plan_id        = {request.plan_id}
  procedure_code = {request.procedure_code}
  provider_id    = {request.provider_id}

INSTRUCTIONS:
1. Call the tool below EXACTLY ONCE using the argument values provided above.
2. Do not call any other tool.
3. Use the tool result to form a concise, factual answer.

Available tool (one only):
{tool_names_str}""",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent    = create_tool_calling_agent(llm, goal_tools, prompt)
            executor = AgentExecutor(
                agent          = agent,
                tools          = goal_tools,
                verbose        = True,
                max_iterations = 2,   # one clean call + one retry at most
            )

            try:
                response = executor.invoke(
                    {"input": user_text, "goal_text": goal_text},
                    config={"callbacks": [callback]},
                )
                goal_responses.append(
                    f"[Goal {goal_priority}] {goal_text}:\n{response['output']}"
                )

            except Exception as e:
                print(f"[ERROR] Goal execution failed for goal_id={goal_id}: {e}")
                goal_responses.append(
                    f"[Goal {goal_priority}] {goal_text}:\nExecution failed: {e}"
                )

        else:
            # No KG tool for this goal — ask the LLM to reason from the
            # structured data already in the user message (no tool calls,
            # so nothing is written to the CG for this goal).
            print(f"              No tool available — LLM will reason from context.")
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are a health insurance customer service agent.

YOUR CURRENT GOAL: {{goal_text}}

STRUCTURED REQUEST DATA (for reference):
  plan_id        = {request.plan_id}
  procedure_code = {request.procedure_code}
  provider_id    = {request.provider_id}
  diagnosis_code = {request.diagnosis_code}

No tools are available for this goal. Reason directly from the information
provided and give a concise, factual answer.""",
                ),
                ("human", "{input}"),
            ])
            chain = prompt | llm
            try:
                response = chain.invoke(
                    {"input": user_text, "goal_text": goal_text},
                    config={"callbacks": [callback]},
                )
                goal_responses.append(
                    f"[Goal {goal_priority}] {goal_text}:\n{response.content}"
                )
            except Exception as e:
                print(f"[ERROR] Goal reasoning failed for goal_id={goal_id}: {e}")
                goal_responses.append(
                    f"[Goal {goal_priority}] {goal_text}:\nReasoning failed: {e}"
                )

    return "\n\n".join(goal_responses)

#%%
# ---------------------------
# Main execution
# ---------------------------

reset_demo_data( GRAPH )

#%%
seed_knowledge_graph( GRAPH )

#%%
request = MemberRequest(
    member_id      = "MEM_001",
    plan_id        = "PLAN_A",
    provider_id    = "PROV_200",
    procedure_code = "29827",
    diagnosis_code = "M75.1",
    requested_date = "2026-03-02",
    location_state = "IL",
)

user_text = (
    f"I'm scheduled for CPT {request.procedure_code} with provider {request.provider_id}. "
    f"My plan is {request.plan_id}. Is it covered, do I need prior auth, and is the provider in-network?"
)

#%%
print("=" * 80)
print("PHASE 0: Creating Context Graph (CG) session...")
print("=" * 80)

session_id, cg_plan_id = create_context_session(GRAPH, request, user_text)
print(f"Session created : {session_id}")
print(f"CG Plan created : {cg_plan_id}\n")

#%%
print("=" * 80)
print("PHASE 1: LLM Goal Planning (analysing user intent)...")
print("=" * 80)

planned_goals = llm_plan_goals(user_text, request)

# Goals are now attached to the CG Plan node, not the Session.
goal_ids = cg_add_dynamic_goals(GRAPH, cg_plan_id, planned_goals)
print()

#%%
print("=" * 80)
print("PHASE 2: LLM Goal Execution (per-goal agent invocations)...")
print("=" * 80)

output = llm_execute_goals_with_tools(GRAPH, session_id, user_text, planned_goals, goal_ids, request)

#%%
print("\n" + "=" * 80)
print("FINAL RESPONSE")
print("=" * 80)
print(output)
print()

#%%
print_visualization_queries(session_id)