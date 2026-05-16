"""
Shared fixtures for CSIP AI evaluation tests.

Creates deterministic test records in Neo4j KG per scenario category
and tears them all down after the test session completes.

Each scenario gets its own member, policy, provider, claim, and PA
so queries don't interfere with each other across tests.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timedelta

import httpx
import pytest
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("CSIP_API_BASE", "https://localhost/agentic/access")
VERIFY_SSL = os.getenv("CSIP_VERIFY_SSL", "false").lower() == "true"
API_USER = os.getenv("CSIP_EVAL_USER", "rpatel")
API_PASS = os.getenv("CSIP_EVAL_PASS", "testuser")
QUERY_TIMEOUT = float(os.getenv("CSIP_QUERY_TIMEOUT", "120"))
DEFAULT_TIMEOUT = float(os.getenv("CSIP_DEFAULT_TIMEOUT", "15"))

NEO4J_KG_URI = os.getenv("NEO4J_KG_URI", "bolt://localhost:7687")
NEO4J_KG_USER = os.getenv("NEO4J_KG_USER", "neo4j")
NEO4J_KG_PASSWORD = os.getenv("NEO4J_KG_PASSWORD", "neo4j_kg_admin")


# ---------------------------------------------------------------------------
# Auth and API client
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def auth_token() -> str:
    response = httpx.post(
        f"{API_BASE}/api/auth/login",
        json={"username": API_USER, "password": API_PASS},
        verify=VERIFY_SSL,
        timeout=DEFAULT_TIMEOUT,
    )
    assert response.status_code == 200, f"Auth failed: {response.text}"
    return response.json()["access_token"]


@pytest.fixture(scope="session")
def api_client(auth_token) -> httpx.Client:
    client = httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {auth_token}"},
        verify=VERIFY_SSL,
        timeout=QUERY_TIMEOUT,
    )
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Record factory — creates a complete member+policy+provider+claim+PA set
# ---------------------------------------------------------------------------
def _create_scenario_records(session, scenario_tag: str, suffix: int,
                             claim_status: str = "APPROVED",
                             member_status: str = "ACTIVE",
                             pa_status: str = "APPROVED"):
    """
    Create a complete set of linked domain records in Neo4j KG.
    Returns a dict with all IDs and known field values.
    All nodes carry testRecord=true and scenarioTag for identification.
    """
    member_id = str(uuid.uuid4())
    policy_id = str(uuid.uuid4())
    provider_id = str(uuid.uuid4())
    claim_id = str(uuid.uuid4())
    claim_number = f"CLM-99{suffix:04d}"
    pa_id = str(uuid.uuid4())
    pa_number = f"PA-99{suffix:04d}"

    now = datetime.now()
    service_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    effective_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    expiration_date = (now + timedelta(days=365)).strftime("%Y-%m-%d")

    paid = 2800.00 if claim_status == "APPROVED" else 0.00
    denial = "Not medically necessary" if claim_status == "DENIED" else None

    # Member
    session.run("""
        CREATE (m:Member {
            memberId: $memberId, firstName: 'Eval', lastName: $tag,
            dateOfBirth: '1985-06-15', email: $email, phone: '555-9900',
            street: '100 Test Ave', city: 'Chicago', state: 'IL',
            zipCode: '60601', enrollmentDate: $effectiveDate,
            status: $status, testRecord: true, scenarioTag: $tag
        })
    """, memberId=member_id, tag=scenario_tag,
         email=f"eval.{scenario_tag}@example.com",
         effectiveDate=effective_date, status=member_status)

    # Policy
    session.run("""
        CREATE (p:Policy {
            policyId: $policyId, policyNumber: $polNum,
            policyType: 'INDIVIDUAL', planName: 'PPO Plan Gold',
            planType: 'PPO', effectiveDate: $effectiveDate,
            expirationDate: $expirationDate, status: 'ACTIVE',
            premium: 450.00, deductible: 1000, outOfPocketMax: 5000,
            testRecord: true, scenarioTag: $tag
        })
    """, policyId=policy_id, polNum=f"POL-99{suffix:04d}",
         effectiveDate=effective_date, expirationDate=expiration_date,
         tag=scenario_tag)

    # Member -> Policy
    session.run("""
        MATCH (m:Member {memberId: $mid})
        MATCH (p:Policy {policyId: $pid})
        CREATE (m)-[:HAS_POLICY]->(p)
    """, mid=member_id, pid=policy_id)

    # Provider
    session.run("""
        CREATE (pr:Provider {
            providerId: $providerId, npi: $npi,
            providerType: 'INDIVIDUAL', firstName: 'Eval',
            lastName: $tag, specialty: 'Orthopedics',
            phone: '555-9901', street: '200 Medical Plaza',
            city: 'Chicago', state: 'IL', zipCode: '60602',
            testRecord: true, scenarioTag: $tag
        })
    """, providerId=provider_id, npi=f"999000{suffix:04d}",
         tag=scenario_tag)

    # Claim
    session.run("""
        CREATE (c:Claim {
            claimId: $claimId, claimNumber: $claimNumber,
            serviceDate: $serviceDate, submissionDate: $serviceDate,
            status: $status, totalAmount: 3500.00, paidAmount: $paid,
            denialReason: $denial, processingDate: $serviceDate,
            testRecord: true, scenarioTag: $tag
        })
    """, claimId=claim_id, claimNumber=claim_number,
         serviceDate=service_date, status=claim_status,
         paid=paid, denial=denial, tag=scenario_tag)

    # Claim relationships
    session.run("""
        MATCH (m:Member {memberId: $mid})
        MATCH (c:Claim {claimId: $cid})
        CREATE (m)-[:FILED_CLAIM]->(c)
    """, mid=member_id, cid=claim_id)

    session.run("""
        MATCH (p:Policy {policyId: $pid})
        MATCH (c:Claim {claimId: $cid})
        CREATE (c)-[:UNDER_POLICY]->(p)
    """, pid=policy_id, cid=claim_id)

    session.run("""
        MATCH (pr:Provider {providerId: $prid})
        MATCH (c:Claim {claimId: $cid})
        CREATE (c)-[:SERVICED_BY]->(pr)
    """, prid=provider_id, cid=claim_id)

    # Prior Authorization
    session.run("""
        CREATE (pa:PriorAuthorization {
            paId: $paId, paNumber: $paNumber,
            procedureCode: '29881',
            procedureDescription: 'Knee arthroscopy',
            requestDate: $serviceDate, status: $paStatus,
            urgency: 'ROUTINE', approvalDate: $serviceDate,
            expirationDate: $expirationDate, denialReason: null,
            testRecord: true, scenarioTag: $tag
        })
    """, paId=pa_id, paNumber=pa_number, serviceDate=service_date,
         expirationDate=expiration_date, paStatus=pa_status,
         tag=scenario_tag)

    session.run("""
        MATCH (m:Member {memberId: $mid})
        MATCH (pa:PriorAuthorization {paId: $paid})
        CREATE (m)-[:REQUESTED_PA]->(pa)
    """, mid=member_id, paid=pa_id)

    session.run("""
        MATCH (pr:Provider {providerId: $prid})
        MATCH (pa:PriorAuthorization {paId: $paid})
        CREATE (pa)-[:REQUESTED_BY]->(pr)
    """, prid=provider_id, paid=pa_id)

    return {
        "member_id": member_id,
        "member_status": member_status,
        "policy_id": policy_id,
        "policy_number": f"POL-99{suffix:04d}",
        "provider_id": provider_id,
        "claim_id": claim_id,
        "claim_number": claim_number,
        "claim_status": claim_status,
        "claim_amount": 3500.00,
        "claim_paid": paid,
        "pa_id": pa_id,
        "pa_number": pa_number,
        "pa_status": pa_status,
    }


# ---------------------------------------------------------------------------
# Session-scoped test data — one record set per scenario
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_data():
    """
    Create isolated record sets for each test scenario.

    Yields a dict keyed by scenario name, each containing a full
    set of IDs and known field values.

    Teardown deletes all nodes with testRecord=true.
    """
    driver = GraphDatabase.driver(
        NEO4J_KG_URI, auth=(NEO4J_KG_USER, NEO4J_KG_PASSWORD)
    )

    scenarios = {}
    with driver.session() as session:
        # Worker faithfulness: claim_lookup + eligibility
        scenarios["worker"] = _create_scenario_records(
            session, "worker", 1001)

        # Consolidation: multi-team (member + claims)
        scenarios["consolidation"] = _create_scenario_records(
            session, "consolidation", 1002)

        # Decision agent: recommendation accuracy
        scenarios["decision_recommend"] = _create_scenario_records(
            session, "decision_recommend", 1003)

        # Decision agent: CG node properties
        scenarios["decision_cg_node"] = _create_scenario_records(
            session, "decision_cg_node", 1004)

        # Decision agent: evidence completeness
        scenarios["decision_evidence"] = _create_scenario_records(
            session, "decision_evidence", 1005)

        # Feedback: single-team routing
        scenarios["feedback_single"] = _create_scenario_records(
            session, "feedback_single", 1006)

        # Feedback: multi-team routing
        scenarios["feedback_multi"] = _create_scenario_records(
            session, "feedback_multi", 1007)

        # Feedback: decision routing
        scenarios["feedback_decision"] = _create_scenario_records(
            session, "feedback_decision", 1008)

        # Feedback: consistency (3 runs of same query)
        scenarios["feedback_consistency"] = _create_scenario_records(
            session, "feedback_consistency", 1009)

        # N-run protocol: dedicated records for variance analysis
        scenarios["nrun"] = _create_scenario_records(
            session, "nrun", 1010)

        # Multi-turn: records for follow-up chain test
        scenarios["multi_turn"] = _create_scenario_records(
            session, "multi_turn", 1011)

    yield scenarios

    # Teardown — delete all test nodes
    with driver.session() as session:
        session.run("MATCH (n {testRecord: true}) DETACH DELETE n")
    driver.close()
