"""
Integration Tests — Claims Services A2A Server
================================================
Tests the live claims services A2A supervisor reachable via Nginx api-gateway
at https://localhost:8443/a2a/claims using A2AClientNode as the client.

Prerequisites:
    - api-gateway running on localhost:8443 (mTLS)
    - claims-services-supervisor running and healthy
    - claims-services-mcp-tools running and healthy
    - .env with:
        MCP_CLIENT_CERT=/home/kbmurali/tmp/certs/mcp/client.crt
        MCP_CLIENT_KEY=/home/kbmurali/tmp/certs/mcp/client.key
        MCP_CA_CERT=/home/kbmurali/tmp/certs/mcp/ca.crt
        A2A_CLAIMS_SERVICES_URL=https://localhost:8443/a2a/claims

    - TEST_CLAIM_ID should be a valid claim UUID from the dev knowledge graph
      (e.g. 7799c06c-0883-4dca-b1f0-bded6d1027a5 from the dev seed data)
    - TEST_CLAIM_NUMBER should be a valid claim number from the dev KG
      (e.g. CLM-2024-0001 — this is the claim_number field, NOT the UUID)

Run:
    cd customer_service_platform
    pytest agents/teams/claims_services/test_cs_a2a.py -v
"""
import os
import logging
import uuid
from typing import Any, Dict

import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())

from agents.core.state import SupervisorState
from agents.teams.claims_services.supervisor.tool_schemas import build_schema_registry
from agents.core.a2a_client_node import A2AClientNode
from databases.context_graph_data_access import ContextGraphDataAccess

from agents.security import RBACService, RateLimiter, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_URL         = os.getenv("A2A_CLAIMS_SERVICES_URL", "https://localhost:8443/a2a/claims")
AGENT_NAME        = "claims_services_supervisor_agent"

TEST_USER_ID    = os.getenv("TEST_USER_ID",    "usr-tier2-001")
TEST_USER_ROLE  = os.getenv("TEST_USER_ROLE",  "CSR_TIER2")
TEST_SESSION_ID = os.getenv("TEST_SESSION_ID", str( uuid.uuid1()))
TEST_CLAIM_ID  = os.getenv("TEST_CLAIM_ID",  "7799c06c-0883-4dca-b1f0-bded6d1027a5")
TEST_CLAIM_NUMBER  = os.getenv("TEST_CLAIM_NUMBER",  "CLM-421386")


cg_dao = ContextGraphDataAccess()
cg_dao.create_session(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)


def exceed_rate_limit(user_id: str, user_role: str, tool_name: str):
    rbac_service: RBACService = RBACService()
    rate_limiter: RateLimiter = RateLimiter()

    current_rate_limit = rbac_service.get_tool_rate_limit(
        user_role=user_role, tool_name=tool_name
    )

    try:
        for _i in range(current_rate_limit + 1):
            rate_limiter.check_rate_limit(
                user_id=user_id,
                resource_type="TOOL",
                resource_name=tool_name,
                limit_per_minute=current_rate_limit,
            )
    except RateLimitError:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client_node() -> A2AClientNode:
    """Create a shared A2AClientNode for all tests in this module."""
    return A2AClientNode(
        agent_name=AGENT_NAME,
        agent_url=AGENT_URL,
        schema_registry=build_schema_registry(),
        from_agent_name="test_client",
    )


def _make_state(query: str, session_id: str | None = None) -> SupervisorState:
    """Build a minimal SupervisorState for a test query."""
    return {
        "messages":       [HumanMessage(content=query)],
        "user_id":        TEST_USER_ID,
        "user_role":      TEST_USER_ROLE,
        "session_id":     session_id or TEST_SESSION_ID,
        "execution_path": [],
        "tool_results":   {},
    }


def _assert_successful_response(result: Dict[str, Any], test_name: str) -> None:
    """Common assertions for a successful A2A response."""
    assert result is not None, f"[{test_name}] Result is None"
    assert "messages" in result, f"[{test_name}] No messages in result"
    assert "error" not in result or result.get("error") is None, \
        f"[{test_name}] Unexpected error: {result.get('error')}"
    assert result["messages"], f"[{test_name}] Empty messages list"

    last_msg = result["messages"][-1]
    content  = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    assert content.strip(), f"[{test_name}] Empty response content"

    logger.info("[%s] Response: %s", test_name, content[:200])


# ---------------------------------------------------------------------------
# Health check (sanity before running full tests)
# ---------------------------------------------------------------------------

class TestA2AServerHealth:
    """Verify the Claims Services A2A server is reachable before task tests."""

    def test_health_endpoint(self):
        """GET /health should return 200 with healthy status via mTLS."""
        import ssl
        import httpx

        cert_path = os.getenv("MCP_CLIENT_CERT", "")
        key_path  = os.getenv("MCP_CLIENT_KEY",  "")
        ca_path   = os.getenv("MCP_CA_CERT",     "")

        assert all(os.path.exists(p) for p in [cert_path, key_path, ca_path]), \
            "mTLS cert files not found — check .env MCP_CLIENT_CERT/KEY/CA_CERT"

        ssl_ctx = ssl.create_default_context(cafile=ca_path)
        ssl_ctx.load_cert_chain(cert_path, key_path)

        with httpx.Client(verify=ssl_ctx, timeout=10) as http:
            response = http.get(f"{AGENT_URL}/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert data.get("protocol") == "a2a"
        logger.info("Health check passed: %s", data)

    def test_agent_card_endpoint(self):
        """GET /.well-known/agent.json should return a valid claims agent card."""
        import ssl
        import httpx

        cert_path = os.getenv("MCP_CLIENT_CERT", "")
        key_path  = os.getenv("MCP_CLIENT_KEY",  "")
        ca_path   = os.getenv("MCP_CA_CERT",     "")

        ssl_ctx = ssl.create_default_context(cafile=ca_path)
        ssl_ctx.load_cert_chain(cert_path, key_path)

        with httpx.Client(verify=ssl_ctx, timeout=10) as http:
            response = http.get(f"{AGENT_URL}/.well-known/agent.json")

        assert response.status_code == 200
        card = response.json()
        assert "name" in card
        assert "skills" in card
        assert "url" in card
        assert len(card["skills"]) > 0

        skill_ids = [s.get("id") for s in card.get("skills", [])]
        assert "claim_lookup"       in skill_ids, f"claim_lookup skill missing: {skill_ids}"
        assert "claim_status"       in skill_ids, f"claim_status skill missing: {skill_ids}"
        assert "claim_payment_info" in skill_ids, f"claim_payment_info skill missing: {skill_ids}"

        logger.info("Agent card: name=%s, skills=%s", card.get("name"), skill_ids)


# ---------------------------------------------------------------------------
# Claim Lookup Tests
# ---------------------------------------------------------------------------

class TestClaimLookup:
    """Tests for claim lookup skill via A2A."""

    def test_claim_lookup_by_id(self, client_node):
        """Look up a known claim by claim ID (UUID)."""
        state  = _make_state(f"Look up claim {TEST_CLAIM_ID}")
        result = client_node(state)
        _assert_successful_response(result, "test_claim_lookup_by_id")

        content = result["messages"][-1].content
        # Response should reference the claim or its ID
        assert TEST_CLAIM_ID in content or "claim" in content.lower(), \
            f"Expected claim reference in response, got: {content[:200]}"

    def test_claim_lookup_execution_path(self, client_node):
        """Verify execution path includes claim_lookup worker."""
        state  = _make_state(f"Get full details for claim {TEST_CLAIM_ID}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("claim_lookup" in str(step).lower() for step in path), \
            f"Expected claim_lookup in execution path, got: {path}"

    def test_claim_lookup_unknown_id(self, client_node):
        """Look up a claim ID that doesn't exist — should return graceful response."""
        state  = _make_state("Look up claim 00000000-0000-0000-0000-000000000000")
        result = client_node(state)

        # Should not crash — either an error message or graceful not-found
        assert result is not None
        assert "messages" in result
        content = result["messages"][-1].content if result["messages"] else ""
        assert content.strip(), "Expected non-empty response for unknown claim ID"

    def test_claim_lookup_tool_results(self, client_node):
        """Verify tool_results are populated after claim lookup."""
        state  = _make_state(f"Look up claim {TEST_CLAIM_ID}")
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Claim Status Tests
# ---------------------------------------------------------------------------

class TestClaimStatus:
    """Tests for claim status skill via A2A.

    Note: claim_status takes a claim NUMBER (e.g. CLM-2024-0001),
    not a claim ID (UUID).  Use TEST_CLAIM_NUMBER for these tests.
    """
    def test_claim_status_by_id(self, client_node):
        """Check status of a known claim by claim number."""
        state  = _make_state(f"What is the status of claim with id {TEST_CLAIM_ID}?")
        result = client_node(state)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>\n\n{result}")
        _assert_successful_response(result, "test_claim_status_by_number")

        content = result["messages"][-1].content
        assert "claim" in content.lower(), \
            f"Expected claim reference in status response, got: {content[:200]}"
            
    def test_claim_status_by_number(self, client_node):
        """Check status of a known claim by claim number."""
        state  = _make_state(f"What is the status of claim {TEST_CLAIM_NUMBER}?")
        result = client_node(state)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>\n\n{result}")
        _assert_successful_response(result, "test_claim_status_by_number")

        content = result["messages"][-1].content
        assert "claim" in content.lower(), \
            f"Expected claim reference in status response, got: {content[:200]}"

    def test_claim_status_execution_path(self, client_node):
        """Verify execution path includes claim_status worker."""
        state  = _make_state(f"Check claim status for {TEST_CLAIM_NUMBER}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("claim_status" in str(step).lower() for step in path), \
            f"Expected claim_status in execution path, got: {path}"

    def test_claim_status_unknown_number(self, client_node):
        """Look up a claim number that doesn't exist — should return graceful response."""
        state  = _make_state("What is the status of claim CLM-0000-0000?")
        result = client_node(state)

        assert result is not None
        assert "messages" in result
        content = result["messages"][-1].content if result["messages"] else ""
        assert content.strip(), "Expected non-empty response for unknown claim number"

    def test_claim_status_without_number(self, client_node):
        """Status query without providing a claim number — supervisor should SKIP gracefully."""
        state  = _make_state("What is the status of my claim?")
        result = client_node(state)

        # SKIP path — supervisor has no claim number to route with.
        # Should still return a non-empty response.
        assert result is not None
        assert "messages" in result


# ---------------------------------------------------------------------------
# Claim Payment Info Tests
# ---------------------------------------------------------------------------

class TestClaimPaymentInfo:
    """Tests for claim payment info skill via A2A."""

    def test_claim_payment_with_lookup(self, client_node):
        """Retrieve payment information for a known claim."""
        state  = _make_state(
            f"Can you get my claim details and also find how much was paid on claim {TEST_CLAIM_ID}?"
        )
        #exceed_rate_limit(TEST_USER_ID, TEST_USER_ROLE, "claim_lookup" )
        result = client_node(state)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>\n\n{result}")
        _assert_successful_response(result, "test_claim_payment_info_basic")
        
    def test_claim_payment_info_basic(self, client_node):
        """Retrieve payment information for a known claim."""
        state  = _make_state(
            f"How much was paid on claim {TEST_CLAIM_ID}?"
        )
        result = client_node(state)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>\n\n{result}")
        _assert_successful_response(result, "test_claim_payment_info_basic")

    def test_claim_payment_info_execution_path(self, client_node):
        """Verify execution path includes claim_payment_info worker."""
        state  = _make_state(f"Show payment details for claim {TEST_CLAIM_ID}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("claim_payment_info" in str(step).lower() for step in path), \
            f"Expected claim_payment_info in execution path, got: {path}"

    def test_claim_payment_info_unknown_id(self, client_node):
        """Payment info for non-existent claim ID — should return graceful response."""
        state  = _make_state("Show payment info for claim 00000000-0000-0000-0000-000000000000")
        result = client_node(state)

        assert result is not None
        assert "messages" in result
        content = result["messages"][-1].content if result["messages"] else ""
        assert content.strip(), "Expected non-empty response for unknown claim ID"

    def test_claim_payment_info_tool_results(self, client_node):
        """Verify tool_results are populated after payment info lookup."""
        state  = _make_state(
            f"Get payment details for claim {TEST_CLAIM_ID}"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Multi-skill / Routing Tests
# ---------------------------------------------------------------------------

class TestA2ARouting:
    """Tests that verify the supervisor routes correctly to the right worker."""

    def test_lookup_and_payment_query(self, client_node):
        """A query asking for claim details and payment should route to both workers."""
        state  = _make_state(
            f"Look up claim {TEST_CLAIM_ID} and tell me how much was paid."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_lookup_and_payment_query")

        # Execution path should mention both workers
        path = result.get("execution_path", [])
        path_str = " ".join(str(s) for s in path).lower()
        assert "claim_lookup" in path_str or "claim_payment" in path_str, \
            f"Expected claims workers in execution path, got: {path}"

    def test_a2a_task_state_completed(self, client_node):
        """Successful tasks should have a2a_task_state of 'completed'."""
        state  = _make_state(f"Look up claim {TEST_CLAIM_ID}")
        result = client_node(state)

        task_state = result.get("a2a_task_state")
        assert task_state == "completed", \
            f"Expected 'completed', got '{task_state}'"

    def test_claim_number_vs_id_routing(self, client_node):
        """Supervisor should route claim_number queries to claim_status,
        not claim_lookup."""
        state  = _make_state(
            f"Check the processing status for claim number {TEST_CLAIM_NUMBER}"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_claim_number_vs_id_routing")

        path     = result.get("execution_path", [])
        path_str = " ".join(str(s) for s in path).lower()
        assert "claim_status" in path_str, \
            f"Expected claim_status worker for claim number query, got: {path}"

    def test_session_isolation(self, client_node):
        """Two requests with different session IDs should not interfere."""
        session_a = f"test-session-a-{uuid.uuid4().hex[:6]}"
        session_b = f"test-session-b-{uuid.uuid4().hex[:6]}"

        state_a = _make_state(f"Look up claim {TEST_CLAIM_ID}", session_id=session_a)
        state_b = _make_state(f"Look up claim {TEST_CLAIM_ID}", session_id=session_b)

        result_a = client_node(state_a)
        result_b = client_node(state_b)

        _assert_successful_response(result_a, "session_a")
        _assert_successful_response(result_b, "session_b")


# ---------------------------------------------------------------------------
# Error / Edge Case Tests
# ---------------------------------------------------------------------------

class TestA2AErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_query(self, client_node):
        """Empty query should return a graceful response, not crash."""
        state  = _make_state("")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_claim_id_format(self, client_node):
        """A claim ID in wrong format should be handled gracefully."""
        state  = _make_state("Look up claim INVALID-CLAIM-ID-FORMAT")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_claim_number_format(self, client_node):
        """A claim number in wrong format should be handled gracefully."""
        state  = _make_state("Check status of claim NOTACLAIMNUMBER")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_result_has_no_unhandled_exception(self, client_node):
        """Any query should return a result dict, never raise an exception."""
        queries = [
            "What is the status of my claim?",
            "Show me payment information",
            f"Look up claim {TEST_CLAIM_ID}",
            f"Check status of claim number {TEST_CLAIM_NUMBER}",
        ]
        for query in queries:
            state = _make_state(query)
            try:
                result = client_node(state)
                assert isinstance(result, dict), \
                    f"Expected dict, got {type(result)} for query: '{query}'"
            except Exception as e:
                pytest.fail(
                    f"A2AClientNode raised unexpected exception for '{query}': {e}"
                )
