"""
Integration Tests — Member Services A2A Server
================================================
Tests the live member services A2A supervisor reachable via Nginx api-gateway
at https://localhost:8443/a2a/member using A2AClientNode as the client.

Prerequisites:
    - api-gateway running on localhost:8443 (mTLS)
    - member-services-supervisor running and healthy
    - member-services-mcp-tools running and healthy
    - .env with:
        MCP_CLIENT_CERT=/home/kbmurali/tmp/certs/mcp/client.crt
        MCP_CLIENT_KEY=/home/kbmurali/tmp/certs/mcp/client.key
        MCP_CA_CERT=/home/kbmurali/tmp/certs/mcp/ca.crt
        A2A_MEMBER_SERVICES_URL=https://localhost:8443/a2a/member

Run:
    cd customer_service_platform
    pytest agents/teams/member_services/test_member_services_a2a.py -v
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
from agents.teams.member_services.supervisor.tool_schemas import build_schema_registry
from agents.core.a2a_client_node import A2AClientNode
from databases.context_graph_data_access import ContextGraphDataAccess

from agents.security import RBACService, RateLimiter, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_URL       = "https://localhost:8443/a2a/member"
AGENT_NAME      = "member_services_supervisor_agent"
TEST_USER_ID    = os.getenv("TEST_USER_ID",    "usr-tier2-001")
TEST_USER_ROLE  = os.getenv("TEST_USER_ROLE",  "CSR_TIER2")
TEST_SESSION_ID = os.getenv("TEST_SESSION_ID", str( uuid.uuid1()))
TEST_MEMBER_ID  = os.getenv("TEST_MEMBER_ID",  "27b71fd8-49b7-46dd-84e3-5ad05d0a5db7")

cg_dao = ContextGraphDataAccess()
    
cg_dao.create_session( session_id=TEST_SESSION_ID, user_id=TEST_USER_ID )

def exceed_rate_limit( user_id: str, user_role: str, tool_name: str ):
    rbac_service: RBACService = RBACService()
    rate_limiter: RateLimiter = RateLimiter()
    
    current_rate_limit = rbac_service.get_tool_rate_limit( user_role=user_role, tool_name=tool_name )
    
    try:
        for _i in range( current_rate_limit+ 1 ):
            rate_limiter.check_rate_limit( user_id=user_id, resource_type="TOOL", resource_name=tool_name, limit_per_minute=current_rate_limit )
    except RateLimitError as e:
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
        "messages": [HumanMessage(content=query)],
        "user_id": TEST_USER_ID,
        "user_role": TEST_USER_ROLE,
        "session_id": session_id or TEST_SESSION_ID,
        "execution_path": [],
        "tool_results": {},
    }


def _assert_successful_response(result: Dict[str, Any], test_name: str) -> None:
    """Common assertions for a successful A2A response."""
    assert result is not None, f"[{test_name}] Result is None"
    assert "messages" in result, f"[{test_name}] No messages in result"
    assert "error" not in result or result.get("error") is None, \
        f"[{test_name}] Unexpected error: {result.get('error')}"
    assert result["messages"], f"[{test_name}] Empty messages list"

    last_msg = result["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    assert content.strip(), f"[{test_name}] Empty response content"

    logger.info("[%s] Response: %s", test_name, content[:200])


# ---------------------------------------------------------------------------
# Health check (sanity before running full tests)
# ---------------------------------------------------------------------------

class TestA2AServerHealth:
    """Verify the A2A server is reachable before running task tests."""

    def test_health_endpoint(self):
        """GET /health should return 200 with healthy status via mTLS."""
        import ssl
        import os
        import httpx

        cert_path = os.getenv("MCP_CLIENT_CERT", "")
        key_path  = os.getenv("MCP_CLIENT_KEY", "")
        ca_path   = os.getenv("MCP_CA_CERT", "")

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
        """GET /.well-known/agent.json should return a valid agent card."""
        import ssl
        import os
        import httpx

        cert_path = os.getenv("MCP_CLIENT_CERT", "")
        key_path  = os.getenv("MCP_CLIENT_KEY", "")
        ca_path   = os.getenv("MCP_CA_CERT", "")

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
        logger.info("Agent card: name=%s, skills=%s",
                    card.get("name"),
                    [s.get("id") for s in card.get("skills", [])])


# ---------------------------------------------------------------------------
# Member Lookup Tests
# ---------------------------------------------------------------------------

class TestMemberLookup:
    """Tests for member lookup skill via A2A."""

    def test_member_lookup_by_id(self, client_node):
        """Look up a known member by ID."""
        #exceed_rate_limit( TEST_USER_ID, TEST_USER_ROLE, "member_lookup" )
        state = _make_state(f"Look up member {TEST_MEMBER_ID}")
        result = client_node(state)
        _assert_successful_response(result, "test_member_lookup_by_id")

        content = result["messages"][-1].content
        # Response should reference the member ID
        assert TEST_MEMBER_ID in content or "member" in content.lower()

    def test_member_lookup_execution_path(self, client_node):
        """Verify execution path includes member_lookup worker."""
        state = _make_state(f"Find information for member {TEST_MEMBER_ID}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("member" in str(step).lower() for step in path), \
            f"Expected member_lookup in execution path, got: {path}"

    def test_member_lookup_unknown_member(self, client_node):
        """Look up a member ID that doesn't exist — should return graceful response."""
        state = _make_state("Look up member M9999999")
        result = client_node(state)

        # Should not crash — either an error message or graceful not-found
        assert result is not None
        assert "messages" in result
        content = result["messages"][-1].content if result["messages"] else ""
        assert content.strip(), "Expected non-empty response for unknown member"


# ---------------------------------------------------------------------------
# Eligibility Check Tests
# ---------------------------------------------------------------------------

class TestEligibilityCheck:
    """Tests for eligibility check skill via A2A."""

    def test_eligibility_check_basic(self, client_node):
        """Check eligibility for a known member."""
        state = _make_state(
            f"Check eligibility for member {TEST_MEMBER_ID} for service date 2026-02-22"
        )
        #exceed_rate_limit( TEST_USER_ID, TEST_USER_ROLE, "member_lookup" )
        result = client_node(state)
        print( f">>>>>>>>>>>>>>>>>>>>>>>>\n\n{result}" )
        _assert_successful_response(result, "test_eligibility_check_basic")

    def test_eligibility_check_without_date(self, client_node):
        """Check eligibility without specifying a service date."""
        state = _make_state(f"Is member {TEST_MEMBER_ID} currently eligible?")
        result = client_node(state)
        print( f">>>>>>>>>>>>>>>>>>>>>>>>\n\n{result}" )
        _assert_successful_response(result, "test_eligibility_check_without_date")

    def test_eligibility_check_tool_results(self, client_node):
        """Verify tool_results are populated after eligibility check."""
        state = _make_state(
            f"Verify eligibility for member {TEST_MEMBER_ID}"
        )
        result = client_node(state)

        assert "tool_results" in result
        # tool_results may be populated by the remote supervisor
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Coverage Lookup Tests
# ---------------------------------------------------------------------------

class TestCoverageLookup:
    """Tests for coverage lookup skill via A2A."""

    def test_coverage_lookup_basic(self, client_node):
        """Look up coverage details for a known member."""
        state = _make_state(
            f"What is the coverage for member {TEST_MEMBER_ID}?"
        )
        result = client_node(state)
        print( f">>>>>>>>>>>>>>>>>\n\n{result}" )
        
        _assert_successful_response(result, "test_coverage_lookup_basic")

    def test_coverage_lookup_with_procedure(self, client_node):
        """Look up coverage for a specific procedure code."""
        state = _make_state(
            f"Is procedure code 99213 covered for member {TEST_MEMBER_ID}?"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_coverage_lookup_with_procedure")

    def test_coverage_deductible_query(self, client_node):
        """Query deductible and copay information."""
        state = _make_state(
            f"What is the deductible remaining and copay for member {TEST_MEMBER_ID}?"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_coverage_deductible_query")


# ---------------------------------------------------------------------------
# Multi-skill / Routing Tests
# ---------------------------------------------------------------------------

class TestA2ARouting:
    """Tests that verify the supervisor routes correctly to the right worker."""

    def test_complex_query_routes_appropriately(self, client_node):
        """A complex query should still produce a meaningful response."""
        state = _make_state(
            f"Look up member {TEST_MEMBER_ID}, check if they are eligible "
            f"for services today, and tell me their deductible."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_complex_query_routes_appropriately")

    def test_a2a_task_state_completed(self, client_node):
        """Successful tasks should have a2a_task_state of 'completed'."""
        state = _make_state(f"Look up member {TEST_MEMBER_ID}")
        result = client_node(state)

        task_state = result.get("a2a_task_state")
        assert task_state == "completed", \
            f"Expected 'completed', got '{task_state}'"

    def test_session_isolation(self, client_node):
        """Two requests with different session IDs should not interfere."""
        session_a = f"test-session-a-{uuid.uuid4().hex[:6]}"
        session_b = f"test-session-b-{uuid.uuid4().hex[:6]}"

        state_a = _make_state(f"Look up member {TEST_MEMBER_ID}", session_id=session_a)
        state_b = _make_state(f"Look up member {TEST_MEMBER_ID}", session_id=session_b)

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
        state = _make_state("")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_member_id_format(self, client_node):
        """A member ID in wrong format should be handled gracefully."""
        state = _make_state("Look up member INVALID-ID-FORMAT")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_result_has_no_unhandled_exception(self, client_node):
        """Any query should return a result dict, never raise an exception."""
        queries = [
            "What is my coverage?",
            "Check eligibility",
            f"Tell me about member {TEST_MEMBER_ID}",
        ]
        for query in queries:
            state = _make_state(query)
            try:
                result = client_node(state)
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            except Exception as e:
                pytest.fail(f"A2AClientNode raised unexpected exception for '{query}': {e}")