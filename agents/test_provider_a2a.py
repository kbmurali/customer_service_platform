"""
Integration Tests — Provider Services A2A Server
=================================================
Tests the live provider services A2A supervisor reachable via Nginx api-gateway
at https://localhost:8443/a2a/provider using A2AClientNode as the client.

Prerequisites:
    - api-gateway running on localhost:8443 (mTLS)
    - provider-services-a2a-server running and healthy
    - provider-services-mcp-tools running and healthy
    - .env with:
        MCP_CLIENT_CERT=/home/kbmurali/tmp/certs/mcp/client.crt
        MCP_CLIENT_KEY=/home/kbmurali/tmp/certs/mcp/client.key
        MCP_CA_CERT=/home/kbmurali/tmp/certs/mcp/ca.crt
        A2A_PROVIDER_SERVICES_URL=https://localhost:8443/a2a/provider

Run:
    cd customer_service_platform
    pytest agents/teams/provider_services/test_provider_a2a.py -v
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
from agents.teams.provider_services.supervisor.tool_schemas import build_provider_schema_registry
from agents.core.a2a_client_node import A2AClientNode
from databases.context_graph_data_access import ContextGraphDataAccess

from agents.security import rbac_service, rate_limiter, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_URL        = os.getenv("A2A_PROVIDER_SERVICES_URL", "https://localhost:8443/a2a/provider")
AGENT_NAME       = "provider_services_supervisor_agent"
TEST_USER_ID     = os.getenv("TEST_USER_ID",     "usr-tier2-001")
TEST_USER_ROLE   = os.getenv("TEST_USER_ROLE",   "CSR_TIER2")
TEST_SESSION_ID  = os.getenv("TEST_SESSION_ID",  str(uuid.uuid1()))
TEST_PROVIDER_ID = os.getenv("TEST_PROVIDER_ID", "1f4f7e66-2db0-4a2b-8e39-0c2e4e93b6eb")
TEST_POLICY_ID   = os.getenv("TEST_POLICY_ID",   "698289fe-64b2-4382-894f-d8ad5ca4a4a4")
TEST_SPECIALTY   = os.getenv("TEST_SPECIALTY",   "Dermatology")
TEST_ZIP_CODE    = os.getenv("TEST_ZIP_CODE",     "30368")

cg_dao = ContextGraphDataAccess()

cg_dao.create_session(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)


def exceed_rate_limit(user_id: str, user_role: str, tool_name: str):

    current_rate_limit = rbac_service.get_tool_rate_limit(user_role=user_role, tool_name=tool_name)

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
        schema_registry=build_provider_schema_registry(),
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
    """Verify the A2A server is reachable before running task tests."""

    def test_health_endpoint(self):
        """GET /health should return 200 with healthy status via mTLS."""
        import ssl
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
        """GET /.well-known/agent.json should return a valid agent card with provider skills."""
        import ssl
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
        assert len(card["skills"]) == 3

        skill_ids = [s.get("id") for s in card.get("skills", [])]
        assert "provider_lookup"              in skill_ids
        assert "provider_network_check"       in skill_ids
        assert "provider_search_by_specialty" in skill_ids

        logger.info("Agent card: name=%s, skills=%s", card.get("name"), skill_ids)


# ---------------------------------------------------------------------------
# Provider Lookup Tests
# ---------------------------------------------------------------------------

class TestProviderLookup:
    """Tests for provider lookup skill via A2A."""

    def test_provider_lookup_by_id(self, client_node):
        """Look up a known provider by ID."""
        state  = _make_state(f"Look up provider {TEST_PROVIDER_ID}")
        
        #exceed_rate_limit( TEST_USER_ID, TEST_USER_ROLE, "provider_lookup" )
        
        result = client_node(state)
        #print( ">>>>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_provider_lookup_by_id")

        content = result["messages"][-1].content
        assert TEST_PROVIDER_ID in content or "provider" in content.lower()

    def test_provider_lookup_execution_path(self, client_node):
        """Verify execution path includes provider_lookup worker."""
        state  = _make_state(f"Find information for provider {TEST_PROVIDER_ID}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("provider_lookup" in str(step).lower() for step in path), \
            f"Expected provider_lookup in execution path, got: {path}"

    def test_provider_lookup_unknown_provider(self, client_node):
        """Look up a provider ID that doesn't exist — should return graceful response."""
        state  = _make_state("Look up provider 00000000-0000-0000-0000-000000000000")
        result = client_node(state)

        # Should not crash — either an error message or graceful not-found
        assert result is not None
        assert "error" in result
        content = result["error"] if result["error"] else ""
        assert content.strip(), "Expected non-empty response for unknown provider"

    def test_provider_lookup_tool_results(self, client_node):
        """Verify tool_results are populated after provider lookup."""
        state  = _make_state(f"Get full details for provider {TEST_PROVIDER_ID}")
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Provider Network Check Tests
# ---------------------------------------------------------------------------

class TestProviderNetworkCheck:
    """Tests for provider network check skill via A2A."""

    def test_network_check_basic(self, client_node):
        """Check network status for a known provider under a known policy."""
        state  = _make_state(
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_network_check_basic")

    def test_network_check_combo(self, client_node):
        """Check network status for a known provider under a known policy."""
        state  = _make_state(
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?"
            f"Also, Look up provider {TEST_PROVIDER_ID}"
        )
        #exceed_rate_limit( TEST_USER_ID, TEST_USER_ROLE, "provider_lookup" )
        result = client_node(state)
        _assert_successful_response(result, "test_network_check_combo")
        
    def test_network_check_execution_path(self, client_node):
        """Verify execution path includes provider_network_check worker."""
        state  = _make_state(
            f"Check network status of provider {TEST_PROVIDER_ID} under policy {TEST_POLICY_ID}"
        )
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("provider_network_check" in str(step).lower() for step in path), \
            f"Expected provider_network_check in execution path, got: {path}"

    def test_network_check_unknown_provider(self, client_node):
        """Network check for an unknown provider — should return graceful not-found response."""
        state  = _make_state(
            f"Is provider 00000000-0000-0000-0000-000000000000 in-network for policy {TEST_POLICY_ID}?"
        )
        result = client_node(state)

        assert result is not None
        assert "messages" in result

    def test_network_check_without_policy_id(self, client_node):
        """Network check without policy ID — should SKIP gracefully (missing required data)."""
        state  = _make_state(f"Is provider {TEST_PROVIDER_ID} in-network?")
        result = client_node(state)

        # SKIP path: supervisor cannot call the tool without policy_id.
        # Execution should complete without crashing.
        assert result is not None
        assert "messages" in result

    def test_network_check_tool_results(self, client_node):
        """Verify tool_results are populated after network check."""
        state  = _make_state(
            f"Check if provider {TEST_PROVIDER_ID} has claim history under policy {TEST_POLICY_ID}"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Provider Search by Specialty Tests
# ---------------------------------------------------------------------------

class TestProviderSearchBySpecialty:
    """Tests for provider search by specialty skill via A2A."""

    def test_search_by_specialty_basic(self, client_node):
        """Search for providers by specialty and ZIP code."""
        state  = _make_state(
            f"Find {TEST_SPECIALTY} providers near ZIP {TEST_ZIP_CODE}"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_by_specialty_basic")

    def test_search_by_specialty_execution_path(self, client_node):
        """Verify execution path includes provider_search_by_specialty worker."""
        state  = _make_state(
            f"Search for {TEST_SPECIALTY} specialists in ZIP {TEST_ZIP_CODE}"
        )
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("provider_search_by_specialty" in str(step).lower() for step in path), \
            f"Expected provider_search_by_specialty in execution path, got: {path}"

    def test_search_without_zip_code(self, client_node):
        """Search without ZIP code — should SKIP gracefully (missing required data)."""
        state  = _make_state(f"Find {TEST_SPECIALTY} providers")
        result = client_node(state)

        # SKIP path: supervisor cannot call the tool without zip_code.
        # Execution should complete without crashing.
        assert result is not None
        assert "messages" in result

    def test_search_multiple_specialties(self, client_node):
        """Search for two different specialties — should produce a 2-goal plan."""
        state  = _make_state(
            f"Find {TEST_SPECIALTY} providers near ZIP {TEST_ZIP_CODE}. "
            f"Also find Orthopedics providers near ZIP {TEST_ZIP_CODE}."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_multiple_specialties")

        assert "execution_path" in result
        path = result["execution_path"]
        # Both steps should appear in the execution path
        specialty_steps = [s for s in path if "provider_search_by_specialty" in str(s).lower()]
        assert len(specialty_steps) >= 2, \
            f"Expected 2 provider_search_by_specialty steps, got: {specialty_steps}"

    def test_search_tool_results(self, client_node):
        """Verify tool_results contain count and providers after search."""
        state  = _make_state(
            f"Search for {TEST_SPECIALTY} doctors in ZIP {TEST_ZIP_CODE}"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Multi-skill / Routing Tests
# ---------------------------------------------------------------------------

class TestA2ARouting:
    """Tests that verify the supervisor routes correctly to the right worker."""

    def test_lookup_and_network_check(self, client_node):
        """Query combining provider lookup and network check — should produce 2-goal plan."""
        state  = _make_state(
            f"Look up provider {TEST_PROVIDER_ID}. "
            f"Also check if they are in-network for policy {TEST_POLICY_ID}."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_lookup_and_network_check")

        path = result.get("execution_path", [])
        assert any("provider_lookup" in str(s) for s in path), \
            f"Expected provider_lookup in path: {path}"
        assert any("provider_network_check" in str(s) for s in path), \
            f"Expected provider_network_check in path: {path}"

    def test_task_state_completed(self, client_node):
        """Successful tasks should include FINISH sentinel in execution path."""
        state  = _make_state(f"Look up provider {TEST_PROVIDER_ID}")
        result = client_node(state)

        exec_path = result.get("execution_path", [])
        assert "provider_services_supervisor -> FINISH (all steps done)" in exec_path, \
            f"Expected FINISH sentinel in execution path, got: {exec_path}"

    def test_provider_id_vs_specialty_routing(self, client_node):
        """provider_lookup and provider_search_by_specialty must not be confused."""
        # This query has both a provider ID (lookup) and a specialty+zip (search)
        state  = _make_state(
            f"Look up provider {TEST_PROVIDER_ID}. "
            f"Also find {TEST_SPECIALTY} providers near ZIP {TEST_ZIP_CODE}."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_provider_id_vs_specialty_routing")

        path = result.get("execution_path", [])
        assert any("provider_lookup" in str(s) for s in path), \
            f"Expected provider_lookup in path: {path}"
        assert any("provider_search_by_specialty" in str(s) for s in path), \
            f"Expected provider_search_by_specialty in path: {path}"

    def test_session_isolation(self, client_node):
        """Two requests with different session IDs should not interfere."""
        session_a = f"test-session-a-{uuid.uuid4().hex[:6]}"
        session_b = f"test-session-b-{uuid.uuid4().hex[:6]}"

        cg_dao.create_session(session_id=session_a, user_id=TEST_USER_ID)
        cg_dao.create_session(session_id=session_b, user_id=TEST_USER_ID)

        state_a = _make_state(f"Look up provider {TEST_PROVIDER_ID}", session_id=session_a)
        state_b = _make_state(f"Look up provider {TEST_PROVIDER_ID}", session_id=session_b)

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

    def test_invalid_provider_id_format(self, client_node):
        """A provider ID in wrong format should be handled gracefully."""
        state  = _make_state("Look up provider INVALID-PROVIDER-FORMAT")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_policy_id_format(self, client_node):
        """A policy ID in wrong format for network check should be handled gracefully."""
        state  = _make_state(
            f"Is provider {TEST_PROVIDER_ID} in-network for policy NOT-A-REAL-POLICY?"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_no_unhandled_exception(self, client_node):
        """Any query should return a result dict, never raise an exception."""
        queries = [
            "Find a cardiologist near me",
            f"Check network status of provider {TEST_PROVIDER_ID}",
            f"Look up provider {TEST_PROVIDER_ID}",
            f"Search for Dermatology providers in ZIP {TEST_ZIP_CODE}",
        ]
        for query in queries:
            state = _make_state(query)
            try:
                result = client_node(state)
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            except Exception as e:
                pytest.fail(f"A2AClientNode raised unexpected exception for '{query}': {e}")
