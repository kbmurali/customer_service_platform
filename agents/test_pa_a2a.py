"""
Integration Tests — PA Services A2A Server
================================================
Tests the live PA services A2A supervisor reachable via Nginx api-gateway
at https://localhost:8443/a2a/pa using A2AClientNode as the client.

Prerequisites:
    - api-gateway running on localhost:8443 (mTLS)
    - pa-services-supervisor running and healthy
    - pa-services-mcp-tools running and healthy
    - .env with:
        MCP_CLIENT_CERT=/home/kbmurali/tmp/certs/mcp/client.crt
        MCP_CLIENT_KEY=/home/kbmurali/tmp/certs/mcp/client.key
        MCP_CA_CERT=/home/kbmurali/tmp/certs/mcp/ca.crt
        A2A_PA_SERVICES_URL=https://localhost:8443/a2a/pa

    - TEST_PA_ID should be a valid PA UUID from the dev knowledge graph
      (e.g. a known paId from the dev seed data)
    - TEST_PA_NUMBER should be a valid PA number from the dev KG
      (e.g. PA-2024-0001 — this is the paNumber field, NOT the UUID)
    - TEST_PROCEDURE_CODE should be a valid CPT code from the dev KG
      (e.g. 27447 — total knee arthroplasty)
    - TEST_POLICY_TYPE should be a valid policy type (HMO, PPO, EPO, POS)

Run:
    cd customer_service_platform
    pytest agents/teams/pa_services/test_pa_a2a.py -v
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
from agents.teams.pa_services.supervisor.tool_schemas import build_pa_schema_registry
from agents.core.a2a_client_node import A2AClientNode
from databases.context_graph_data_access import ContextGraphDataAccess

from agents.security import rbac_service, rate_limiter, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_URL  = os.getenv("A2A_PA_SERVICES_URL", "https://localhost:8443/a2a/pa")
AGENT_NAME = "pa_services_supervisor_agent"

TEST_USER_ID        = os.getenv("TEST_USER_ID",        "usr-tier2-001")
TEST_USER_ROLE      = os.getenv("TEST_USER_ROLE",      "CSR_TIER2")
TEST_SESSION_ID     = os.getenv("TEST_SESSION_ID",     str(uuid.uuid1()))
TEST_PA_ID          = os.getenv("TEST_PA_ID",          "cc0af705-9a9b-46e7-b308-a69c4502b817")
TEST_PA_NUMBER      = os.getenv("TEST_PA_NUMBER",      "PA-844196")
TEST_PROCEDURE_CODE = os.getenv("TEST_PROCEDURE_CODE", "29881")
TEST_POLICY_TYPE    = os.getenv("TEST_POLICY_TYPE",    "PPO")


cg_dao = ContextGraphDataAccess()
cg_dao.create_session(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)


def exceed_rate_limit(user_id: str, user_role: str, tool_name: str):
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
        schema_registry=build_pa_schema_registry(),
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
    """Verify the PA Services A2A server is reachable before task tests."""

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
        """GET /.well-known/agent.json should return a valid PA agent card."""
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
        assert "pa_lookup"       in skill_ids, f"pa_lookup skill missing: {skill_ids}"
        assert "pa_status"       in skill_ids, f"pa_status skill missing: {skill_ids}"
        assert "pa_requirements" in skill_ids, f"pa_requirements skill missing: {skill_ids}"

        logger.info("Agent card: name=%s, skills=%s", card.get("name"), skill_ids)


# ---------------------------------------------------------------------------
# PA Lookup Tests
# ---------------------------------------------------------------------------

class TestPALookup:
    """Tests for PA lookup skill via A2A."""

    def test_pa_lookup_by_id(self, client_node):
        """Look up a known prior authorization by PA ID (UUID)."""
        state  = _make_state(f"Look up prior authorization {TEST_PA_ID}")
        
        #exceed_rate_limit( user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="pa_lookup" )
        
        result = client_node(state)
        
        #print( f">>>>>>>>\n\n{result}\n")
        
        _assert_successful_response(result, "test_pa_lookup_by_id")

        content = result["messages"][-1].content
        # Response should reference the PA or its ID
        assert TEST_PA_ID in content or "authorization" in content.lower() or "pa" in content.lower(), \
            f"Expected PA reference in response, got: {content[:200]}"

    def test_pa_lookup_execution_path(self, client_node):
        """Verify execution path includes pa_lookup worker."""
        state  = _make_state(f"Get full details for PA {TEST_PA_ID}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("pa_lookup" in str(step).lower() for step in path), \
            f"Expected pa_lookup in execution path, got: {path}"

    def test_pa_lookup_unknown_id(self, client_node):
        """Look up a PA ID that doesn't exist — should return graceful response."""
        state  = _make_state("Look up prior authorization 00000000-0000-0000-0000-000000000000")
        result = client_node(state)

        # Should not crash — either an error message or graceful not-found
        assert result is not None
        assert "error" in result
        content = result["error"] if result["error"] else ""
        assert content.strip(), "Expected non-empty response for unknown PA ID"

    def test_pa_lookup_tool_results(self, client_node):
        """Verify tool_results are populated after PA lookup."""
        state  = _make_state(f"Look up prior authorization {TEST_PA_ID}")
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# PA Status Tests
# ---------------------------------------------------------------------------

class TestPAStatus:
    """Tests for PA status skill via A2A.

    Note: pa_status takes a PA ID (UUID), not a PA number.
    Use TEST_PA_ID for these tests.
    """

    def test_pa_status_by_id(self, client_node):
        """Check status of a known prior authorization by PA ID."""
        state  = _make_state(f"What is the status of prior authorization {TEST_PA_ID}?")
        result = client_node(state)
        _assert_successful_response(result, "test_pa_status_by_id")

        content = result["messages"][-1].content
        assert "authorization" in content.lower() or "status" in content.lower() or "pa" in content.lower(), \
            f"Expected PA reference in status response, got: {content[:200]}"

    def test_pa_status_execution_path(self, client_node):
        """Verify execution path includes pa_status worker."""
        state  = _make_state(f"Check PA status for {TEST_PA_ID}")
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("pa_status" in str(step).lower() for step in path), \
            f"Expected pa_status in execution path, got: {path}"

    def test_pa_status_unknown_id(self, client_node):
        """Look up a PA ID that doesn't exist — should return graceful response."""
        state  = _make_state("What is the status of PA 00000000-0000-0000-0000-000000000000?")
        result = client_node(state)

        assert result is not None
        assert "error" in result
        content = result["error"] if result["error"] else ""
        assert content.strip(), "Expected non-empty response for unknown PA ID"

    def test_pa_status_without_id(self, client_node):
        """Status query without providing a PA ID — supervisor should SKIP gracefully."""
        state  = _make_state("What is the status of my prior authorization?")
        result = client_node(state)

        # SKIP path — supervisor has no PA ID to route with.
        # Should still return a non-empty response.
        assert result is not None
        assert "messages" in result


# ---------------------------------------------------------------------------
# PA Requirements Tests
# ---------------------------------------------------------------------------

class TestPARequirements:
    """Tests for PA requirements skill via A2A.

    Note: pa_requirements takes a procedure code (CPT code) AND a policy
    type — both are required. Use TEST_PROCEDURE_CODE and TEST_POLICY_TYPE.
    """

    def test_pa_requirements_basic(self, client_node):
        """Look up PA requirements for a known procedure code and policy type."""
        state  = _make_state(
            f"Does procedure {TEST_PROCEDURE_CODE} require prior authorization "
            f"under a {TEST_POLICY_TYPE} plan?"
        )
        result = client_node(state)
        #print( f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_pa_requirements_basic")

        content = result["messages"][-1].content
        assert (
            "authorization" in content.lower()
            or "pa" in content.lower()
            or "procedure" in content.lower()
            or TEST_PROCEDURE_CODE in content
        ), f"Expected PA requirements reference in response, got: {content[:200]}"

    def test_pa_requirements_combo(self, client_node):
        """Look up PA requirements for a known procedure code and policy type."""
        state  = _make_state(
            f"Does procedure {TEST_PROCEDURE_CODE} require prior authorization "
            f"under a {TEST_POLICY_TYPE} plan?."
            f"Also what is the status of prior authorization {TEST_PA_ID}?"
        )
        #exceed_rate_limit( user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="pa_requirements" )
        result = client_node(state)
        #print( f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_pa_requirements_combo")

        content = result["messages"][-1].content
        assert (
            "authorization" in content.lower()
            or "pa" in content.lower()
            or "procedure" in content.lower()
            or TEST_PROCEDURE_CODE in content
        ), f"Expected PA requirements reference in response, got: {content[:200]}"
        
    def test_pa_requirements_execution_path(self, client_node):
        """Verify execution path includes pa_requirements worker."""
        state  = _make_state(
            f"Check if CPT {TEST_PROCEDURE_CODE} needs PA for {TEST_POLICY_TYPE} policy"
        )
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("pa_requirements" in str(step).lower() for step in path), \
            f"Expected pa_requirements in execution path, got: {path}"

    def test_pa_requirements_all_policy_types(self, client_node):
        """Check PA requirements for the same procedure across all policy types."""
        for policy_type in ["HMO", "PPO", "EPO", "POS"]:
            state  = _make_state(
                f"Does CPT {TEST_PROCEDURE_CODE} require PA under a {policy_type} plan?"
            )
            result = client_node(state)
            assert result is not None, f"Result is None for policy_type={policy_type}"
            assert "messages" in result, f"No messages for policy_type={policy_type}"
            logger.info(
                "PA requirements for %s/%s: %s",
                TEST_PROCEDURE_CODE, policy_type,
                result["messages"][-1].content[:100] if result["messages"] else "no content"
            )

    def test_pa_requirements_unknown_procedure(self, client_node):
        """Unknown procedure code — should return graceful not-found response."""
        state  = _make_state(
            f"Does procedure 00000 require prior authorization under a {TEST_POLICY_TYPE} plan?"
        )
        result = client_node(state)

        assert result is not None
        assert "messages" in result

    def test_pa_requirements_without_policy_type(self, client_node):
        """Requirements query missing policy type — supervisor should SKIP gracefully."""
        state  = _make_state(
            f"Does procedure {TEST_PROCEDURE_CODE} require prior authorization?"
        )
        result = client_node(state)

        # SKIP path — supervisor has no policy type to route with.
        # Should still return a non-empty response.
        assert result is not None
        assert "messages" in result

    def test_pa_requirements_tool_results(self, client_node):
        """Verify tool_results are populated after PA requirements lookup."""
        state  = _make_state(
            f"Does CPT {TEST_PROCEDURE_CODE} need PA under {TEST_POLICY_TYPE}?"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Multi-skill / Routing Tests
# ---------------------------------------------------------------------------

class TestA2ARouting:
    """Tests that verify the supervisor routes correctly to the right worker."""

    def test_lookup_and_status_query(self, client_node):
        """A query asking for PA details and status should route to both workers."""
        state  = _make_state(
            f"Look up prior authorization {TEST_PA_ID} and tell me its current status."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_lookup_and_status_query")

        path     = result.get("execution_path", [])
        path_str = " ".join(str(s) for s in path).lower()
        assert "pa_lookup" in path_str or "pa_status" in path_str, \
            f"Expected PA workers in execution path, got: {path}"

    def test_a2a_task_state_completed(self, client_node):
        """Successful tasks should have execution path ending with FINISH."""
        state  = _make_state(f"Look up prior authorization {TEST_PA_ID}")
        result = client_node(state)

        exec_path_list = result.get("execution_path")
        assert "pa_services_supervisor -> FINISH (all steps done)" in exec_path_list, \
            f"Expected FINISH in execution path, got '{exec_path_list}'"

    def test_pa_id_vs_requirements_routing(self, client_node):
        """Supervisor should route procedure/policy queries to pa_requirements,
        not pa_lookup or pa_status."""
        state  = _make_state(
            f"Does procedure {TEST_PROCEDURE_CODE} need PA for {TEST_POLICY_TYPE}?"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_pa_id_vs_requirements_routing")

        path     = result.get("execution_path", [])
        path_str = " ".join(str(s) for s in path).lower()
        assert "pa_requirements" in path_str, \
            f"Expected pa_requirements worker for procedure/policy query, got: {path}"

    def test_session_isolation(self, client_node):
        """Two requests with different session IDs should not interfere."""
        session_a = f"test-session-a-{uuid.uuid4().hex[:6]}"
        session_b = f"test-session-b-{uuid.uuid4().hex[:6]}"

        state_a = _make_state(f"Look up prior authorization {TEST_PA_ID}", session_id=session_a)
        state_b = _make_state(f"Look up prior authorization {TEST_PA_ID}", session_id=session_b)

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

    def test_invalid_pa_id_format(self, client_node):
        """A PA ID in wrong format should be handled gracefully."""
        state  = _make_state("Look up prior authorization INVALID-PA-ID-FORMAT")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_procedure_code_format(self, client_node):
        """A procedure code in wrong format should be handled gracefully."""
        state  = _make_state(
            f"Does procedure NOTACPTCODE require PA under {TEST_POLICY_TYPE}?"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_policy_type(self, client_node):
        """An unrecognised policy type should be handled gracefully."""
        state  = _make_state(
            f"Does procedure {TEST_PROCEDURE_CODE} require PA under a UNKNOWN plan?"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_result_has_no_unhandled_exception(self, client_node):
        """Any query should return a result dict, never raise an exception."""
        queries = [
            "What is the status of my prior authorization?",
            "Does my procedure require prior authorization?",
            f"Look up prior authorization {TEST_PA_ID}",
            f"Check status of PA {TEST_PA_ID}",
            f"Does CPT {TEST_PROCEDURE_CODE} need PA under {TEST_POLICY_TYPE}?",
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
