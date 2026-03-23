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
from agents.security import rbac_service, rate_limiter, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_URL         = os.getenv("A2A_MEMBER_SERVICES_URL", "https://localhost:8443/a2a/member")
AGENT_NAME      = "member_services_supervisor_agent"
TEST_USER_ID    = os.getenv("TEST_USER_ID",    "usr-tier2-001")
TEST_USER_ROLE  = os.getenv("TEST_USER_ROLE",  "CSR_TIER2")
TEST_SESSION_ID = os.getenv("TEST_SESSION_ID", str( uuid.uuid1()))
# TEST_MEMBER_ID must be a valid member UUID from the dev seed data.
# The default below is a known seed member — override via TEST_MEMBER_ID env var.
TEST_MEMBER_ID  = os.getenv("TEST_MEMBER_ID",  "68a42d4f-9656-4f4b-bbbc-dda380dc09e1")
TEST_MEMBER_ID2 = os.getenv("TEST_MEMBER_ID2", "68a42d4f-9656-4f4b-bbbc-dda380dc09e1")

cg_dao = ContextGraphDataAccess()
cg_dao.create_session( session_id=TEST_SESSION_ID, user_id=TEST_USER_ID )


def exceed_rate_limit( user_id: str, user_role: str, tool_name: str ):
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


def _get_response_content(result: Dict[str, Any]) -> str:
    """Safely extract the last message content from a result dict.

    Returns empty string if messages list is empty or content is missing,
    rather than raising IndexError.
    """
    messages = result.get("messages", [])
    if not messages:
        return ""
    last_msg = messages[-1]
    return last_msg.content if hasattr(last_msg, "content") else str(last_msg)


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
        """GET /.well-known/agent.json should return a valid agent card.

        Skill IDs must match MySQL tool_names exactly:
            - check_eligibility (NOT eligibility_check)
            - member_policy_lookup (added during development)
        """
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

            skill_ids = [s.get("id") for s in card.get("skills", [])]

            # Core skills
            assert "member_lookup"       in skill_ids, f"member_lookup skill missing: {skill_ids}"
            assert "check_eligibility"   in skill_ids, f"check_eligibility skill missing: {skill_ids}"
            assert "coverage_lookup"     in skill_ids, f"coverage_lookup skill missing: {skill_ids}"
            assert "update_member_info"  in skill_ids, f"update_member_info skill missing: {skill_ids}"
            assert "member_policy_lookup" in skill_ids, f"member_policy_lookup skill missing: {skill_ids}"

            # Verify old name is NOT present (A2A/MySQL reconciliation)
            assert "eligibility_check" not in skill_ids, (
                f"eligibility_check (old name) should NOT be in agent card, "
                f"expected check_eligibility. Got: {skill_ids}"
            )

            logger.info("Agent card: name=%s, skills=%s", card.get("name"), skill_ids)


# ---------------------------------------------------------------------------
# Member Lookup Tests
# ---------------------------------------------------------------------------

class TestMemberLookup:
    """Tests for member lookup skill via A2A."""

    def test_member_lookup_by_id(self, client_node):
        """Look up a known member by ID."""
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
        assert "error" in result
        content = result["error"] if result["error"] else ""
        assert content.strip(), "Expected non-empty response for unknown member"


# ---------------------------------------------------------------------------
# Eligibility Check Tests
# ---------------------------------------------------------------------------

class TestEligibilityCheck:
    """Tests for check_eligibility skill via A2A."""

    def test_eligibility_check_basic(self, client_node):
        """Check eligibility for a known member."""
        state = _make_state(
            f"Check eligibility for member {TEST_MEMBER_ID} for service date 2026-02-22"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_eligibility_check_basic")

    def test_eligibility_check_without_date(self, client_node):
        """Check eligibility without specifying a service date."""
        state = _make_state(f"Is member {TEST_MEMBER_ID} currently eligible?")
        result = client_node(state)
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
        _assert_successful_response(result, "test_coverage_lookup_basic")

    def test_coverage_lookup_combo(self, client_node):
        """Look up coverage details for a known member."""
        state = _make_state(
            f"Check eligibility for member {TEST_MEMBER_ID} for service date 2026-02-22."
            f"Also, What is the coverage for member {TEST_MEMBER_ID2} for procedure code 99213?."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_coverage_lookup_combo")

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
# Member Policy Lookup Tests
# ---------------------------------------------------------------------------

class TestMemberPolicyLookup:
    """Tests for member_policy_lookup skill via A2A.

    member_policy_lookup returns member demographics + associated policy
    details (plan name, plan type, effective/expiration dates, premium,
    deductible, out-of-pocket max).
    """

    def test_member_policy_lookup_basic(self, client_node):
        """Look up member + policy details for a known member."""
        state = _make_state(
            f"Show me the policy details for member {TEST_MEMBER_ID}"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_member_policy_lookup_basic")

    def test_member_policy_lookup_includes_plan_info(self, client_node):
        """Response should reference plan/policy information."""
        state = _make_state(
            f"What insurance plan does member {TEST_MEMBER_ID} have?"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_member_policy_lookup_plan_info")
        content = _get_response_content(result).lower()
        assert any(kw in content for kw in [
            "plan", "policy", "premium", "deductible", "coverage"
        ]), f"Expected plan/policy info in response: {content[:200]}"


# ---------------------------------------------------------------------------
# Update Member Info Tests
# ---------------------------------------------------------------------------

class TestUpdateMemberInfo:
    """Tests for update_member_info skill via A2A.

    Note: update_member_info takes a member ID (UUID), a field to update,
    the new value, and a reason for the change.
    Updatable fields: phone, email, address_street, address_city,
    address_state, address_zip, enrollmentDate, status.
    HIGH-IMPACT write operation — requires human approval workflow.
    """

    def test_update_phone_number(self, client_node):
        """Update a member's phone number with a valid reason."""
        state = _make_state(
            f"Update phone number for member {TEST_MEMBER_ID} to 555-9876 — "
            f"member called in to request contact information update."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_update_phone_number")
        content = _get_response_content(result)
        assert (
            "updated" in content.lower()
            or "phone" in content.lower()
            or "member" in content.lower()
            or "approval" in content.lower()
            or "pending" in content.lower()
        ), f"Expected update/approval reference in response, got: {content[:200]}"

    def test_update_email_address(self, client_node):
        """Update a member's email address with a valid reason."""
        state = _make_state(
            f"Change email for member {TEST_MEMBER_ID} to newemail@example.com — "
            f"member reported previous email address is no longer valid."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_update_email_address")
        content = _get_response_content(result)
        assert (
            "updated" in content.lower()
            or "email" in content.lower()
            or "member" in content.lower()
            or "approval" in content.lower()
            or "pending" in content.lower()
        ), f"Expected update/approval reference in response, got: {content[:200]}"

    def test_update_member_info_tool_results(self, client_node):
        """Verify update_member_info key is present in tool_results."""
        state = _make_state(
            f"Update address_city for member {TEST_MEMBER_ID} to Springfield — "
            f"member submitted address change form."
        )
        result = client_node(state)
        assert "tool_results" in result
        assert "update_member_info" in result["tool_results"], \
            f"Expected update_member_info in tool_results, got: {list(result['tool_results'].keys())}"
        logger.info("Update tool results: %s",
            result["tool_results"].get("update_member_info", {}).get("output", "")[:150])

    def test_update_without_reason_skips(self, client_node):
        """Update query missing a reason — supervisor should SKIP gracefully."""
        state = _make_state(
            f"Update phone for member {TEST_MEMBER_ID} to 555-1111"
        )
        result = client_node(state)
        # SKIP path — no reason provided.
        # Should still return a non-empty response without crashing.
        assert result is not None
        assert "messages" in result

    def test_update_without_member_id_skips(self, client_node):
        """Update query missing a member ID — supervisor should SKIP gracefully."""
        state = _make_state(
            "Update the member's phone number to 555-1111 — member requested change."
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_update_unknown_member_id(self, client_node):
        """Updating a zero-UUID member ID via the write path.

        The @require_approvals MCP decorator intercepts before the KG lookup,
        returning a pending-approval response rather than a not-found error.
        Assert the supervisor routed correctly and returned a non-empty response.

        Guard against empty messages list (approval workflow may return
        result with error field instead of messages).
        """
        state = _make_state(
            "Update phone for member 00000000-0000-0000-0000-000000000000 to 555-9999 — "
            "member requested update."
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

        # Safely extract content — messages may be empty if approval
        # workflow short-circuits the response
        content = _get_response_content(result)
        error = result.get("error", "")
        assert content.strip() or (error and str(error).strip()), (
            "Expected either a non-empty response or an error for zero-UUID member update, "
            f"got messages={result.get('messages')}, error={error}"
        )

    def test_update_invalid_field_handled(self, client_node):
        """An unrecognised field name should be handled gracefully."""
        state = _make_state(
            f"Update ssn for member {TEST_MEMBER_ID} to 000-00-0000 — correction needed."
        )
        result = client_node(state)
        # ssn is not in the allowed fields list — should either SKIP or
        # return a graceful error from the MCP tool.
        assert result is not None
        assert "messages" in result


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
        """Successful tasks should complete without errors."""
        state = _make_state(f"Look up member {TEST_MEMBER_ID}")
        result = client_node(state)
        # Team-internal FINISH entries no longer propagate to A2AClientNode callers.
        # Verify task completion via absence of error and non-empty response.
        assert result.get("error") is None, \
            f"Expected no error on completed task, got: {result.get('error')}"
        assert result.get("messages"), "Expected non-empty messages on completed task"
        last_content = _get_response_content(result)
        assert last_content.strip(), "Expected non-empty response content on completed task"

    def test_session_isolation(self, client_node):
        """Two requests with different session IDs should not interfere."""
        session_a = f"test-session-a-{uuid.uuid4().hex[:6]}"
        session_b = f"test-session-b-{uuid.uuid4().hex[:6]}"

        cg_dao.create_session( session_id=session_a, user_id=TEST_USER_ID )
        cg_dao.create_session( session_id=session_b, user_id=TEST_USER_ID )

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
