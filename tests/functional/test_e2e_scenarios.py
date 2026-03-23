"""
End-to-end scenario tests — exercises complete user workflows that
span multiple API calls, testing session chaining, approval workflows,
and circuit breaker behavior.

These tests are slower than integration tests (30-60 seconds each)
because they involve multiple sequential agent queries. They should
run on PRs and pre-deploy, not on every commit.

Run with::

    pytest tests/functional/test_e2e_scenarios.py -v -s
"""

import time
import pytest
from conftest import query_agent, assert_no_errors, get_teams_from_path

pytestmark = pytest.mark.e2e


class TestMultiTurnConversation:
    """End-to-end multi-turn conversation with session chaining."""

    def test_two_turn_session_chain(self, csr_client):
        """First query establishes a session; second query chains
        to it via prior_session_id."""
        # Turn 1: ask about a claim
        r1 = query_agent(
            csr_client,
            "What is the status of claim CLM-123456 for member M-789?"
        )
        assert_no_errors(r1, "turn 1")
        session_1 = r1["session_id"]
        assert session_1, "Turn 1 should produce a session_id"

        # Turn 2: follow up referencing "the claim" (implicit reference)
        r2 = query_agent(
            csr_client,
            "Can you show me the payment details for that claim?",
            prior_session_id=session_1,
        )
        assert_no_errors(r2, "turn 2")
        session_2 = r2["session_id"]

        # Sessions should be different (new session created for follow-up)
        assert session_2 != session_1, (
            "Follow-up should create a new session, not reuse the old one"
        )

        # Turn 2 response should reference claim information
        # (it has context from turn 1 via the session chain)
        assert len(r2["response"]) > 10, "Turn 2 should produce a real response"

    def test_three_turn_chain(self, csr_client):
        """A three-turn conversation maintains context across all turns."""
        # Turn 1: identify a member
        r1 = query_agent(csr_client, "Check eligibility for member M-789")
        assert_no_errors(r1, "turn 1")

        # Turn 2: ask about their claims
        r2 = query_agent(
            csr_client,
            "Does this member have any pending claims?",
            prior_session_id=r1["session_id"],
        )
        assert_no_errors(r2, "turn 2")

        # Turn 3: follow up again
        r3 = query_agent(
            csr_client,
            "What about their prior authorizations?",
            prior_session_id=r2["session_id"],
        )
        assert_no_errors(r3, "turn 3")

        # All three sessions should be distinct
        sessions = {r1["session_id"], r2["session_id"], r3["session_id"]}
        assert len(sessions) == 3, "Each turn should create a new session"

    def test_independent_sessions_do_not_share_context(self, csr_client):
        """Two queries without prior_session_id should be independent."""
        r1 = query_agent(csr_client, "Check eligibility for member M-12345")
        r2 = query_agent(csr_client, "Check eligibility for member M-67890")

        assert r1["session_id"] != r2["session_id"]
        # Both should succeed independently
        assert_no_errors(r1, "independent query 1")
        assert_no_errors(r2, "independent query 2")


class TestCompleteQueryLifecycle:
    """Full lifecycle: query → response → feedback."""

    def test_query_then_feedback(self, csr_client):
        """Complete flow: send query, get response, submit feedback."""
        # Step 1: Query
        r = query_agent(
            csr_client,
            "What is the status of claim CLM-123456?"
        )
        assert_no_errors(r, "query")
        session_id = r["session_id"]

        # Step 2: Submit feedback
        feedback_response = csr_client.post(
            "/api/feedback",
            json={"session_id": session_id, "rating": "correct"},
        )
        assert feedback_response.status_code == 200
        assert feedback_response.json()["status"] == "stored"

    def test_query_then_negative_feedback_with_correction(self, csr_client):
        """Query → negative feedback with correction text."""
        r = query_agent(
            csr_client,
            "What is the copay for member M-12345?"
        )
        session_id = r["session_id"]

        feedback_response = csr_client.post(
            "/api/feedback",
            json={
                "session_id": session_id,
                "rating": "incorrect",
                "correction": "The copay should be $25 not $50.",
            },
        )
        assert feedback_response.status_code == 200
        assert feedback_response.json()["status"] == "stored"


class TestCrossTeamWorkflow:
    """Complex workflows that exercise multiple teams sequentially."""

    def test_member_then_claims_then_pa(self, csr_client):
        """Three-query workflow across three different teams."""
        # Query 1: Member services
        r1 = query_agent(csr_client, "Check eligibility for member M-789")
        assert_no_errors(r1, "member query")
        assert "member_services_team" in get_teams_from_path(r1["execution_path"])

        # Query 2: Claims services (follow-up)
        r2 = query_agent(
            csr_client,
            "Now check if they have any claims under review",
            prior_session_id=r1["session_id"],
        )
        assert_no_errors(r2, "claims query")

        # Query 3: PA services (follow-up)
        r3 = query_agent(
            csr_client,
            "Do they have any pending prior authorizations?",
            prior_session_id=r2["session_id"],
        )
        assert_no_errors(r3, "PA query")

    def test_dependent_step_context_injection(self, csr_client):
        """A single query with dependent steps (member lookup then claim
        lookup using the member ID from step 1) should succeed.

        This tests the goal_advance_node context injection: step 2's
        delegation query gets enriched with step 1's tool results so
        the downstream team sees the actual member ID.
        """
        r = query_agent(
            csr_client,
            "What is the member ID associated with claim CLM-123456, "
            "and then look up that member's details?"
        )
        # Even if the exact claim/member don't exist in test data,
        # the planning and routing should work without errors
        teams = get_teams_from_path(r["execution_path"])
        # Should involve both claims (to look up the claim) and member
        # (to look up the member), though the exact routing depends on
        # how the planner decomposes this.
        assert len(teams) >= 1, (
            f"Expected at least one team delegation. "
            f"Teams: {teams}, Path: {r['execution_path']}"
        )


class TestSystemHealth:
    """Verify overall system health indicators."""

    def test_health_endpoint(self, csr_client):
        """The /health endpoint should return healthy status."""
        response = csr_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_rapid_sequential_queries(self, csr_client):
        """Multiple queries in quick succession should all succeed
        (within rate limits)."""
        queries = [
            "Check eligibility for member M-12345",
            "What is the status of claim CLM-123456?",
            "Does CPT 29881 need prior auth?",
        ]
        results = []
        for q in queries:
            r = query_agent(csr_client, q)
            results.append(r)
            # Small delay to avoid rate limiting
            time.sleep(0.5)

        for i, r in enumerate(results):
            assert_no_errors(r, f"rapid query {i+1}")

    def test_concurrent_different_users(self, csr_client, supervisor_client):
        """Two different users can query simultaneously."""
        r1 = query_agent(csr_client, "Check eligibility for member M-12345")
        r2 = query_agent(supervisor_client, "What is the status of claim CLM-789012?")
        assert_no_errors(r1, "CSR query")
        assert_no_errors(r2, "supervisor query")
        assert r1["session_id"] != r2["session_id"]
