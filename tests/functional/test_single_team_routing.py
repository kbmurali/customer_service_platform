"""
Single-team routing tests — verifies that queries route to the correct
team and produce structured responses with the expected tool results.

Each test sends a query that should be handled by exactly one team
and asserts on the execution path and tool result keys.

Run with::

    pytest tests/functional/test_single_team_routing.py -v
"""

import pytest
from conftest import query_agent, assert_no_errors, assert_team_routed

pytestmark = pytest.mark.integration


class TestMemberServicesRouting:
    """Queries that should route to member_services_team."""

    def test_check_eligibility(self, csr_client):
        """Eligibility queries route to member services (check_eligibility tool)."""
        r = query_agent(csr_client, "Check eligibility for member M-12345")
        assert_no_errors(r, "check_eligibility")
        assert_team_routed(r, "member_services_team")
        assert r["response"], "Response should not be empty"

    def test_member_lookup(self, csr_client):
        """Member lookup queries route to member services."""
        r = query_agent(csr_client, "Look up member M-67890 and show me their details")
        assert_no_errors(r, "member lookup")
        assert_team_routed(r, "member_services_team")

    def test_member_policy_lookup(self, csr_client):
        """Member+policy queries route to member services (member_policy_lookup tool)."""
        r = query_agent(csr_client, "Show me the policy details for member M-12345")
        assert_no_errors(r, "member_policy_lookup")
        assert_team_routed(r, "member_services_team")

    def test_coverage_question(self, csr_client):
        """Coverage questions route to member services."""
        r = query_agent(csr_client, "What is the coverage for member M-11111 under their current policy?")
        assert_no_errors(r, "coverage question")
        assert_team_routed(r, "member_services_team")


class TestClaimsServicesRouting:
    """Queries that should route to claims_services_team."""

    def test_claim_status(self, csr_client):
        """Claim status queries route to claims services."""
        r = query_agent(csr_client, "What is the status of claim CLM-123456?")
        assert_no_errors(r, "claim status")
        assert_team_routed(r, "claims_services_team")

    def test_claim_payment_info(self, csr_client):
        """Claim payment queries route to claims services."""
        r = query_agent(csr_client, "What are the payment details for claim CLM-789012?")
        assert_no_errors(r, "claim payment")
        assert_team_routed(r, "claims_services_team")

    def test_claim_denial_reason(self, csr_client):
        """Claim denial queries route to claims services."""
        r = query_agent(csr_client, "Why was claim CLM-456789 denied?")
        assert_no_errors(r, "claim denial")
        assert_team_routed(r, "claims_services_team")


class TestPAServicesRouting:
    """Queries that should route to pa_services_team."""

    def test_pa_requirements(self, csr_client):
        """Prior auth requirement queries route to PA services."""
        r = query_agent(csr_client, "Does CPT 29881 require prior authorization under a PPO plan?")
        assert_no_errors(r, "PA requirements")
        assert_team_routed(r, "pa_services_team")

    def test_pa_status(self, csr_client):
        """PA status queries route to PA services."""
        r = query_agent(csr_client, "What is the status of prior authorization PA-2024-001?")
        assert_no_errors(r, "PA status")
        assert_team_routed(r, "pa_services_team")


class TestProviderServicesRouting:
    """Queries that should route to provider_services_team."""

    def test_network_status(self, csr_client):
        """Provider network queries route to provider services."""
        r = query_agent(csr_client, "Is Dr. Chen with NPI 1234567890 in-network?")
        assert_no_errors(r, "network status")
        assert_team_routed(r, "provider_services_team")

    def test_provider_search(self, csr_client):
        """Provider search queries route to provider services."""
        r = query_agent(csr_client, "Find in-network cardiologists near ZIP 60601")
        assert_no_errors(r, "provider search")
        assert_team_routed(r, "provider_services_team")


class TestSearchServicesRouting:
    """Queries that should route to search_services_team."""

    def test_medical_code_lookup(self, csr_client):
        """Medical code queries route to search services."""
        r = query_agent(
            csr_client,
            "What is the CPT procedure code for knee replacement surgery for a health insurance claim?"
        )
        assert_no_errors(r, "medical code")
        assert_team_routed(r, "search_services_team")

    def test_policy_question(self, csr_client):
        """Policy document queries route to search services."""
        r = query_agent(csr_client, "What does the PPO policy say about out-of-network mental health coverage?")
        assert_no_errors(r, "policy question")
        assert_team_routed(r, "search_services_team")


class TestResponseStructure:
    """Verify the structural integrity of all query responses."""

    def test_session_id_is_present(self, csr_client):
        """Every response must include a non-empty session_id."""
        r = query_agent(csr_client, "Check eligibility for member M-12345")
        assert r["session_id"], "session_id should not be empty"
        assert len(r["session_id"]) > 8, "session_id should be a UUID-like string"

    def test_execution_path_is_list(self, csr_client):
        """execution_path must be a non-empty list of strings."""
        r = query_agent(csr_client, "What is the status of claim CLM-123456?")
        assert isinstance(r["execution_path"], list)
        assert len(r["execution_path"]) > 0
        assert all(isinstance(n, str) for n in r["execution_path"])

    def test_execution_path_contains_create_plan(self, csr_client):
        """Every query should pass through create_plan."""
        r = query_agent(csr_client, "Check eligibility for member M-12345")
        assert "create_plan" in r["execution_path"], (
            f"Expected create_plan in path: {r['execution_path']}"
        )

    def test_tool_results_is_dict(self, csr_client):
        """tool_results must be a dict."""
        r = query_agent(csr_client, "What is the status of claim CLM-123456?")
        assert isinstance(r["tool_results"], dict)

    def test_response_is_nonempty_string(self, csr_client):
        """The natural-language response must be a non-empty string."""
        r = query_agent(csr_client, "Check eligibility for member M-12345")
        assert isinstance(r["response"], str)
        assert len(r["response"]) > 10, "Response is suspiciously short"
