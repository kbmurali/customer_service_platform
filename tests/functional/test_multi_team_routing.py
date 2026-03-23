"""
Multi-team routing tests — verifies that queries requiring multiple
teams are correctly decomposed into multi-goal plans and dispatched
to the right teams.

Run with::

    pytest tests/functional/test_multi_team_routing.py -v
"""

import pytest
from conftest import query_agent, assert_no_errors, get_teams_from_path

pytestmark = pytest.mark.integration


class TestTwoTeamQueries:
    """Queries that should decompose into exactly two team delegations."""

    def test_provider_and_claims(self, csr_client):
        """Network status + claim denial should route to provider + claims."""
        r = query_agent(
            csr_client,
            "Is Dr. Chen in-network for member M-12345, and why was claim CLM-789012 denied?"
        )
        assert_no_errors(r, "provider + claims")
        teams = get_teams_from_path(r["execution_path"])
        assert "provider_services_team" in teams, f"Missing provider team: {teams}"
        assert "claims_services_team" in teams, f"Missing claims team: {teams}"

    def test_pa_and_search(self, csr_client):
        """PA check + code lookup should route to PA + search."""
        r = query_agent(
            csr_client,
            "Does CPT 29881 need prior auth under PPO, and what is the "
            "procedure code for knee replacement surgery?"
        )
        assert_no_errors(r, "PA + search")
        teams = get_teams_from_path(r["execution_path"])
        assert "pa_services_team" in teams, f"Missing PA team: {teams}"
        assert "search_services_team" in teams, f"Missing search team: {teams}"

    def test_member_and_claims(self, csr_client):
        """Eligibility + claim status should route to member + claims."""
        r = query_agent(
            csr_client,
            "Check eligibility for member M-67890 and get the status of "
            "their claim CLM-456789"
        )
        assert_no_errors(r, "member + claims")
        teams = get_teams_from_path(r["execution_path"])
        assert "member_services_team" in teams, f"Missing member team: {teams}"
        assert "claims_services_team" in teams, f"Missing claims team: {teams}"


class TestSequentialDependency:
    """Queries where step ordering matters (member context needed first)."""

    def test_member_before_claims(self, csr_client):
        """Member eligibility should be resolved before claim lookup
        when both are requested together."""
        r = query_agent(
            csr_client,
            "Check the health insurance eligibility for member 27b71fd8-49b7-46dd-84e3-5ad05d0a5db7. "
            "Also, what is the payment status of claim 7799c06c-0883-4dca-b1f0-bded6d1027a5?"
        )
        assert_no_errors(r, "sequential member→claims")
        teams = get_teams_from_path(r["execution_path"])
        assert "member_services_team" in teams, f"Missing member team: {teams}"
        assert "claims_services_team" in teams, f"Missing claims team: {teams}"

        # Verify member team appears before claims in execution path
        path = r["execution_path"]
        member_idx = next(
            (i for i, n in enumerate(path) if "member" in n), -1
        )
        claims_idx = next(
            (i for i, n in enumerate(path) if "claims" in n), -1
        )
        if member_idx >= 0 and claims_idx >= 0:
            assert member_idx < claims_idx, (
                f"Member should route before claims. "
                f"member at {member_idx}, claims at {claims_idx}"
            )


class TestGoalDecomposition:
    """Verify that multi-part queries produce appropriate goal counts."""

    def test_two_part_query_produces_multiple_path_nodes(self, csr_client):
        """A two-part query should produce more execution path nodes
        than a single-part query."""
        single = query_agent(csr_client, "What is the status of claim CLM-123456?")
        multi = query_agent(
            csr_client,
            "Is Dr. Chen in-network, and why was claim CLM-789012 denied?"
        )
        assert len(multi["execution_path"]) > len(single["execution_path"]), (
            f"Multi-team path ({len(multi['execution_path'])}) should be "
            f"longer than single-team ({len(single['execution_path'])})"
        )

    def test_multi_team_produces_multiple_tool_results(self, csr_client):
        """A multi-team query should produce tool results from
        multiple workers."""
        r = query_agent(
            csr_client,
            "Is provider 1f4f7e66-2db0-4a2b-8e39-0c2e4e93b6eb in-network "
            "for policy 698289fe-64b2-4382-894f-d8ad5ca4a4a4? "
            "Also, what is the payment status for claim 7799c06c-0883-4dca-b1f0-bded6d1027a5?"
        )
        assert_no_errors(r, "multi tool results")
        assert len(r["tool_results"]) >= 2, (
            f"Expected >=2 tool result keys, got {len(r['tool_results'])}: "
            f"{list(r['tool_results'].keys())}"
        )


class TestCrossEntityRouting:
    """Rule 11: route by noun requested, not identifier type.

    "claims for member X" → claims_services_team (member_claims tool)
    "PAs for member X"    → pa_services_team (member_prior_authorizations)
    "member details for X" → member_services_team
    """

    def test_claims_for_member_routes_to_claims(self, csr_client):
        """'claims for member X' should route to claims_services_team,
        NOT member_services_team, despite containing a member ID."""
        r = query_agent(
            csr_client,
            "Find the claims associated with member 2c0d65ea-29b8-4261-89e6-85bbb995200c"
        )
        assert_no_errors(r, "claims-for-member routing")
        teams = get_teams_from_path(r["execution_path"])
        assert "claims_services_team" in teams, (
            f"Expected claims_services_team for claims-by-member query, "
            f"got teams: {teams}"
        )

    def test_prior_auths_for_member_routes_to_pa(self, csr_client):
        """'PAs for member X' should route to pa_services_team,
        NOT member_services_team."""
        r = query_agent(
            csr_client,
            "Show me the prior authorizations for member 2c0d65ea-29b8-4261-89e6-85bbb995200c"
        )
        assert_no_errors(r, "PAs-for-member routing")
        teams = get_teams_from_path(r["execution_path"])
        assert "pa_services_team" in teams, (
            f"Expected pa_services_team for PAs-by-member query, "
            f"got teams: {teams}"
        )

    def test_member_details_still_routes_to_member(self, csr_client):
        """'member details for X' should still route to member_services_team.
        Rule 11 only redirects when the noun is claims/PAs, not member info."""
        r = query_agent(
            csr_client,
            "Get the member details for member 2c0d65ea-29b8-4261-89e6-85bbb995200c"
        )
        assert_no_errors(r, "member details routing")
        teams = get_teams_from_path(r["execution_path"])
        assert "member_services_team" in teams, (
            f"Expected member_services_team for member-details query, "
            f"got teams: {teams}"
        )
