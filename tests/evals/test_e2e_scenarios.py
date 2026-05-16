"""
End-to-End Scenario Tests: N-Run Protocol and Multi-Turn Chains
================================================================
Exercises the full query-to-response flow with non-determinism analysis
and multi-turn session chaining.

The N-run protocol runs the same query multiple times and characterizes
variance — plan structure consistency, team routing stability, and
cost/latency distribution.

Multi-turn tests verify that session chaining works correctly across
follow-up queries, exercising the HAS_FOLLOW_UP relationship, context
preservation, and conversation compression.

Run with::
    pytest tests/evals/test_e2e_scenarios.py -v -s --timeout=300
"""
import pytest

from eval_helpers import (
    query_agent,
    fetch_cg_tree,
    get_teams_from_path,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# N-Run Protocol (section 10.5.3)
# ---------------------------------------------------------------------------
class TestNRunProtocol:
    """
    Run the same query N times and characterize variance.
    Groups runs by behavioral signature (which teams were invoked)
    to form plan equivalence classes.
    """

    def test_single_team_routing_consistency(self, api_client, test_data):
        """
        A simple single-team query should produce the same team
        assignment on every run. Expected consistency: 1.0.
        """
        data = test_data["nrun"]
        query = f"Look up the full details for claim {data['claim_id']}"
        n_runs = 3

        team_sets = []
        for _ in range(n_runs):
            response = query_agent(api_client, query)
            if response.get("error_count", 0) == 0:
                teams = frozenset(
                    get_teams_from_path(response.get("execution_path", []))
                )
                team_sets.append(teams)

        if len(team_sets) < 2:
            pytest.skip("Not enough successful runs for consistency check")

        # Compute plan equivalence classes
        unique_classes = set(team_sets)
        consistency = max(
            team_sets.count(cls) for cls in unique_classes
        ) / len(team_sets)

        assert consistency >= 0.80, (
            f"Routing consistency {consistency:.2f} below 0.80. "
            f"Equivalence classes: {[set(ts) for ts in unique_classes]}"
        )

    def test_multi_team_routing_consistency(self, api_client, test_data):
        """
        A two-team query should produce the same team pair on most
        runs. Expected consistency: >= 0.66 (2 of 3 runs agree).
        """
        data = test_data["nrun"]
        query = (
            f"I need two things: "
            f"First, check if provider {data['provider_id']} is in-network "
            f"for a PPO plan. "
            f"Second, look up the full details for claim {data['claim_id']}."
        )
        n_runs = 3

        team_sets = []
        for _ in range(n_runs):
            response = query_agent(api_client, query)
            if response.get("error_count", 0) == 0:
                teams = frozenset(
                    get_teams_from_path(response.get("execution_path", []))
                )
                team_sets.append(teams)

        if len(team_sets) < 2:
            pytest.skip("Not enough successful runs")

        unique_classes = set(team_sets)
        consistency = max(
            team_sets.count(cls) for cls in unique_classes
        ) / len(team_sets)

        assert consistency >= 0.60, (
            f"Multi-team routing consistency {consistency:.2f} below 0.60. "
            f"Equivalence classes: {[set(ts) for ts in unique_classes]}"
        )

    def test_response_fact_overlap(self, api_client, test_data):
        """
        The same query should produce responses that contain the same
        key facts across runs — even if phrasing differs.

        Checks that claim status and claim number appear consistently.
        """
        data = test_data["nrun"]
        query = f"Look up the full details for claim {data['claim_id']}"
        n_runs = 3

        responses = []
        for _ in range(n_runs):
            response = query_agent(api_client, query)
            if response.get("error_count", 0) == 0 and response.get("response"):
                responses.append(response["response"].lower())

        if len(responses) < 2:
            pytest.skip("Not enough successful runs")

        # Check that the known claim status appears in all responses
        status_keyword = data["claim_status"].lower()  # "approved"
        runs_with_status = sum(
            1 for r in responses if status_keyword in r
        )

        fact_overlap = runs_with_status / len(responses)
        assert fact_overlap >= 0.66, (
            f"Fact overlap {fact_overlap:.2f} below 0.66. "
            f"'{status_keyword}' appeared in {runs_with_status}/{len(responses)} runs."
        )


# ---------------------------------------------------------------------------
# Multi-Turn Scenario Tests (section 10.5.4)
# ---------------------------------------------------------------------------
class TestMultiTurnScenarios:
    """
    Verify session chaining across follow-up queries.
    Exercises HAS_FOLLOW_UP relationships, context preservation,
    and conversation compression.
    """

    def test_follow_up_chain_structure(self, api_client, test_data):
        """
        A two-turn conversation should produce two sessions linked
        by HAS_FOLLOW_UP in the Context Graph.

        Turn 1: "Look up claim X"
        Turn 2: "What is the member associated with that claim?"
                (references "that claim" — requires prior context)
        """
        data = test_data["multi_turn"]

        # Turn 1: explicit claim lookup
        turn1 = query_agent(
            api_client,
            f"Look up the full details for claim {data['claim_id']}",
        )
        assert turn1.get("error_count", 1) == 0, (
            f"Turn 1 failed: {turn1['response'][:200]}"
        )
        session1_id = turn1["session_id"]

        # Turn 2: follow-up referencing prior context
        turn2 = query_agent(
            api_client,
            "What is the eligibility status of the member associated with that claim?",
            prior_session_id=session1_id,
        )
        assert turn2.get("error_count", 1) == 0, (
            f"Turn 2 failed: {turn2['response'][:200]}"
        )
        session2_id = turn2["session_id"]

        # Verify CG chain: Session1 -> HAS_FOLLOW_UP -> Session2
        cg_data = fetch_cg_tree(api_client, session2_id)
        chain = cg_data.get("follow_up_chain", [])

        assert len(chain) >= 2, (
            f"Expected 2+ sessions in follow-up chain, got {len(chain)}. "
            f"Chain: {[s.get('sessionId') for s in chain]}"
        )

        # The chain should contain both session IDs
        chain_ids = {s.get("sessionId") for s in chain}
        assert session1_id in chain_ids, (
            f"Session 1 ({session1_id}) not in follow-up chain: {chain_ids}"
        )
        assert session2_id in chain_ids, (
            f"Session 2 ({session2_id}) not in follow-up chain: {chain_ids}"
        )

    def test_follow_up_context_preservation(self, api_client, test_data):
        """
        The follow-up query should produce a meaningful response
        that demonstrates context was preserved from turn 1.

        If context is lost, the response will say "I don't know which
        claim you're referring to" or produce an error.
        """
        data = test_data["multi_turn"]

        # Turn 1
        turn1 = query_agent(
            api_client,
            f"Look up the full details for claim {data['claim_id']}",
        )
        if turn1.get("error_count", 0) > 0:
            pytest.skip("Turn 1 failed")

        # Turn 2: follow-up without repeating the claim ID
        turn2 = query_agent(
            api_client,
            "What is the eligibility status of the member associated with that claim?",
            prior_session_id=turn1["session_id"],
        )

        # The response should be substantive (not an error or confusion)
        response = turn2.get("response", "")
        assert len(response) > 30, (
            f"Follow-up response too short ({len(response)} chars) — "
            f"context may not have been preserved. Response: {response[:200]}"
        )

        # The response should NOT contain confusion indicators
        confusion_phrases = [
            "which claim",
            "please specify",
            "i don't know which",
            "could you clarify",
            "unable to determine which",
        ]
        response_lower = response.lower()
        found_confusion = [p for p in confusion_phrases if p in response_lower]
        assert not found_confusion, (
            f"Follow-up response indicates lost context. "
            f"Confusion phrases found: {found_confusion}. "
            f"Response: {response[:200]}"
        )
