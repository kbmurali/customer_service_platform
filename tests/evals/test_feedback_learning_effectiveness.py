"""
Feedback Learning Effectiveness Evaluation
===========================================
Measures baseline planning quality and evaluates whether the
experience store improves planning.

Run with::

    pytest tests/evals/test_feedback_learning_effectiveness.py -v -s --timeout=300
"""

import json
import uuid
import pytest
from eval_helpers import (
    query_agent,
    get_teams_from_path,
    extract_tool_results_text,
    seed_experience,
    remove_experience,
)

pytestmark = pytest.mark.eval


class TestBaselinePlanningQuality:
    """Measures planning accuracy — queries should route to expected teams."""

    def test_single_team_routing(self, api_client, test_data):
        """A claim lookup query routes to claims_services_team."""
        data = test_data["feedback_single"]
        query = f"Look up the full details for claim {data['claim_id']}"
        response = query_agent(api_client, query)
        assert response.get("error_count", 1) == 0
        teams = get_teams_from_path(response.get("execution_path", []))
        assert "claims_services_team" in teams, (
            f"Expected claims_services_team, got: {teams}"
        )

    def test_multi_team_routing(self, api_client, test_data):
        """A two-part query routes to both provider and claims teams."""
        data = test_data["feedback_multi"]
        query = (
            f"I need two things: "
            f"First, check if provider {data['provider_id']} is in-network for a PPO plan. "
            f"Second, look up the full details for claim {data['claim_id']}."
        )
        response = query_agent(api_client, query)
        assert response.get("error_count", 1) == 0
        teams = get_teams_from_path(response.get("execution_path", []))
        assert "provider_services_team" in teams
        assert "claims_services_team" in teams

    def test_decision_agent_routing(self, api_client, test_data):
        """A decision query routes to claims for adjudication."""
        data = test_data["feedback_decision"]
        query = (
            f"Should claim {data['claim_id']} be approved or denied? "
            f"Check member {data['member_id']} eligibility first."
        )
        response = query_agent(api_client, query)
        assert response.get("error_count", 1) == 0
        teams = get_teams_from_path(response.get("execution_path", []))
        assert "claims_services_team" in teams


class TestResponseFaithfulnessDeepeval:
    """Apply FaithfulnessMetric to the final response."""

    def test_single_team_faithfulness(self, api_client, test_data):
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        data = test_data["feedback_single"]
        query = f"Look up the full details for claim {data['claim_id']}"
        response = query_agent(api_client, query)

        if response["error_count"] > 0:
            pytest.skip("Query produced errors")

        tool_context = extract_tool_results_text(
            response.get("tool_results", {})
        )
        if not tool_context:
            pytest.skip("No tool results")

        test_case = LLMTestCase(
            input=query,
            actual_output=response["response"],
            retrieval_context=[tool_context],
        )
        metric = FaithfulnessMetric(threshold=0.7)
        metric.measure(test_case)
        print(f"  FaithfulnessMetric score: {metric.score:.4f}")
        assert metric.score >= 0.5, (
            f"Faithfulness {metric.score:.2f} below threshold.\n"
            f"Reason: {metric.reason}"
        )


class TestExperienceStoreEffectiveness:
    """Seed an experience and verify the planner uses it."""

    def test_seeded_experience_influences_planning(self, api_client, test_data):
        test_session_id = f"eval-seed-{uuid.uuid4().hex[:8]}"
        test_plan = json.dumps({
            "goals": [{"id": "g1", "description": "Check claim status",
                        "priority": 1}],
            "steps": [{"step_id": "s1", "goal_id": "g1",
                        "agent": "claims_services_team", "order": 1}],
        })
        try:
            success = seed_experience(
                client=api_client,
                session_id=test_session_id,
                query_text="What is the current processing status of my insurance claim?",
                plan_json=test_plan,
                team_assignments="claims_services_team",
            )
            if not success:
                pytest.skip("Failed to seed experience")

            response = query_agent(
                api_client,
                "Can you check the processing status of my claim?",
            )
            assert response.get("error_count", 1) == 0
            teams = get_teams_from_path(
                response.get("execution_path", [])
            )
            assert "claims_services_team" in teams
        finally:
            remove_experience(test_session_id)


class TestPlanningConsistency:
    """Run the same query multiple times and verify consistent routing."""

    def test_routing_consistency_across_runs(self, api_client, test_data):
        data = test_data["feedback_consistency"]
        query = f"Look up the full details for claim {data['claim_id']}"
        team_sets = []
        for _ in range(3):
            response = query_agent(api_client, query)
            if response.get("error_count", 0) == 0:
                teams = frozenset(
                    get_teams_from_path(response.get("execution_path", []))
                )
                team_sets.append(teams)

        if len(team_sets) < 2:
            pytest.skip("Not enough successful runs")

        unique = set(team_sets)
        assert len(unique) <= 2, (
            f"Inconsistent routing: {[set(ts) for ts in unique]}"
        )
