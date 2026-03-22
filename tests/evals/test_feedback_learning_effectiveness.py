"""
Feedback Learning Effectiveness Evaluation — measures whether the
experience store improves planning quality.

Uses DeepEval to compare baseline (empty store) vs augmented (populated
store) performance on a golden dataset of CSIP queries.

Run with::

    pytest tests/eval/test_feedback_learning_effectiveness.py -v -s

Requires:
    - A running CSIP Docker stack (API, Neo4j, Chroma, MySQL)
    - ``pip install deepeval`` (evaluation framework)
    - ``pip install httpx`` (async HTTP client)
"""

import json
import os
import pytest
import httpx
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("CSIP_API_BASE", "https://localhost/agentic/access")
API_USER = os.getenv("CSIP_TEST_USER", "csr2")
API_PASS = os.getenv("CSIP_TEST_PASS", "testuser")
VERIFY_SSL = False  # Self-signed cert in dev


# ---------------------------------------------------------------------------
# Golden Dataset — queries with expected planning structure
# ---------------------------------------------------------------------------

GOLDEN_DATASET = [
    {
        "query": "What is the status of claim CLM-123456?",
        "expected_teams": ["claims_services_team"],
        "expected_goal_count": 1,
        "expected_step_count": 1,
        "category": "single_team_read",
    },
    {
        "query": "Does CPT 29881 require prior authorization under a PPO plan?",
        "expected_teams": ["pa_services_team"],
        "expected_goal_count": 1,
        "expected_step_count": 1,
        "category": "single_team_read",
    },
    {
        "query": "Is Dr. Chen in-network for member M-12345, and why was claim CLM-789012 denied?",
        "expected_teams": ["provider_services_team", "claims_services_team"],
        "expected_goal_count": 2,
        "expected_step_count": 2,
        "category": "multi_team_read",
    },
    {
        "query": "Check eligibility for member M-67890 and get their claim CLM-456 payment info",
        "expected_teams": ["member_services_team", "claims_services_team"],
        "expected_goal_count": 2,
        "expected_step_count": 2,
        "category": "multi_team_sequential",
    },
    {
        "query": "What is the ICD-10 code for knee osteoarthritis?",
        "expected_teams": ["search_services_team"],
        "expected_goal_count": 1,
        "expected_step_count": 1,
        "category": "search_only",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def auth_token() -> str:
    """Authenticate once for all tests."""
    response = httpx.post(
        f"{API_BASE}/api/auth/login",
        json={"username": API_USER, "password": API_PASS},
        verify=VERIFY_SSL,
    )
    assert response.status_code == 200, f"Auth failed: {response.text}"
    return response.json()["access_token"]


@pytest.fixture(scope="session")
def api_client(auth_token) -> httpx.Client:
    """Authenticated HTTP client."""
    return httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {auth_token}"},
        verify=VERIFY_SSL,
        timeout=60.0,
    )


def query_agent(client: httpx.Client, query: str) -> Dict[str, Any]:
    """Send a query to the CSIP agent and return the structured response."""
    response = client.post(
        "/api/agent/query",
        json={"query": query},
    )
    assert response.status_code == 200, f"Query failed: {response.text}"
    return response.json()


# ---------------------------------------------------------------------------
# Planning Quality Metrics (custom — no DeepEval equivalent)
# ---------------------------------------------------------------------------

def evaluate_plan_quality(
    response: Dict[str, Any],
    expected_teams: List[str],
    expected_goal_count: int,
    expected_step_count: int,
) -> Dict[str, Any]:
    """
    Evaluate planning quality by comparing the actual execution path
    against expected team assignments and plan structure.

    Returns a dict of metric scores.
    """
    execution_path = response.get("execution_path", [])
    tool_results = response.get("tool_results", {})
    error_count = response.get("error_count", 0)

    # Team routing accuracy: did the expected teams appear in the path?
    teams_in_path = set()
    for node in execution_path:
        if node.startswith("a2a_"):
            # Extract team name: "a2a_claims_services_team" -> "claims_services_team"
            team = node[4:]
            teams_in_path.add(team)

    expected_set = set(expected_teams)
    team_recall = len(expected_set & teams_in_path) / max(len(expected_set), 1)
    team_precision = len(expected_set & teams_in_path) / max(len(teams_in_path), 1)

    # Goal count accuracy
    # Count create_plan occurrences as proxy for planning calls
    plan_calls = execution_path.count("create_plan")

    return {
        "team_recall": team_recall,
        "team_precision": team_precision,
        "error_count": error_count,
        "teams_routed": sorted(teams_in_path),
        "plan_calls": plan_calls,
        "has_response": bool(response.get("response", "")),
    }


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestBaselinePlanningQuality:
    """
    Baseline evaluation — measures planning quality with the current
    experience store state (may be empty or populated).

    These tests establish the baseline against which experience-augmented
    planning is compared.
    """

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c["query"][:50] for c in GOLDEN_DATASET])
    def test_planning_produces_valid_response(self, api_client, case):
        """Every query should produce a non-error response."""
        response = query_agent(api_client, case["query"])
        assert response.get("error_count", 1) == 0, (
            f"Query produced errors: {response}"
        )
        assert response.get("response", "") != "", (
            "Empty response from agent"
        )

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c["query"][:50] for c in GOLDEN_DATASET])
    def test_correct_team_routing(self, api_client, case):
        """Queries should route to the expected teams."""
        response = query_agent(api_client, case["query"])
        metrics = evaluate_plan_quality(
            response,
            case["expected_teams"],
            case["expected_goal_count"],
            case["expected_step_count"],
        )
        assert metrics["team_recall"] >= 0.5, (
            f"Expected teams {case['expected_teams']} but got {metrics['teams_routed']}"
        )


class TestDeepEvalAnswerQuality:
    """
    Answer quality evaluation using DeepEval metrics.

    These tests require ``pip install deepeval``.
    """

    @pytest.mark.parametrize("case", GOLDEN_DATASET[:3],
                             ids=[c["query"][:40] for c in GOLDEN_DATASET[:3]])
    def test_faithfulness(self, api_client, case):
        """Agent responses should be grounded in tool results."""
        try:
            from deepeval import evaluate as deepeval_evaluate
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        response = query_agent(api_client, case["query"])
        tool_results_str = json.dumps(response.get("tool_results", {}))

        test_case = LLMTestCase(
            input=case["query"],
            actual_output=response["response"],
            retrieval_context=[tool_results_str] if tool_results_str != "{}" else [],
        )

        metric = FaithfulnessMetric(threshold=0.7)
        metric.measure(test_case)

        assert metric.score >= 0.5, (
            f"Faithfulness score {metric.score} below threshold. "
            f"Reason: {metric.reason}"
        )

    @pytest.mark.parametrize("case", GOLDEN_DATASET[:3],
                             ids=[c["query"][:40] for c in GOLDEN_DATASET[:3]])
    def test_answer_relevancy(self, api_client, case):
        """Agent responses should be relevant to the question asked."""
        try:
            from deepeval import evaluate as deepeval_evaluate
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import AnswerRelevancyMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        response = query_agent(api_client, case["query"])

        test_case = LLMTestCase(
            input=case["query"],
            actual_output=response["response"],
        )

        metric = AnswerRelevancyMetric(threshold=0.7)
        metric.measure(test_case)

        assert metric.score >= 0.5, (
            f"Relevancy score {metric.score} below threshold. "
            f"Reason: {metric.reason}"
        )


class TestExperienceStoreEffectiveness:
    """
    Comparative evaluation — measures whether populating the experience
    store improves planning quality.

    This test class should be run in two phases:
    1. Run with an empty experience store → record baseline scores
    2. Populate the store (via seed_experience_collections.py or
       real feedback) → rerun → compare

    The comparison is manual (inspect pytest output) or automated via
    a CI script that captures scores from both runs.
    """

    def test_experience_store_does_not_degrade_quality(self, api_client):
        """
        Sanity check: with the experience store populated,
        basic queries should still produce valid responses.
        """
        for case in GOLDEN_DATASET:
            response = query_agent(api_client, case["query"])
            assert response.get("error_count", 1) == 0, (
                f"Query '{case['query'][:50]}' produced errors with "
                f"experience store active"
            )
            assert response.get("response", "") != "", (
                f"Empty response for '{case['query'][:50]}' with "
                f"experience store active"
            )
