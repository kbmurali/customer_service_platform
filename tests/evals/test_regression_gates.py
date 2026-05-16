"""
Regression Gate Tests
=====================
Enforces metric baselines from config/test_baselines.yaml against
live metric values from the /api/stats/agentic-health endpoint.

Each metric is compared against its stored baseline. Metrics where
higher is better (accuracy, success rate) must meet or exceed the
baseline. Metrics where lower is better (latency, error rate, token
count) must not exceed the baseline.

Run with::
    pytest tests/evals/test_regression_gates.py -v -s --timeout=60

Requires:
    - A running CSIP Docker stack with the evaluation pipeline active
    - config/test_baselines.yaml with threshold values
"""
import os
import pytest
import yaml

from eval_helpers import API_BASE

pytestmark = pytest.mark.eval

# Metrics where HIGHER is better — current value must be >= baseline
HIGHER_IS_BETTER = {
    "planning_routing_accuracy",
    "positive_feedback_rate",
    "tool_success_rate",
}

# Metrics where LOWER is better — current value must be <= baseline
LOWER_IS_BETTER = {
    "agent_error_rate",
    "avg_agent_latency_seconds",
    "avg_e2e_latency_seconds",
    "estimated_tokens_per_query",
    "llm_calls_per_query",
    "avg_plan_goals",
    "avg_plan_steps",
}


def _load_baselines() -> dict:
    """Load baseline thresholds from config/test_baselines.yaml."""
    # Search for the config file relative to common locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "config", "test_baselines.yaml"),
        os.path.join(os.getcwd(), "tests", "evals", "config", "test_baselines.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)

    pytest.skip(
        "config/test_baselines.yaml not found. "
        "Create it with baseline thresholds to enable regression gates."
    )
    return {}


def _fetch_agentic_health(api_client) -> dict:
    """Fetch current metric values from /api/stats/agentic-health."""
    response = api_client.get("/api/stats/agentic-health")
    if response.status_code == 403:
        pytest.skip(
            "agentic-health endpoint requires CSR_TIER2 or CSR_SUPERVISOR role"
        )
    assert response.status_code == 200, (
        f"agentic-health returned {response.status_code}: {response.text[:200]}"
    )
    return response.json().get("metrics", {})


class TestRegressionGates:
    """
    Compare live metric values against stored baselines.
    Fails the test if any metric crosses its threshold.
    """

    def test_quality_metrics_meet_baselines(self, api_client, test_data):
        """
        Metrics where higher is better (accuracy, success rate,
        feedback rate) must meet or exceed their baselines.
        """
        baselines = _load_baselines()
        metrics = _fetch_agentic_health(api_client)

        if not metrics or all(v == 0 for v in metrics.values()):
            pytest.skip(
                "All metrics are zero — evaluation pipeline may not have "
                "completed a cycle yet. Run after processing some queries."
            )

        violations = []
        for metric_name in HIGHER_IS_BETTER:
            baseline = baselines.get(metric_name)
            if baseline is None:
                continue
            current = metrics.get(metric_name, 0)
            if current < baseline:
                violations.append(
                    f"  {metric_name}: {current:.4f} < baseline {baseline}"
                )

        assert not violations, (
            "Quality metrics below baseline:\n" + "\n".join(violations)
        )

    def test_cost_and_latency_within_bounds(self, api_client, test_data):
        """
        Metrics where lower is better (latency, error rate, token
        count) must not exceed their baselines.
        """
        baselines = _load_baselines()
        metrics = _fetch_agentic_health(api_client)

        if not metrics or all(v == 0 for v in metrics.values()):
            pytest.skip(
                "All metrics are zero — evaluation pipeline may not have "
                "completed a cycle yet."
            )

        violations = []
        for metric_name in LOWER_IS_BETTER:
            baseline = baselines.get(metric_name)
            if baseline is None:
                continue
            current = metrics.get(metric_name, 0)
            if current > baseline:
                violations.append(
                    f"  {metric_name}: {current:.4f} > baseline {baseline}"
                )

        assert not violations, (
            "Cost/latency metrics above baseline:\n" + "\n".join(violations)
        )
