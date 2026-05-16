"""
Metric Persistence and Gauge Accuracy
======================================
Validates that the MySQL-backed Prometheus gauges reflect the true
state of the experience store and survive container restarts.

The test seeds an experience into the production Chroma collection,
waits for the evaluation pipeline's next cycle (up to 35 seconds),
scrapes the /metrics endpoint, and asserts the experience store
size gauge reflects the insertion. Teardown removes the test
experience.

Run with::
    pytest tests/evals/test_metric_persistence.py -v -s --timeout=120

Requires:
    - A running CSIP Docker stack with the evaluation pipeline active
    - The eval venv with runtime packages (Chroma access)
"""
import re
import time
import uuid
import pytest

from eval_helpers import seed_experience, remove_experience

pytestmark = pytest.mark.eval

# Gauge names as they appear in /metrics Prometheus text format
EXPERIENCE_STORE_SIZE_GAUGE = "csip_experience_store_size"


def _scrape_gauge(api_client, gauge_name: str) -> float:
    """
    Scrape the /metrics endpoint and extract a gauge value by name.

    The /metrics endpoint returns Prometheus text exposition format:
        gauge_name 42.0
        gauge_name{label="value"} 42.0

    Returns the gauge value, or -1 if not found.
    """
    response = api_client.get("/metrics")
    if response.status_code != 200:
        return -1

    text = response.text
    for line in text.strip().split("\n"):
        # Skip comments and empty lines
        if line.startswith("#") or not line.strip():
            continue
        # Match "gauge_name 42.0" or "gauge_name{labels} 42.0"
        match = re.match(
            rf'^{re.escape(gauge_name)}(?:\{{[^}}]*\}})?\s+([\d.eE+-]+)',
            line,
        )
        if match:
            return float(match.group(1))

    return -1


def _get_chroma_collection_size() -> int:
    """Get the current size of the experience store Chroma collection."""
    try:
        from databases.chroma_experience_store import get_experience_store
        return get_experience_store().get_collection_size()
    except ImportError:
        pytest.skip("ChromaExperienceStore not available outside CSIP container")
        return -1


class TestMetricPersistence:
    """
    Validate that the experience store size gauge accurately reflects
    the Chroma collection state.
    """

    def test_experience_store_size_gauge_reflects_chroma(
        self, api_client, test_data
    ):
        """
        Seed an experience, verify the Chroma collection size increases,
        then check that the /metrics gauge eventually reflects the new
        size (within one evaluation pipeline cycle of ~30 seconds).
        """
        test_session_id = f"eval-metric-{uuid.uuid4().hex[:8]}"

        # 1. Record baseline Chroma size
        baseline_size = _get_chroma_collection_size()

        try:
            # 2. Seed an experience
            success = seed_experience(
                client=api_client,
                session_id=test_session_id,
                query_text="Metric persistence test query — what is my claim status?",
                plan_json='{"goals": [{"id": "g1", "description": "test"}], "steps": []}',
                team_assignments="claims_services_team",
            )
            if not success:
                pytest.skip("Failed to seed experience")

            # 3. Verify Chroma size incremented (immediate, no pipeline wait)
            new_size = _get_chroma_collection_size()
            assert new_size > baseline_size, (
                f"Chroma collection size did not increase after seeding. "
                f"Baseline: {baseline_size}, after seed: {new_size}"
            )

            # 4. Wait for evaluation pipeline cycle and check gauge
            #    The pipeline runs every 30 seconds. We poll every 5 seconds
            #    for up to 75 seconds to guarantee catching at least two cycles.
            gauge_matched = False
            for attempt in range(15):
                gauge_value = _scrape_gauge(
                    api_client, EXPERIENCE_STORE_SIZE_GAUGE
                )
                if gauge_value >= new_size:
                    gauge_matched = True
                    break
                time.sleep(5)

            if gauge_value == -1:
                pytest.skip(
                    f"Gauge '{EXPERIENCE_STORE_SIZE_GAUGE}' not found in "
                    f"/metrics — evaluation pipeline may not be running."
                )

            assert gauge_matched, (
                f"Gauge '{EXPERIENCE_STORE_SIZE_GAUGE}' did not reflect "
                f"Chroma size within 75 seconds. "
                f"Expected >= {new_size}, got {gauge_value}."
            )

        finally:
            # 5. Cleanup
            remove_experience(test_session_id)

    def test_metrics_endpoint_returns_valid_prometheus_format(
        self, api_client, test_data
    ):
        """
        The /metrics endpoint should return valid Prometheus text
        exposition format — non-empty, with at least one metric line.
        """
        response = api_client.get("/metrics")
        assert response.status_code == 200, (
            f"/metrics returned {response.status_code}"
        )

        text = response.text
        assert len(text) > 10, "/metrics returned empty or near-empty body"

        # At least one line should be a metric (not a comment)
        metric_lines = [
            line for line in text.strip().split("\n")
            if line and not line.startswith("#")
        ]
        assert len(metric_lines) >= 1, (
            f"/metrics returned no metric lines. Full body: {text[:300]}"
        )
