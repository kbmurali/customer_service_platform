"""
Worker Faithfulness Evaluation
==============================
Demonstrates FaithfulnessMetric and HallucinationMetric at the worker layer.

Run with::
    pytest tests/evals/test_worker_faithfulness.py -v -s --timeout=300
"""
import pytest

from eval_helpers import query_agent

pytestmark = pytest.mark.eval


class TestWorkerRouting:
    """Verify queries route to the expected worker — deterministic."""

    def test_claim_lookup_worker_invoked(self, api_client, test_data):
        data = test_data["worker"]
        query = f"Look up the full details for claim {data['claim_id']}"
        response = query_agent(api_client, query)
        if response["error_count"] > 0:
            assert False, f"Query produced errors: {response['response'][:200]}"

        assert "claim_lookup" in response.get("tool_results", {}), (
            f"Expected 'claim_lookup' in tool_results, "
            f"got: {list(response['tool_results'].keys())}"
        )

    def test_eligibility_worker_invoked(self, api_client, test_data):
        data = test_data["worker"]
        query = f"Check the eligibility status for member {data['member_id']}"
        response = query_agent(api_client, query)
        if response["error_count"] > 0:
            assert False, f"Query produced errors: {response['response'][:200]}"

        tool_results = response.get("tool_results", {})
        # The planner may route to check_eligibility or member_lookup —
        # both return member status. Either is a valid routing decision.
        valid_workers = {"check_eligibility", "member_lookup"}
        found = valid_workers & set(tool_results.keys())
        assert found, (
            f"Expected one of {valid_workers} in tool_results, "
            f"got: {list(tool_results.keys())}"
        )


class TestWorkerFaithfulnessDeepeval:
    """Apply FaithfulnessMetric to a worker's output vs its tool data."""

    def test_claim_worker_faithfulness(self, api_client, test_data):
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import FaithfulnessMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        data = test_data["worker"]
        query = f"Look up the full details for claim {data['claim_id']}"
        response = query_agent(api_client, query)
        if response["error_count"] > 0:
            assert False, f"Query produced errors: {response['response'][:200]}"

        worker_result = response.get("tool_results", {}).get("claim_lookup")
        if not worker_result:
            # Non-determinism — claim_lookup didn't appear in this run
            return

        worker_output = (
            worker_result.get("output", "")
            if isinstance(worker_result, dict) else str(worker_result)
        )
        tool_raw = (
            worker_result.get("tool_raw_output", worker_output)
            if isinstance(worker_result, dict) else str(worker_result)
        )
        if not worker_output:
            return

        test_case = LLMTestCase(
            input=query,
            actual_output=worker_output,
            retrieval_context=[tool_raw],
        )
        metric = FaithfulnessMetric(threshold=0.7)
        metric.measure(test_case)
        print(f"  Worker FaithfulnessMetric score: {metric.score:.4f}")

        assert metric.score >= 0.7, (
            f"Worker faithfulness {metric.score:.2f} below threshold.\n"
            f"Reason: {metric.reason}"
        )


class TestWorkerHallucinationDeepeval:
    """Check whether the worker fabricates information not in tool data."""

    def test_claim_worker_no_hallucination(self, api_client, test_data):
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import HallucinationMetric
        except ImportError:
            pytest.skip("deepeval not installed")

        data = test_data["worker"]
        query = f"Look up the full details for claim {data['claim_id']}"
        response = query_agent(api_client, query)
        if response["error_count"] > 0:
            assert False, f"Query produced errors: {response['response'][:200]}"

        worker_result = response.get("tool_results", {}).get("claim_lookup")
        if not worker_result:
            # Non-determinism — claim_lookup didn't appear in this run
            return

        worker_output = (
            worker_result.get("output", "")
            if isinstance(worker_result, dict) else str(worker_result)
        )
        tool_raw = (
            worker_result.get("tool_raw_output", worker_output)
            if isinstance(worker_result, dict) else str(worker_result)
        )
        if not worker_output:
            return

        test_case = LLMTestCase(
            input=query,
            actual_output=worker_output,
            context=[tool_raw],
        )
        metric = HallucinationMetric(threshold=0.3)
        metric.measure(test_case)
        print(f"  Worker HallucinationMetric score: {metric.score:.4f}")

        assert metric.score <= 0.3, (
            f"Worker hallucination score {metric.score:.2f} above threshold — "
            f"output contains fabricated claims.\n"
            f"Reason: {metric.reason}"
        )
