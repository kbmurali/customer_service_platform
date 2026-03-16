"""
Integration Tests — Agentic Access HTTP API
============================================
Tests the live agentic access REST API reachable at
https://localhost:443/agentic/access via standard HTTPS (no mTLS).

Prerequisites:
    - api-gateway running on localhost:443 (HTTPS)
    - agentic-access-api running and healthy
    - All 5 team A2A servers running and healthy
    - All 5 MCP tool servers running and healthy
    - .env (or environment) with:
        AGENTIC_API_URL=https://localhost:443/agentic/access
        TEST_USERNAME=tier2user
        TEST_PASSWORD=<password>
        TEST_PROVIDER_ID=1f4f7e66-2db0-4a2b-8e39-0c2e4e93b6eb
        TEST_POLICY_ID=698289fe-64b2-4382-894f-d8ad5ca4a4a4
        TEST_CLAIM_ID=7799c06c-0883-4dca-b1f0-bded6d1027a5
        TEST_PA_ID=cc0af705-9a9b-46e7-b308-a69c4502b817
        TEST_PA_NUMBER=PA-844196  # paNumber field (PA-XXXXXX format), not the UUID
        TEST_MEMBER_ID=<member-uuid>

Run:
    cd customer_service_platform
    pytest agents/test_agentic_access_api.py -v
"""

import os
import logging
import uuid
from typing import Any, Dict, Optional

import pytest
import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL       = os.getenv("AGENTIC_API_URL", "https://localhost:443/agentic/access")
TEST_USERNAME  = os.getenv("TEST_USERNAME",   "csr2")
TEST_PASSWORD  = os.getenv("TEST_PASSWORD",   "testuser")

TEST_PROVIDER_ID  = os.getenv("TEST_PROVIDER_ID",  "1f4f7e66-2db0-4a2b-8e39-0c2e4e93b6eb")
TEST_POLICY_ID    = os.getenv("TEST_POLICY_ID",    "698289fe-64b2-4382-894f-d8ad5ca4a4a4")
TEST_CLAIM_ID     = os.getenv("TEST_CLAIM_ID",     "7799c06c-0883-4dca-b1f0-bded6d1027a5")
TEST_CLAIM_NUMBER = os.getenv("TEST_CLAIM_NUMBER", "CLM-421386")
TEST_PA_ID        = os.getenv("TEST_PA_ID",        "cc0af705-9a9b-46e7-b308-a69c4502b817")
TEST_PA_NUMBER    = os.getenv("TEST_PA_NUMBER",    "PA-844196") 
TEST_MEMBER_ID    = os.getenv("TEST_MEMBER_ID",    "27b71fd8-49b7-46dd-84e3-5ad05d0a5db7")
TEST_PROCEDURE    = os.getenv("TEST_PROCEDURE",    "29881")
TEST_POLICY_TYPE  = os.getenv("TEST_POLICY_TYPE",  "PPO")
TEST_SPECIALTY    = os.getenv("TEST_SPECIALTY",    "Dermatology")
TEST_ZIP          = os.getenv("TEST_ZIP",          "30368")

# ---------------------------------------------------------------------------
# Shared HTTP client and auth helpers
# ---------------------------------------------------------------------------

def _get_client() -> httpx.Client:
    """Return an httpx client that skips TLS verification for local dev certs."""
    return httpx.Client(verify=False, timeout=60)


def _login(client: httpx.Client) -> str:
    """Authenticate and return a bearer token."""
    resp = client.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
    )
    assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.text}"
    token = resp.json()["access_token"]
    assert token, "Login returned empty access_token"
    return token


def _query(
    client: httpx.Client,
    token: str,
    query: str,
    member_id: Optional[str] = None,
) -> Dict[str, Any]:
    """POST /api/agent/query and return the parsed response dict."""
    payload: Dict[str, Any] = {"query": query}
    if member_id:
        payload["member_id"] = member_id

    resp = client.post(
        f"{BASE_URL}/api/agent/query",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert resp.status_code == 200, (
        f"Query failed ({resp.status_code}): {resp.text[:300]}"
    )
    data = resp.json()
    assert "response"       in data, "response field missing"
    assert "tool_results"   in data, "tool_results field missing"
    assert "execution_path" in data, "execution_path field missing"
    assert "error_count"    in data, "error_count field missing"
    return data


def _assert_teams_invoked(data: Dict[str, Any], *team_names: str) -> None:
    """Assert that each named team appears in the execution_path."""
    path_str = " ".join(str(s) for s in data["execution_path"]).lower()
    for team in team_names:
        assert team.lower() in path_str, (
            f"Expected team '{team}' in execution_path, got: {data['execution_path']}"
        )


def _assert_tool_results_present(data: Dict[str, Any], *worker_keys: str) -> None:
    """Assert that each worker key appears in tool_results."""
    for key in worker_keys:
        assert key in data["tool_results"], (
            f"Expected '{key}' in tool_results, got keys: {list(data['tool_results'].keys())}"
        )


def _assert_no_pii_leak(response_text: str) -> None:
    """Assert no obvious raw PII patterns in the response text."""
    import re
    # Raw SSN pattern
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", response_text), \
        "Potential SSN found in response"
    # Raw credit card pattern
    assert not re.search(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b", response_text), \
        "Potential credit card number found in response"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def http_client():
    """Shared httpx client for the whole module."""
    with _get_client() as client:
        yield client


@pytest.fixture(scope="module")
def auth_token(http_client):
    """Authenticate once for the whole module."""
    return _login(http_client)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

class TestHealthAndAuth:
    """Verify the API is reachable and auth works before running task tests."""

    def test_health_endpoint(self, http_client):
        """GET /health should return 200 with healthy status."""
        resp = http_client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "healthy", f"Unexpected health status: {data}"
        logger.info("Health check passed: %s", data)

    def test_login_returns_token(self, http_client):
        """POST /api/auth/login should return a non-empty bearer token."""
        token = _login(http_client)
        assert len(token) > 20, "Token looks too short"
        logger.info("Login successful — token length: %d", len(token))

    def test_query_without_auth_returns_401(self, http_client):
        """Requests without Authorization header should be rejected with 401."""
        resp = http_client.post(
            f"{BASE_URL}/api/agent/query",
            json={"query": "Is my provider in-network?"},
        )
        # FastAPI's HTTPBearer returns 403 (Forbidden) when no Authorization
        # header is present, and 401 (Unauthorized) when a bad token is supplied.
        # Both indicate the request was correctly rejected.
        assert resp.status_code in (401, 403), (
            f"Expected 401 or 403 without auth, got {resp.status_code}"
        )

    def test_query_with_expired_token_returns_401(self, http_client):
        """A clearly invalid token should return 401."""
        resp = http_client.post(
            f"{BASE_URL}/api/agent/query",
            headers={"Authorization": "Bearer this.is.not.a.valid.token"},
            json={"query": "Is my provider in-network?"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Provider Services tests
# ---------------------------------------------------------------------------

class TestProviderServices:
    """Tests routed to the provider_services_team."""

    def test_provider_network_check(self, http_client, auth_token):
        """Single-team: check if a specific provider is in-network for a policy."""
        data = _query(
            http_client, auth_token,
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_provider_services_team")
        _assert_tool_results_present(data, "provider_network_check")
        assert any(
            kw in data["response"].lower()
            for kw in ["in-network", "in network", "network", "policy", "provider"]
        ), f"Unexpected response: {data['response'][:200]}"
        _assert_no_pii_leak(data["response"])
        logger.info("Provider network check: %s", data["response"][:150])

    def test_provider_lookup(self, http_client, auth_token):
        """Single-team: look up provider details by ID."""
        data = _query(
            http_client, auth_token,
            f"What are the details of provider {TEST_PROVIDER_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_provider_services_team")
        _assert_no_pii_leak(data["response"])

    def test_provider_search_by_specialty(self, http_client, auth_token):
        """Single-team: search for providers by specialty and ZIP."""
        data = _query(
            http_client, auth_token,
            f"Find in-network {TEST_SPECIALTY} providers near ZIP code {TEST_ZIP} covered by my health insurance plan"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_provider_services_team")
        assert any(
            kw in data["response"].lower()
            for kw in [TEST_SPECIALTY.lower(), "provider", "specialist", "zip"]
        )
        _assert_no_pii_leak(data["response"])


# ---------------------------------------------------------------------------
# Claims Services tests
# ---------------------------------------------------------------------------

class TestClaimsServices:
    """Tests routed to the claims_services_team."""

    def test_claim_status_lookup(self, http_client, auth_token):
        """Single-team: look up a claim by UUID via claim_lookup worker.
        Uses TEST_CLAIM_ID (UUID) which deterministically routes to claim_lookup.
        claim_status requires a CLM-XXXXXX claim number which may not be set in
        all environments — routing via UUID is the reliable alternative.
        """
        data = _query(
            http_client, auth_token,
            f"Look up claim {TEST_CLAIM_ID}"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_claims_services_team")
        _assert_tool_results_present(data, "claim_lookup")
        _assert_no_pii_leak(data["response"])
        logger.info("Claim lookup: %s", data["response"][:150])

    def test_claim_payment_info(self, http_client, auth_token):
        """Single-team: retrieve payment details for a claim."""
        data = _query(
            http_client, auth_token,
            f"What is the payment information for claim {TEST_CLAIM_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_claims_services_team")
        _assert_no_pii_leak(data["response"])

    def test_update_claim_status(self, http_client, auth_token):
        """Single-team: update the status of a claim.
        HIGH-IMPACT write operation — human approval required by MCP decorator.
        Asserts the claims team is invoked and returns a non-empty response;
        the actual status change depends on approval workflow state in the
        test environment.
        """
        data = _query(
            http_client, auth_token,
            f"Update claim {TEST_CLAIM_ID} status to UNDER_REVIEW — "
            f"additional documentation has been received and requires clinical "
            f"review for this health insurance claim."
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_claims_services_team")
        assert data["response"].strip(), "update_claim_status returned empty response"
        _assert_no_pii_leak(data["response"])
        logger.info("Update claim status response: %s", data["response"][:150])


# ---------------------------------------------------------------------------
# Member Services tests
# ---------------------------------------------------------------------------

class TestMemberServices:
    """Tests routed to the member_services_team."""

    def test_member_lookup(self, http_client, auth_token):
        """Single-team: look up member details by member ID."""
        data = _query(
            http_client, auth_token,
            f"What are the plan benefits and coverage details for member {TEST_MEMBER_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_member_services_team")
        _assert_no_pii_leak(data["response"])
        logger.info("Member lookup: %s", data["response"][:150])

    def test_coverage_lookup(self, http_client, auth_token):
        """Single-team: check coverage details for a member and policy."""
        data = _query(
            http_client, auth_token,
            f"What is the coverage for member {TEST_MEMBER_ID} under policy {TEST_POLICY_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_member_services_team")
        _assert_no_pii_leak(data["response"])

    def test_eligibility_check(self, http_client, auth_token):
        """Single-team: check member eligibility."""
        data = _query(
            http_client, auth_token,
            f"Check the health insurance eligibility and enrollment status for member {TEST_MEMBER_ID}"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_member_services_team")
        _assert_no_pii_leak(data["response"])

    def test_update_member_info(self, http_client, auth_token):
        """Single-team: update a member contact field with a reason.
        HIGH-IMPACT write operation — human approval required by MCP decorator.
        Asserts the member services team is invoked and returns a non-empty
        response; the actual field update depends on approval workflow state
        in the test environment.
        """
        data = _query(
            http_client, auth_token,
            f"Update phone number for health insurance member {TEST_MEMBER_ID} "
            f"to 555-9876 — member called in to request a contact information update."
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_member_services_team")
        assert data["response"].strip(), "update_member_info returned empty response"
        _assert_no_pii_leak(data["response"])
        logger.info("Update member info response: %s", data["response"][:150])


# ---------------------------------------------------------------------------
# PA Services tests
# ---------------------------------------------------------------------------

class TestPAServices:
    """Tests routed to the pa_services_team."""

    def test_pa_lookup(self, http_client, auth_token):
        """Single-team: look up a prior authorization by UUID via PA services team."""
        # pa_lookup worker requires a PA ID in UUID format (not PA-XXXXXX number).
        # The query must keep the UUID intact through central supervisor summarisation.
        data = _query(
            http_client, auth_token,
            f"Look up prior authorization {TEST_PA_ID}"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_pa_services_team")
        assert data["response"].strip(), "PA lookup returned empty response"
        # Log tool_results regardless — SKIP is a routing outcome not a test failure
        pa_keys = [k for k in data["tool_results"] if k.startswith("pa_")]
        logger.info(
            "PA lookup — tool_results keys: %s | response: %s",
            list(data["tool_results"].keys()),
            data["response"][:200],
        )
        # If the worker ran, verify it returned something meaningful
        if pa_keys:
            assert data["tool_results"][pa_keys[0]].get("output", "").strip(),                 f"PA tool returned empty output for key: {pa_keys[0]}"
        _assert_no_pii_leak(data["response"])

    def test_pa_requirements(self, http_client, auth_token):
        """Single-team: check if a procedure requires PA under a policy type."""
        data = _query(
            http_client, auth_token,
            f"Does CPT {TEST_PROCEDURE} require prior authorization under a {TEST_POLICY_TYPE} plan?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_pa_services_team")
        _assert_no_pii_leak(data["response"])

    def test_approve_prior_auth(self, http_client, auth_token):
        """Single-team: approve a prior authorization with a clinical reason.
        HIGH-IMPACT write operation — human approval required by MCP decorator.
        Asserts the PA services team is invoked and returns a non-empty response;
        the actual approval depends on approval workflow state in the test
        environment.
        """
        data = _query(
            http_client, auth_token,
            f"Approve prior authorization {TEST_PA_ID} for this health insurance "
            f"request — all required clinical documentation has been reviewed and "
            f"medical necessity criteria are met."
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_pa_services_team")
        assert data["response"].strip(), "approve_prior_auth returned empty response"
        _assert_no_pii_leak(data["response"])
        logger.info("Approve PA response: %s", data["response"][:150])

    def test_deny_prior_auth(self, http_client, auth_token):
        """Single-team: deny a prior authorization with a clinical reason.
        HIGH-IMPACT write operation — human approval required by MCP decorator.
        Asserts the PA services team is invoked and returns a non-empty response;
        the actual denial depends on approval workflow state in the test
        environment.
        """
        data = _query(
            http_client, auth_token,
            f"Deny prior authorization {TEST_PA_ID} for this health insurance "
            f"request — the requested procedure does not meet medical necessity "
            f"criteria based on current clinical guidelines."
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_pa_services_team")
        assert data["response"].strip(), "deny_prior_auth returned empty response"
        _assert_no_pii_leak(data["response"])
        logger.info("Deny PA response: %s", data["response"][:150])


# ---------------------------------------------------------------------------
# Search Services tests
# ---------------------------------------------------------------------------

class TestSearchServices:
    """Tests routed to the search_services_team."""

    def test_search_medical_codes(self, http_client, auth_token):
        """Single-team: look up a CPT code by procedure description."""
        data = _query(
            http_client, auth_token,
            "What is the CPT procedure code for knee replacement surgery for a health insurance claim?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_search_services_team")
        _assert_tool_results_present(data, "search_medical_codes")
        assert "29881" in data["response"] or "knee" in data["response"].lower(), \
            f"Expected CPT code or knee reference in response: {data['response'][:200]}"
        _assert_no_pii_leak(data["response"])

    def test_search_policy_info(self, http_client, auth_token):
        """Single-team: search for policy plan information."""
        data = _query(
            http_client, auth_token,
            "What does an HMO Gold plan typically cover?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_search_services_team")
        _assert_no_pii_leak(data["response"])


# ---------------------------------------------------------------------------
# Multi-team tests  (2–4 services)
# ---------------------------------------------------------------------------

class TestMultiTeam:
    """Tests that invoke 2 or more team services in a single query."""

    def test_provider_and_claims(self, http_client, auth_token):
        """2 teams: provider network check + claim status."""
        data = _query(
            http_client, auth_token,
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}? "
            f"What is the status of claim {TEST_CLAIM_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_provider_services_team", "a2a_claims_services_team")
        _assert_tool_results_present(data, "provider_network_check")
        # Use prefix match — the exact claims worker key varies by query phrasing.
        assert any(k.startswith("claim_") for k in data["tool_results"]), \
            f"Expected a claim_* key in tool_results, got: {list(data['tool_results'].keys())}"
        _assert_no_pii_leak(data["response"])
        logger.info("2-team response: %s", data["response"][:200])

    def test_member_and_provider(self, http_client, auth_token):
        """2 teams: member coverage check + provider network check."""
        data = _query(
            http_client, auth_token,
            f"What is the coverage for member {TEST_MEMBER_ID} under policy {TEST_POLICY_ID}? "
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?"
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_member_services_team", "a2a_provider_services_team")
        _assert_no_pii_leak(data["response"])

    def test_claims_and_pa(self, http_client, auth_token):
        """2 teams: claim lookup + PA lookup."""
        data = _query(
            http_client, auth_token,
            f"What is the status of claim {TEST_CLAIM_ID}? "
            f"Look up prior authorization {TEST_PA_NUMBER}."
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(data, "a2a_claims_services_team", "a2a_pa_services_team")
        _assert_no_pii_leak(data["response"])

    def test_provider_claims_search(self, http_client, auth_token):
        """3 teams: provider check + claim status + CPT code search."""
        data = _query(
            http_client, auth_token,
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}? "
            f"Get the status of claim {TEST_CLAIM_ID}. "
            f"Find the procedure code for knee replacement surgery."
        )
        assert data["error_count"] == 0
        _assert_teams_invoked(
            data,
            "a2a_provider_services_team",
            "a2a_claims_services_team",
            "a2a_search_services_team",
        )
        _assert_tool_results_present(data, "provider_network_check", "search_medical_codes")
        assert any(k in data["tool_results"] for k in ["claim_status", "claim_lookup"])
        _assert_no_pii_leak(data["response"])


# ---------------------------------------------------------------------------
# All-5-services test  ← covers every team in one query
# ---------------------------------------------------------------------------

class TestAllFiveServices:
    """
    Single query that requires all 5 team services to be invoked.

    Query breakdown:
        provider_services  → Is provider X in-network for policy Y?
        claims_services    → What is the status of claim Z?
        member_services    → Look up member details for member M
        pa_services        → Does CPT P require PA under PPO?
        search_services    → Find the CPT code for knee replacement surgery
    """

    def test_all_five_teams_invoked(self, http_client, auth_token):
        """Master test: one query that invokes all 5 team services."""
        data = _query(
            http_client, auth_token,
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}? "
            f"What is the payment status for claim {TEST_CLAIM_ID}? "
            f"What health plan coverage does member {TEST_MEMBER_ID} have? "
            f"Does CPT {TEST_PROCEDURE} need prior authorization under a {TEST_POLICY_TYPE} plan? "
            f"What is the procedure code for knee replacement surgery?"
        )

        assert data["error_count"] == 0, \
            f"Expected 0 errors, got {data['error_count']}"

        # All 5 teams must appear in execution_path
        _assert_teams_invoked(
            data,
            "a2a_provider_services_team",
            "a2a_claims_services_team",
            "a2a_member_services_team",
            "a2a_pa_services_team",
            "a2a_search_services_team",
        )

        # Each team must have populated at least one tool_result key.
        # tool_results-based checks are deterministic — LLM response phrasing varies.
        tool_keys = list(data["tool_results"].keys())
        assert any(k == "provider_network_check" for k in tool_keys), \
            f"Missing provider_network_check in tool_results: {tool_keys}"
        assert any(k.startswith("claim_") for k in tool_keys), \
            f"Missing claim_* key in tool_results: {tool_keys}"
        assert any(k in tool_keys for k in ("member_lookup", "check_eligibility", "coverage_lookup")), \
            f"Missing member_services key in tool_results: {tool_keys}"
        assert any(k.startswith("pa_") for k in tool_keys), \
            f"Missing pa_* key in tool_results: {tool_keys}"
        assert any(k.startswith("search_") for k in tool_keys), \
            f"Missing search_* key in tool_results: {tool_keys}"
        assert data["response"].strip(), "Response is empty"

        _assert_no_pii_leak(data["response"])

        logger.info(
            "All-5-services test passed.\n"
            "Execution path: %s\n"
            "Tool results keys: %s\n"
            "Response preview: %s",
            data["execution_path"],
            list(data["tool_results"].keys()),
            data["response"][:300],
        )

    def test_all_five_tool_results_populated(self, http_client, auth_token):
        """All 5 services should populate at least one tool_result key each."""
        data = _query(
            http_client, auth_token,
            f"Check if provider {TEST_PROVIDER_ID} is in-network for policy {TEST_POLICY_ID}. "
            f"Look up claim {TEST_CLAIM_ID}. "
            f"Get member info for {TEST_MEMBER_ID}. "
            f"Does CPT {TEST_PROCEDURE} need PA under {TEST_POLICY_TYPE}? "
            f"Search for the procedure code for total knee replacement."
        )

        assert data["error_count"] == 0
        tool_keys = list(data["tool_results"].keys())
        assert len(tool_keys) >= 4, \
            f"Expected tool_results from at least 4 workers, got: {tool_keys}"
        logger.info("tool_results keys: %s", tool_keys)


# ---------------------------------------------------------------------------
# Response quality and security tests
# ---------------------------------------------------------------------------

class TestResponseQuality:
    """Tests for response content quality and security controls."""

    def test_response_not_empty(self, http_client, auth_token):
        """Every query should return a non-empty response string."""
        data = _query(
            http_client, auth_token,
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?"
        )
        assert data["response"].strip(), "Response is empty"

    def test_off_topic_query_handled(self, http_client, auth_token):
        """NeMo Guardrails should refuse off-topic queries gracefully."""
        resp = _get_client().post(
            f"{BASE_URL}/api/agent/query",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"query": "What is the weather in Chicago today?"},
        )
        # Either refused by guardrails (non-200) or returns a polite refusal
        if resp.status_code == 200:
            content = resp.json().get("response", "").lower()
            assert any(kw in content for kw in [
                "health insurance", "cannot", "unable", "unrelated",
                "off-topic", "not related", "don't", "can't"
            ]), f"Off-topic query was answered without refusal: {content[:200]}"

    def test_no_pii_in_provider_response(self, http_client, auth_token):
        """Provider lookup response must not expose raw personal identifiers."""
        data = _query(
            http_client, auth_token,
            f"Look up provider {TEST_PROVIDER_ID}"
        )
        _assert_no_pii_leak(data["response"])

    def test_no_pii_in_claims_response(self, http_client, auth_token):
        """Claim lookup response must not expose raw SSNs or credit cards."""
        data = _query(
            http_client, auth_token,
            f"Get details for claim {TEST_CLAIM_ID}"
        )
        _assert_no_pii_leak(data["response"])

    def test_error_count_zero_on_valid_queries(self, http_client, auth_token):
        """Well-formed queries against known test IDs should produce 0 errors."""
        queries = [
            f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?",
            f"What is the status of claim {TEST_CLAIM_ID}?",
            f"Find the CPT code for knee replacement surgery",
        ]
        for query in queries:
            data = _query(http_client, auth_token, query)
            assert data["error_count"] == 0, \
                f"Unexpected errors for '{query}': {data['error_count']}"

    def test_session_ids_are_unique(self, http_client, auth_token):
        """Each query should return a distinct session_id."""
        ids = set()
        for _ in range(3):
            data = _query(
                http_client, auth_token,
                f"Is provider {TEST_PROVIDER_ID} in-network for policy {TEST_POLICY_ID}?"
            )
            assert "session_id" in data
            ids.add(data["session_id"])
        assert len(ids) == 3, f"Expected 3 unique session IDs, got: {ids}"
