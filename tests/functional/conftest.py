"""
Shared fixtures for CSIP functional tests.

Provides authenticated HTTP clients, query helpers, and configuration
for all integration and end-to-end tests. Every test file in this
directory imports these fixtures automatically via pytest's conftest
mechanism.

Usage::

    pytest tests/functional/ -v                     # all functional tests
    pytest tests/functional/ -v -m integration      # integration only
    pytest tests/functional/ -v -m e2e              # end-to-end only
    pytest tests/functional/ -v -k "claims"         # claims-related only

Requires:
    - A running CSIP Docker stack with all services healthy
    - pip install httpx pytest
"""

import os
import time
import pytest
import httpx

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

API_BASE = os.getenv("CSIP_API_BASE", "https://localhost/agentic/access")
VERIFY_SSL = os.getenv("CSIP_VERIFY_SSL", "false").lower() == "true"

# Test user credentials (must exist in MySQL users table)
CSR_USER = os.getenv("CSIP_CSR_USER", "csr2")
CSR_PASS = os.getenv("CSIP_CSR_PASS", "testuser")
CSR_ROLE = "CSR_TIER2"

SUPERVISOR_USER = os.getenv("CSIP_SUPERVISOR_USER", "jchen")
SUPERVISOR_PASS = os.getenv("CSIP_SUPERVISOR_PASS", "testuser")
SUPERVISOR_ROLE = "CSR_SUPERVISOR"

# Timeouts — agent queries can be slow for multi-team plans
QUERY_TIMEOUT = float(os.getenv("CSIP_QUERY_TIMEOUT", "120"))
DEFAULT_TIMEOUT = float(os.getenv("CSIP_DEFAULT_TIMEOUT", "15"))


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------

def _authenticate(username: str, password: str) -> dict:
    """
    Authenticate and return the full login response.

    Returns:
        {"access_token": "...", "token_type": "bearer", "user": {...}}
    """
    with httpx.Client(verify=VERIFY_SSL, timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            f"{API_BASE}/api/auth/login",
            json={"username": username, "password": password},
        )
        assert response.status_code == 200, (
            f"Authentication failed for {username}: "
            f"{response.status_code} {response.text}"
        )
        data = response.json()
        assert "access_token" in data, f"No access_token in response: {data}"
        return data


# ---------------------------------------------------------------------------
# Session-scoped fixtures (one login per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def csr_auth():
    """Authenticate as CSR_TIER2 — used for standard query tests."""
    return _authenticate(CSR_USER, CSR_PASS)


@pytest.fixture(scope="session")
def csr_token(csr_auth):
    """JWT token for the CSR user."""
    return csr_auth["access_token"]


@pytest.fixture(scope="session")
def csr_user(csr_auth):
    """User dict for the CSR user."""
    return csr_auth["user"]


@pytest.fixture(scope="session")
def supervisor_auth():
    """Authenticate as CSR_SUPERVISOR — used for HITL approval tests."""
    return _authenticate(SUPERVISOR_USER, SUPERVISOR_PASS)


@pytest.fixture(scope="session")
def supervisor_token(supervisor_auth):
    """JWT token for the supervisor user."""
    return supervisor_auth["access_token"]


# ---------------------------------------------------------------------------
# Function-scoped HTTP clients (per-test isolation)
# ---------------------------------------------------------------------------

def _make_client(token: str, timeout: float = QUERY_TIMEOUT) -> httpx.Client:
    """Create an authenticated httpx.Client."""
    return httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {token}"},
        verify=VERIFY_SSL,
        timeout=timeout,
    )


@pytest.fixture
def csr_client(csr_token):
    """Authenticated HTTP client for the CSR user."""
    client = _make_client(csr_token)
    yield client
    client.close()


@pytest.fixture
def supervisor_client(supervisor_token):
    """Authenticated HTTP client for the supervisor user."""
    client = _make_client(supervisor_token)
    yield client
    client.close()


@pytest.fixture
def unauthenticated_client():
    """HTTP client with no auth token."""
    client = httpx.Client(
        base_url=API_BASE,
        verify=VERIFY_SSL,
        timeout=DEFAULT_TIMEOUT,
    )
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def query_agent(client: httpx.Client, query: str,
                prior_session_id: str = None) -> dict:
    """
    Send a query to the CSIP agent and return the structured response.

    Args:
        client:            Authenticated httpx.Client.
        query:             Natural-language query string.
        prior_session_id:  Previous session ID for multi-turn chaining.

    Returns:
        QueryResponse dict with keys:
            session_id, response, execution_path, tool_results, error_count
    """
    body = {"query": query}
    if prior_session_id:
        body["prior_session_id"] = prior_session_id

    response = client.post("/api/agent/query", json=body)
    assert response.status_code == 200, (
        f"Agent query failed: {response.status_code} {response.text[:200]}"
    )
    data = response.json()

    # Validate response shape
    assert "session_id" in data, f"Missing session_id: {data.keys()}"
    assert "response" in data, f"Missing response: {data.keys()}"
    assert "execution_path" in data, f"Missing execution_path: {data.keys()}"
    assert "tool_results" in data, f"Missing tool_results: {data.keys()}"
    assert "error_count" in data, f"Missing error_count: {data.keys()}"

    return data


# ---------------------------------------------------------------------------
# Response analysis helpers
# ---------------------------------------------------------------------------

def get_teams_from_path(execution_path: list) -> set:
    """Extract team names from an execution path.

    Nodes like 'a2a_claims_services_team' → 'claims_services_team'.
    """
    teams = set()
    for node in execution_path:
        if node.startswith("a2a_"):
            teams.add(node[4:])
    return teams


def assert_no_errors(response: dict, context: str = ""):
    """Assert that the agent response has zero errors."""
    assert response["error_count"] == 0, (
        f"Expected 0 errors{' (' + context + ')' if context else ''}, "
        f"got {response['error_count']}. "
        f"Response: {response['response'][:200]}"
    )


def assert_team_routed(response: dict, expected_team: str):
    """Assert that the expected team appears in the execution path."""
    teams = get_teams_from_path(response["execution_path"])
    assert expected_team in teams, (
        f"Expected {expected_team} in execution path, "
        f"got teams: {teams}. "
        f"Full path: {response['execution_path']}"
    )


def assert_has_tool_result(response: dict, tool_key: str):
    """Assert that a specific tool result key exists."""
    assert tool_key in response["tool_results"], (
        f"Expected tool result '{tool_key}', "
        f"got keys: {list(response['tool_results'].keys())}"
    )
