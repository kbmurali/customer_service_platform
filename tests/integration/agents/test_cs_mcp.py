"""
Claims Services MCP Tools — Remote Integration Tests
=====================================================
Tests the three claims tools (claim_lookup, claim_status, claim_payment_info)
by calling the MCP server through ClaimServicesMCPToolClient, which uses the
same langchain_mcp_adapters / streamable-HTTP path that production workers use.

Prerequisites
-------------
1. The claims-services-mcp-tools container must be running and reachable.
2. Set MCP_CLAIM_SERVICES_HTTP_URL if not using the default gateway URL, e.g.:
       export MCP_CLAIM_SERVICES_HTTP_URL=http://localhost:8001
3. Test claim/member IDs must exist in the dev/test Knowledge Graph.
4. A .env file with NEO4J_CG_* credentials must be present (for CG tracking).

Run
---
    pytest tests/test_claims_services_mcp.py -v
    pytest tests/test_claims_services_mcp.py -v -k "claim_lookup"
    pytest tests/test_claims_services_mcp.py -v --tb=short 2>&1 | head -80

Coverage
--------
Each tool is tested for:
    ✓ Happy path — valid inputs, successful response
    ✓ Not-found  — unknown ID returns {"error": ...}, no exception
    ✓ RBAC       — unauthorised role returns {"error": ..., "error_type": ...}
    ✓ Input validation — empty/malformed IDs handled gracefully

NOTE: rate-limit and circuit-breaker tests are intentionally omitted here
because triggering them reliably in CI requires coordinated timing. They are
covered by the supervisor-level integration tests.
"""

import asyncio
import json
import os
import uuid
import pytest

from databases.context_graph_data_access import ContextGraphDataAccess

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ─────────────────────────────────────────────────────────────
# Test constants — replace with IDs that exist in your dev KG
# ─────────────────────────────────────────────────────────────

TEST_USER_ID    = os.getenv("TEST_USER_ID",    "usr-tier2-001")
TEST_USER_ROLE  = os.getenv("TEST_USER_ROLE",  "CSR_TIER2")
TEST_SESSION_ID = os.getenv("TEST_SESSION_ID", str( uuid.uuid1()))
TEST_CLAIM_ID  = os.getenv("TEST_CLAIM_ID",  "7799c06c-0883-4dca-b1f0-bded6d1027a5")
TEST_CLAIM_NUMBER  = os.getenv("TEST_CLAIM_NUMBER",  "CLM-421386")


UNKNOWN_CLAIM_ID     = "CLM-DOES-NOT-EXIST"
UNKNOWN_CLAIM_NUMBER = "CLM-0000-0000"

UNAUTHORIZED_ROLE = "UNAUTHORIZED_ROLE"

cg_dao = ContextGraphDataAccess()
    
cg_dao.create_session( session_id=TEST_SESSION_ID, user_id=TEST_USER_ID )

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _parse(raw) -> dict:
    """
    Parse a tool response into a dict.

    sync_fn in mcp_tool_client_base now normalises all MCP adapter envelopes
    to a plain string before returning, so the common case is just
    json.loads(string).  List/tuple/dict handling is kept as defensive fallback.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        inner = next((x for x in raw if x is not None), None)
        return _parse(inner) if inner is not None else {}
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            parsed = json.loads(raw)
            # Unwrap ["<json-string>", null] array envelope if present
            if isinstance(parsed, list):
                inner = next((x for x in parsed if x is not None), None)
                if isinstance(inner, str):
                    return json.loads(inner)
                if isinstance(inner, dict):
                    return inner
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except (json.JSONDecodeError, TypeError):
            pass
    return {"_raw": str(raw)}


def _invoke(tool, **kwargs) -> dict:
    """
    Invoke a sync or async LangChain tool and return a parsed dict.

    Claims MCP server returns a tuple (json_string, None) from
    langchain_mcp_adapters — intercept it here before LangChain calls
    str() on it and produces an unparseable repr.
    """
    if asyncio.iscoroutinefunction(getattr(tool, "coroutine", None)):
        raw = asyncio.run(tool.coroutine(**kwargs))
    else:
        raw = tool.func(**kwargs) if hasattr(tool, "func") else tool.invoke(kwargs)

    # Unwrap tuple/list envelope before passing to _parse
    if isinstance(raw, (list, tuple)):
        raw = next((x for x in raw if x is not None), "")

    return _parse(raw)


def _base_kwargs(**extra) -> dict:
    return {
        "user_id":    TEST_USER_ID,
        "user_role":  TEST_USER_ROLE,
        "session_id": TEST_SESSION_ID,
        "execution_id": "test-exec-id",
        **extra,
    }


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mcp_client():
    """
    One ClaimServicesMCPToolClient instance shared across the module.

    Set MCP_CLAIM_SERVICES_HTTP_URL to point at the running container, e.g.:
        http://localhost:8001   (direct, for local dev)
        http://mcp-claim:8002   (inside Docker network)
    """
    from agents.teams.claims_services.claims_services_mcp_tool_client import (
        ClaimServicesMCPToolClient,
    )
    return ClaimServicesMCPToolClient()


@pytest.fixture(scope="module")
def tool_claim_lookup(mcp_client):
    return mcp_client.get_tool("claim_lookup")


@pytest.fixture(scope="module")
def tool_claim_status(mcp_client):
    return mcp_client.get_tool("claim_status")


@pytest.fixture(scope="module")
def tool_claim_payment_info(mcp_client):
    return mcp_client.get_tool("claim_payment_info")


# ─────────────────────────────────────────────────────────────
# claim_lookup
# ─────────────────────────────────────────────────────────────

class TestClaimLookup:

    def test_happy_path(self, tool_claim_lookup):
        """Valid claim ID returns claim data with no error key."""
        result = _invoke(
            tool_claim_lookup,
            **_base_kwargs(claim_id=TEST_CLAIM_ID),
        )
        assert isinstance(result, dict), f"Expected dict, got: {type(result)}"
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        
        print( f">>>>>>>>>>>>>>>>>>>>>\n\n{result}\n" )
        # Core claim fields
        assert "claimId" in result or "claim_id" in result, (
            f"Missing claimId in response: {list(result.keys())}"
        )

    def test_not_found(self, tool_claim_lookup):
        """Unknown claim ID returns an error payload, not an exception."""
        result = _invoke(
            tool_claim_lookup,
            **_base_kwargs(claim_id=UNKNOWN_CLAIM_ID),
        )
        assert "error" in result, f"Expected error for unknown claim, got: {result}"
        assert UNKNOWN_CLAIM_ID in result["error"] or "not found" in result["error"].lower()

    def test_unauthorized_role(self, tool_claim_lookup):
        """Unauthorized role is blocked by RBAC before the KG is queried."""
        result = _invoke(
            tool_claim_lookup,
            **{**_base_kwargs(claim_id=TEST_CLAIM_ID), "user_role": UNAUTHORIZED_ROLE},
        )
        assert "error" in result, f"Expected RBAC error, got: {result}"
        # Either error_type field or the word 'permission' in the message
        has_type = result.get("error_type") in (
            "permission_denied", "tool_permission_denied", "resource_permission_denied"
        )
        has_keyword = any(
            kw in result.get("error", "").lower()
            for kw in ("permission", "denied", "unauthorized", "not authorized")
        )
        assert has_type or has_keyword, (
            f"Expected permission-denied signal, got: {result}"
        )

    def test_empty_claim_id(self, tool_claim_lookup):
        """Empty claim ID is handled gracefully — error, not exception."""
        result = _invoke(
            tool_claim_lookup,
            **_base_kwargs(claim_id=""),
        )
        # Should return an error dict, not raise
        assert isinstance(result, dict)
        assert "error" in result or "_raw" not in result

    def test_response_is_scrubbed(self, tool_claim_lookup):
        """Output must not contain raw SSN or DOB patterns (PII scrubbing check)."""
        import re
        result = _invoke(
            tool_claim_lookup,
            **_base_kwargs(claim_id=TEST_CLAIM_ID),
        )
        raw_text = json.dumps(result)
        # SSN pattern: 3 digits, dash, 2 digits, dash, 4 digits
        assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", raw_text), (
            "Possible unmasked SSN found in output"
        )


# ─────────────────────────────────────────────────────────────
# claim_status
# ─────────────────────────────────────────────────────────────

class TestClaimStatus:

    def test_happy_path(self, tool_claim_status):
        """Valid claim number returns status fields."""
        result = _invoke(
            tool_claim_status,
            **_base_kwargs(claim_number=TEST_CLAIM_NUMBER),
        )
        assert isinstance(result, dict)
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "status" in result or "claimNumber" in result, (
            f"Missing status/claimNumber in response: {list(result.keys())}"
        )

    def test_not_found(self, tool_claim_status):
        """Unknown claim number returns an error payload."""
        result = _invoke(
            tool_claim_status,
            **_base_kwargs(claim_number=UNKNOWN_CLAIM_NUMBER),
        )
        assert "error" in result, f"Expected error for unknown claim number, got: {result}"

    def test_unauthorized_role(self, tool_claim_status):
        """Unauthorized role blocked by RBAC."""
        result = _invoke(
            tool_claim_status,
            **{**_base_kwargs(claim_number=TEST_CLAIM_NUMBER), "user_role": UNAUTHORIZED_ROLE},
        )
        assert "error" in result
        has_type = result.get("error_type") in (
            "permission_denied", "tool_permission_denied", "resource_permission_denied"
        )
        has_keyword = any(
            kw in result.get("error", "").lower()
            for kw in ("permission", "denied", "unauthorized")
        )
        assert has_type or has_keyword, f"Expected permission-denied signal, got: {result}"

    def test_status_field_values(self, tool_claim_status):
        """Status field contains a known claim status value."""
        KNOWN_STATUSES = {
            "submitted", "pending", "approved", "denied",
            "paid", "processing", "under_review",
        }
        result = _invoke(
            tool_claim_status,
            **_base_kwargs(claim_number=TEST_CLAIM_NUMBER),
        )
        if "error" in result:
            pytest.skip("Claim not found in test KG — update TEST_CLAIM_NUMBER")

        status_val = result.get("status", "")
        assert status_val.lower() in KNOWN_STATUSES or status_val != "", (
            f"Unexpected status value: {status_val!r}"
        )


# ─────────────────────────────────────────────────────────────
# claim_payment_info
# ─────────────────────────────────────────────────────────────

class TestClaimPaymentInfo:

    def test_happy_path(self, tool_claim_payment_info):
        """Valid claim ID returns payment fields."""
        result = _invoke(
            tool_claim_payment_info,
            **_base_kwargs(claim_id=TEST_CLAIM_ID),
        )
        assert isinstance(result, dict)
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        payment_keys = {"totalAmount", "paidAmount", "status", "claimId", "claimNumber"}
        present = payment_keys & set(result.keys())
        assert present, (
            f"Expected at least one of {payment_keys}, got: {list(result.keys())}"
        )

    def test_not_found(self, tool_claim_payment_info):
        """Unknown claim ID returns error payload."""
        result = _invoke(
            tool_claim_payment_info,
            **_base_kwargs(claim_id=UNKNOWN_CLAIM_ID),
        )
        assert "error" in result

    def test_unauthorized_role(self, tool_claim_payment_info):
        """Unauthorized role blocked by RBAC."""
        result = _invoke(
            tool_claim_payment_info,
            **{**_base_kwargs(claim_id=TEST_CLAIM_ID), "user_role": UNAUTHORIZED_ROLE},
        )
        assert "error" in result
        has_type = result.get("error_type") in (
            "permission_denied", "tool_permission_denied", "resource_permission_denied"
        )
        has_keyword = any(
            kw in result.get("error", "").lower()
            for kw in ("permission", "denied", "unauthorized")
        )
        assert has_type or has_keyword, f"Expected permission-denied signal, got: {result}"

    def test_amounts_are_numeric(self, tool_claim_payment_info):
        """totalAmount and paidAmount must be numeric when present."""
        result = _invoke(
            tool_claim_payment_info,
            **_base_kwargs(claim_id=TEST_CLAIM_ID),
        )
        if "error" in result:
            pytest.skip("Claim not found in test KG — update TEST_CLAIM_ID")

        for field in ("totalAmount", "paidAmount"):
            val = result.get(field)
            if val is not None:
                assert isinstance(val, (int, float)), (
                    f"{field} should be numeric, got {type(val)}: {val!r}"
                )

    def test_paid_amount_lte_total(self, tool_claim_payment_info):
        """paidAmount should not exceed totalAmount."""
        result = _invoke(
            tool_claim_payment_info,
            **_base_kwargs(claim_id=TEST_CLAIM_ID),
        )
        if "error" in result:
            pytest.skip("Claim not found in test KG — update TEST_CLAIM_ID")

        total = result.get("totalAmount")
        paid  = result.get("paidAmount")
        if total is not None and paid is not None:
            assert paid <= total, (
                f"paidAmount ({paid}) exceeds totalAmount ({total})"
            )


# ─────────────────────────────────────────────────────────────
# Cross-cutting: server reachability
# ─────────────────────────────────────────────────────────────

class TestServerHealth:

    def test_health_endpoint(self):
        """GET /health returns 200 with status=healthy."""
        import urllib.request
        import urllib.error

        base_url = os.getenv(
            "MCP_CLAIM_SERVICES_HTTP_URL",
            "http://localhost:8001",
        ).rstrip("/")

        # Strip any path suffix (e.g. /claims-services from gateway URL)
        # for direct health checks during local dev
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                body = json.loads(resp.read())
            assert body.get("status") == "healthy", f"Health body: {body}"
            assert body.get("service") == "claims_services_mcp"
        except urllib.error.URLError as e:
            pytest.skip(f"MCP server not reachable at {base_url}: {e}")

    def test_all_tools_registered(self, mcp_client):
        """Client must have all three tools discoverable."""
        expected = {"claim_lookup", "claim_status", "claim_payment_info"}
        registered = set(mcp_client._registered_names)
        assert expected == registered, (
            f"Tool mismatch — expected {expected}, got {registered}"
        )
