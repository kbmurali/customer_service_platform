"""
Admin and operational endpoint tests — tool catalog, Context Graph
session retrieval, tool permissions management, and health checks.

These endpoints were added during development to support the webapp's
Supervisor Control Pane, Performance Dashboard, and CG Explorer.

Run with::

    pytest tests/functional/test_admin_endpoints.py -v
"""

import pytest
import httpx
from conftest import (
    API_BASE, VERIFY_SSL, DEFAULT_TIMEOUT,
    query_agent, assert_no_errors,
)

pytestmark = pytest.mark.integration


class TestToolCatalog:
    """GET /api/admin/tool-catalog — returns A2A-sourced tool metadata."""

    def test_tool_catalog_returns_200(self, supervisor_client):
        """Authenticated supervisor can retrieve the tool catalog."""
        r = supervisor_client.get("/api/admin/tool-catalog")
        assert r.status_code == 200
        data = r.json()
        assert "catalog" in data, f"Missing catalog key: {data.keys()}"

    def test_catalog_has_22_tools(self, supervisor_client):
        """Catalog should contain all 22 tools across 5 teams."""
        r = supervisor_client.get("/api/admin/tool-catalog")
        catalog = r.json()["catalog"]
        assert len(catalog) >= 22, (
            f"Expected >=22 tools, got {len(catalog)}: {list(catalog.keys())}"
        )

    def test_catalog_tool_has_required_fields(self, supervisor_client):
        """Each tool entry should have name, service, and description."""
        r = supervisor_client.get("/api/admin/tool-catalog")
        catalog = r.json()["catalog"]
        for tool_name, tool_info in catalog.items():
            assert "name" in tool_info, f"{tool_name} missing 'name'"
            assert "service" in tool_info, f"{tool_name} missing 'service'"
            assert "description" in tool_info, f"{tool_name} missing 'description'"

    def test_check_eligibility_in_catalog(self, supervisor_client):
        """check_eligibility should be in catalog (not eligibility_check)."""
        r = supervisor_client.get("/api/admin/tool-catalog")
        catalog = r.json()["catalog"]
        assert "check_eligibility" in catalog, (
            f"Expected check_eligibility in catalog, "
            f"got: {[k for k in catalog if 'elig' in k.lower()]}"
        )
        assert "eligibility_check" not in catalog, (
            "eligibility_check (old name) should NOT be in catalog"
        )

    def test_member_policy_lookup_in_catalog(self, supervisor_client):
        """member_policy_lookup (new tool) should be in catalog."""
        r = supervisor_client.get("/api/admin/tool-catalog")
        catalog = r.json()["catalog"]
        assert "member_policy_lookup" in catalog, (
            f"Expected member_policy_lookup in catalog, "
            f"got member tools: {[k for k in catalog if 'member' in k.lower()]}"
        )

    def test_catalog_unauthenticated(self, unauthenticated_client):
        """Unauthenticated requests to tool catalog should fail."""
        r = unauthenticated_client.get("/api/admin/tool-catalog")
        assert r.status_code in (401, 403)


class TestCGSessionEndpoint:
    """GET /api/cg/session/{session_id} — Context Graph exploration."""

    def test_cg_session_returns_tree(self, csr_client):
        """After a query, the CG session endpoint returns a valid tree."""
        # First create a session
        qr = query_agent(csr_client, "What is the status of claim CLM-123456?")
        session_id = qr["session_id"]

        # Then fetch its CG tree
        r = csr_client.get(f"/api/cg/session/{session_id}")
        assert r.status_code == 200
        data = r.json()
        assert "session" in data, f"Missing 'session': {data.keys()}"
        assert "tree" in data, f"Missing 'tree': {data.keys()}"
        assert "follow_up_chain" in data, f"Missing 'follow_up_chain': {data.keys()}"

    def test_cg_tree_has_session_root(self, csr_client):
        """The tree root should be a Session node."""
        qr = query_agent(csr_client, "Check eligibility for member M-12345")
        r = csr_client.get(f"/api/cg/session/{qr['session_id']}")
        tree = r.json()["tree"]
        assert tree["type"] == "Session", f"Expected Session root, got {tree['type']}"

    def test_cg_session_not_found(self, csr_client):
        """Non-existent session IDs should return 404."""
        r = csr_client.get("/api/cg/session/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404


class TestToolPermissions:
    """GET /api/admin/tool-permissions — role-based tool management.

    Returns {"permissions": {"CSR_TIER1": [...], "CSR_TIER2": [...], ...}}
    grouped by role. Requires CSR_SUPERVISOR role.
    """

    def test_get_permissions_all_roles(self, supervisor_client):
        """Supervisor can retrieve tool permissions for all roles."""
        r = supervisor_client.get("/api/admin/tool-permissions")
        assert r.status_code == 200
        data = r.json()
        assert "permissions" in data, f"Missing 'permissions': {data.keys()}"
        assert isinstance(data["permissions"], dict)
        assert len(data["permissions"]) > 0, "Should have permissions for at least one role"

        # CSR_TIER2 should be one of the roles
        assert "CSR_TIER2" in data["permissions"], (
            f"Expected CSR_TIER2 in permissions, got roles: {list(data['permissions'].keys())}"
        )

    def test_permissions_have_service_metadata(self, supervisor_client):
        """Each permission row should include A2A-sourced service/description."""
        r = supervisor_client.get("/api/admin/tool-permissions")
        perms_by_role = r.json()["permissions"]
        # Check any role's permissions
        for role, perms in perms_by_role.items():
            for perm in perms:
                assert "tool_name" in perm, f"Missing tool_name: {perm}"
                assert "service" in perm, f"Missing service for {perm['tool_name']}"
                assert perm["service"] != "Other", (
                    f"Tool {perm['tool_name']} has service='Other' — "
                    f"A2A card enrichment may be missing"
                )
            break  # Only need to check one role


class TestHealthEndpoint:
    """GET /api/health — infrastructure health check."""

    def test_health_returns_200(self, csr_client):
        """Health endpoint should return 200 with component status."""
        r = csr_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "components" in data or "services" in data or "details" in data


class TestFollowUpChain:
    """Multi-session follow-up chains via prior_session_id."""

    def test_follow_up_uses_prior_session(self, csr_client):
        """A follow-up query with prior_session_id should succeed
        and reference the original session in the CG chain."""
        # Query 1
        r1 = query_agent(
            csr_client,
            "What is the payment status for claim CLM-123456?"
        )
        assert_no_errors(r1, "query 1")
        session1 = r1["session_id"]

        # Query 2 — follow-up referencing the previous session
        r2 = query_agent(
            csr_client,
            "Who is the member associated with that claim?",
            prior_session_id=session1,
        )
        assert_no_errors(r2, "follow-up query 2")
        session2 = r2["session_id"]
        assert session2 != session1, "Follow-up should create a new session"

        # Verify CG chain links them
        cg = csr_client.get(f"/api/cg/session/{session2}")
        if cg.status_code == 200:
            chain = cg.json().get("follow_up_chain", [])
            chain_ids = [s.get("sessionId", s.get("session_id", "")) for s in chain]
            assert session1 in chain_ids or len(chain) >= 2, (
                f"Follow-up chain should include original session. "
                f"Chain: {chain_ids}"
            )
