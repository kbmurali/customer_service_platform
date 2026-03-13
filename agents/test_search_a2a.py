"""
Integration Tests — Search Services A2A Server
================================================
Tests the live Search services A2A supervisor reachable via Nginx api-gateway
at https://localhost:8443/a2a/search using A2AClientNode as the client.

Prerequisites:
    - api-gateway running on localhost:8443 (mTLS)
    - search-services-supervisor running and healthy
    - search-services-mcp-tools running and healthy
    - .env with:
        MCP_CLIENT_CERT=/home/kbmurali/tmp/certs/mcp/client.crt
        MCP_CLIENT_KEY=/home/kbmurali/tmp/certs/mcp/client.key
        MCP_CA_CERT=/home/kbmurali/tmp/certs/mcp/ca.crt
        A2A_SEARCH_SERVICES_URL=https://localhost:8443/a2a/search

    - TEST_SEARCH_QUERY should be a natural-language query relevant to the
      dev knowledge base (e.g. "What is the referral process for specialists?")
    - TEST_SOURCE should be a valid source filter (faqs, guidelines, regulations, all)
    - TEST_MEDICAL_QUERY should be a natural-language description of a procedure
      or diagnosis (e.g. "knee replacement surgery")
    - TEST_CODE_TYPE should be a valid code type (procedure, diagnosis, both)
    - TEST_POLICY_QUERY should be a natural-language policy question
      (e.g. "What is my annual deductible?")
    - TEST_PLAN_TYPE should be a valid plan type (HMO, PPO, EPO, POS)

Run:
    cd customer_service_platform
    pytest agents/teams/search_services/test_search_a2a.py -v
"""
import os
import logging
import uuid
from typing import Any, Dict

import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())

from agents.core.state import SupervisorState
from agents.teams.search_services.supervisor.tool_schemas import build_search_schema_registry
from agents.core.a2a_client_node import A2AClientNode
from databases.context_graph_data_access import ContextGraphDataAccess

from agents.security import rbac_service, rate_limiter, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_URL  = os.getenv("A2A_SEARCH_SERVICES_URL", "https://localhost:8443/a2a/search")
AGENT_NAME = "search_services_supervisor_agent"

TEST_USER_ID      = os.getenv("TEST_USER_ID",      "usr-tier2-001")
TEST_USER_ROLE    = os.getenv("TEST_USER_ROLE",     "CSR_TIER2")
TEST_SESSION_ID   = os.getenv("TEST_SESSION_ID",    str(uuid.uuid1()))
TEST_SEARCH_QUERY = os.getenv("TEST_SEARCH_QUERY",  "What is the referral process for specialist visits?")
TEST_SOURCE       = os.getenv("TEST_SOURCE",        "faqs")
TEST_MEDICAL_QUERY = os.getenv("TEST_MEDICAL_QUERY", "knee replacement surgery")
TEST_CODE_TYPE    = os.getenv("TEST_CODE_TYPE",     "procedure")
TEST_POLICY_QUERY = os.getenv("TEST_POLICY_QUERY",  "What is my annual deductible?")
TEST_PLAN_TYPE    = os.getenv("TEST_PLAN_TYPE",     "PPO")


cg_dao = ContextGraphDataAccess()
cg_dao.create_session(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)


def exceed_rate_limit(user_id: str, user_role: str, tool_name: str):
    current_rate_limit = rbac_service.get_tool_rate_limit(
        user_role=user_role, tool_name=tool_name
    )

    try:
        for _i in range(current_rate_limit + 1):
            rate_limiter.check_rate_limit(
                user_id=user_id,
                resource_type="TOOL",
                resource_name=tool_name,
                limit_per_minute=current_rate_limit,
            )
    except RateLimitError:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client_node() -> A2AClientNode:
    """Create a shared A2AClientNode for all tests in this module."""
    return A2AClientNode(
        agent_name=AGENT_NAME,
        agent_url=AGENT_URL,
        schema_registry=build_search_schema_registry(),
        from_agent_name="test_client",
    )


def _make_state(query: str, session_id: str | None = None) -> SupervisorState:
    """Build a minimal SupervisorState for a test query."""
    return {
        "messages":       [HumanMessage(content=query)],
        "user_id":        TEST_USER_ID,
        "user_role":      TEST_USER_ROLE,
        "session_id":     session_id or TEST_SESSION_ID,
        "execution_path": [],
        "tool_results":   {},
    }


def _assert_successful_response(result: Dict[str, Any], test_name: str) -> None:
    """Common assertions for a successful A2A response."""
    assert result is not None, f"[{test_name}] Result is None"
    assert "messages" in result, f"[{test_name}] No messages in result"
    assert "error" not in result or result.get("error") is None, \
        f"[{test_name}] Unexpected error: {result.get('error')}"
    assert result["messages"], f"[{test_name}] Empty messages list"

    last_msg = result["messages"][-1]
    content  = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    assert content.strip(), f"[{test_name}] Empty response content"

    logger.info("[%s] Response: %s", test_name, content[:200])


# ---------------------------------------------------------------------------
# Health check (sanity before running full tests)
# ---------------------------------------------------------------------------

class TestA2AServerHealth:
    """Verify the Search Services A2A server is reachable before task tests."""

    def test_health_endpoint(self):
        """GET /health should return 200 with healthy status via mTLS."""
        import ssl
        import httpx

        cert_path = os.getenv("MCP_CLIENT_CERT", "")
        key_path  = os.getenv("MCP_CLIENT_KEY",  "")
        ca_path   = os.getenv("MCP_CA_CERT",     "")

        assert all(os.path.exists(p) for p in [cert_path, key_path, ca_path]), \
            "mTLS cert files not found — check .env MCP_CLIENT_CERT/KEY/CA_CERT"

        ssl_ctx = ssl.create_default_context(cafile=ca_path)
        ssl_ctx.load_cert_chain(cert_path, key_path)

        with httpx.Client(verify=ssl_ctx, timeout=10) as http:
            response = http.get(f"{AGENT_URL}/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert data.get("protocol") == "a2a"
        logger.info("Health check passed: %s", data)

    def test_agent_card_endpoint(self):
        """GET /.well-known/agent.json should return a valid Search agent card."""
        import ssl
        import httpx

        cert_path = os.getenv("MCP_CLIENT_CERT", "")
        key_path  = os.getenv("MCP_CLIENT_KEY",  "")
        ca_path   = os.getenv("MCP_CA_CERT",     "")

        ssl_ctx = ssl.create_default_context(cafile=ca_path)
        ssl_ctx.load_cert_chain(cert_path, key_path)

        with httpx.Client(verify=ssl_ctx, timeout=10) as http:
            response = http.get(f"{AGENT_URL}/.well-known/agent.json")

        assert response.status_code == 200
        card = response.json()
        assert "name" in card
        assert "skills" in card
        assert "url" in card
        assert len(card["skills"]) > 0

        skill_ids = [s.get("id") for s in card.get("skills", [])]
        assert "search_knowledge_base" in skill_ids, f"search_knowledge_base skill missing: {skill_ids}"
        assert "search_medical_codes"  in skill_ids, f"search_medical_codes skill missing: {skill_ids}"
        assert "search_policy_info"    in skill_ids, f"search_policy_info skill missing: {skill_ids}"

        logger.info("Agent card: name=%s, skills=%s", card.get("name"), skill_ids)


# ---------------------------------------------------------------------------
# Search Knowledge Base Tests
# ---------------------------------------------------------------------------

class TestSearchKnowledgeBase:
    """Tests for search_knowledge_base skill via A2A."""

    def test_search_knowledge_base_basic(self, client_node):
        """Search knowledge base with a natural-language query and source filter."""
        state  = _make_state(
            f"Search the {TEST_SOURCE} for: {TEST_SEARCH_QUERY}"
        )
        #exceed_rate_limit(user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="search_knowledge_base")
        result = client_node(state)
        #print(f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_search_knowledge_base_basic")

        content = result["messages"][-1].content
        assert (
            "faq" in content.lower()
            or "guideline" in content.lower()
            or "regulation" in content.lower()
            or "referral" in content.lower()
            or "result" in content.lower()
        ), f"Expected knowledge base reference in response, got: {content[:200]}"

    def test_search_knowledge_base_all_sources(self, client_node):
        """Search across all knowledge base sources."""
        state  = _make_state(
            f"Search all knowledge base sources for: {TEST_SEARCH_QUERY}"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_knowledge_base_all_sources")

    def test_search_knowledge_base_execution_path(self, client_node):
        """Verify execution path includes search_knowledge_base worker."""
        state  = _make_state(
            f"Find FAQs about: {TEST_SEARCH_QUERY}"
        )
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("search_knowledge_base" in str(step).lower() for step in path), \
            f"Expected search_knowledge_base in execution path, got: {path}"

    def test_search_knowledge_base_guidelines(self, client_node):
        """Search clinical guidelines specifically."""
        state  = _make_state(
            "Search clinical guidelines for diabetes management"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_knowledge_base_guidelines")

    def test_search_knowledge_base_regulations(self, client_node):
        """Search regulations specifically."""
        state  = _make_state(
            "Find regulations about emergency care coverage"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_knowledge_base_regulations")

    def test_search_knowledge_base_tool_results(self, client_node):
        """Verify tool_results are populated after knowledge base search."""
        state  = _make_state(
            f"Search FAQs for: {TEST_SEARCH_QUERY}"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Search Medical Codes Tests
# ---------------------------------------------------------------------------

class TestSearchMedicalCodes:
    """Tests for search_medical_codes skill via A2A.

    Note: search_medical_codes takes a natural-language description and
    a code_type filter (procedure, diagnosis, or both).
    """

    def test_search_medical_codes_basic(self, client_node):
        """Search for medical codes using a natural-language description."""
        state  = _make_state(
            f"Find the {TEST_CODE_TYPE} code for {TEST_MEDICAL_QUERY}"
        )
        #exceed_rate_limit(user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="search_medical_codes")
        result = client_node(state)
        #print(f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_search_medical_codes_basic")

        content = result["messages"][-1].content
        assert (
            "code" in content.lower()
            or "cpt" in content.lower()
            or "procedure" in content.lower()
            or TEST_MEDICAL_QUERY.split()[0].lower() in content.lower()
        ), f"Expected medical code reference in response, got: {content[:200]}"

    def test_search_medical_codes_kb_combo(self, client_node):
        """Search for medical codes using a natural-language description."""
        state  = _make_state(
            f"Find the {TEST_CODE_TYPE} code for {TEST_MEDICAL_QUERY}"
            f"Also, Search all knowledge base sources for: {TEST_SEARCH_QUERY}"
        )
        #exceed_rate_limit(user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="search_medical_codes")
        result = client_node(state)
        #print(f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_search_medical_codes_kb_combo")

        content = result["messages"][-1].content
        assert (
            "code" in content.lower()
            or "cpt" in content.lower()
            or "procedure" in content.lower()
            or TEST_MEDICAL_QUERY.split()[0].lower() in content.lower()
        ), f"Expected medical code reference in response, got: {content[:200]}"
        

    def test_search_medical_codes_diagnosis(self, client_node):
        """Search for ICD-10 diagnosis codes specifically."""
        state  = _make_state(
            "Find the ICD-10 diagnosis code for type 2 diabetes"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_medical_codes_diagnosis")

        content = result["messages"][-1].content
        assert (
            "icd" in content.lower()
            or "diagnosis" in content.lower()
            or "code" in content.lower()
            or "diabetes" in content.lower()
        ), f"Expected diagnosis code reference in response, got: {content[:200]}"

    def test_search_medical_codes_both(self, client_node):
        """Search for both procedure and diagnosis codes."""
        state  = _make_state(
            f"Search both procedure and diagnosis codes for {TEST_MEDICAL_QUERY}"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_search_medical_codes_both")

    def test_search_medical_codes_execution_path(self, client_node):
        """Verify execution path includes search_medical_codes worker."""
        state  = _make_state(
            f"What is the CPT code for {TEST_MEDICAL_QUERY}?"
        )
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("search_medical_codes" in str(step).lower() for step in path), \
            f"Expected search_medical_codes in execution path, got: {path}"

    def test_search_medical_codes_all_code_types(self, client_node):
        """Search for the same description across all code types."""
        for code_type in ["procedure", "diagnosis", "both"]:
            state  = _make_state(
                f"Find {code_type} codes for colonoscopy"
            )
            result = client_node(state)
            assert result is not None, f"Result is None for code_type={code_type}"
            assert "messages" in result, f"No messages for code_type={code_type}"
            logger.info(
                "Medical codes for %s/%s: %s",
                "colonoscopy", code_type,
                result["messages"][-1].content[:100] if result["messages"] else "no content"
            )

    def test_search_medical_codes_unknown_description(self, client_node):
        """Unknown medical description — should return a graceful response."""
        state  = _make_state(
            "Find the CPT code for XYZNOTAREALMEDICALTERM12345"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_search_medical_codes_tool_results(self, client_node):
        """Verify tool_results are populated after medical codes search."""
        state  = _make_state(
            f"Find procedure codes for {TEST_MEDICAL_QUERY}"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Search Policy Info Tests
# ---------------------------------------------------------------------------

class TestSearchPolicyInfo:
    """Tests for search_policy_info skill via A2A.

    Note: search_policy_info takes a natural-language query and a plan_type
    (HMO, PPO, EPO, or POS) — both are required.
    """

    def test_search_policy_info_basic(self, client_node):
        """Search policy documents for a known query and plan type."""
        state  = _make_state(
            f"{TEST_POLICY_QUERY} under my {TEST_PLAN_TYPE} plan."
        )
        #exceed_rate_limit(user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="search_policy_info")
        result = client_node(state)
        #print(f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_search_policy_info_basic")

        content = result["messages"][-1].content
        assert (
            "deductible" in content.lower()
            or "policy" in content.lower()
            or "plan" in content.lower()
            or TEST_PLAN_TYPE.lower() in content.lower()
        ), f"Expected policy reference in response, got: {content[:200]}"

    def test_search_policy_info_combo(self, client_node):
        """Search policy info and knowledge base in a single multi-intent query."""
        state  = _make_state(
            f"{TEST_POLICY_QUERY} under my {TEST_PLAN_TYPE} plan. "
            f"Also search the FAQs for: {TEST_SEARCH_QUERY}"
        )
        #exceed_rate_limit(user_id=TEST_USER_ID, user_role=TEST_USER_ROLE, tool_name="search_policy_info")
        result = client_node(state)
        #print(f">>>>>>>>\n\n{result}\n")
        _assert_successful_response(result, "test_search_policy_info_combo")

        content = result["messages"][-1].content
        assert (
            "policy" in content.lower()
            or "plan" in content.lower()
            or "faq" in content.lower()
            or "result" in content.lower()
        ), f"Expected policy or FAQ reference in response, got: {content[:200]}"

    def test_search_policy_info_execution_path(self, client_node):
        """Verify execution path includes search_policy_info worker."""
        state  = _make_state(
            f"What is the out-of-pocket maximum for my {TEST_PLAN_TYPE} plan?"
        )
        result = client_node(state)

        assert "execution_path" in result
        path = result["execution_path"]
        assert any("search_policy_info" in str(step).lower() for step in path), \
            f"Expected search_policy_info in execution path, got: {path}"

    def test_search_policy_info_all_plan_types(self, client_node):
        """Search policy info for the same question across all plan types."""
        for plan_type in ["HMO", "PPO", "EPO", "POS"]:
            state  = _make_state(
                f"What is the annual deductible under a {plan_type} plan?"
            )
            result = client_node(state)
            assert result is not None, f"Result is None for plan_type={plan_type}"
            assert "messages" in result, f"No messages for plan_type={plan_type}"
            logger.info(
                "Policy info for %s: %s",
                plan_type,
                result["messages"][-1].content[:100] if result["messages"] else "no content"
            )

    def test_search_policy_info_without_plan_type(self, client_node):
        """Policy query missing plan type — supervisor should SKIP gracefully."""
        state  = _make_state(
            "What is my annual deductible?"
        )
        result = client_node(state)

        # SKIP path — supervisor has no plan_type to route with.
        # Should still return a non-empty response.
        assert result is not None
        assert "messages" in result

    def test_search_policy_info_tool_results(self, client_node):
        """Verify tool_results are populated after policy info search."""
        state  = _make_state(
            f"What are the premium costs for a {TEST_PLAN_TYPE} plan?"
        )
        result = client_node(state)

        assert "tool_results" in result
        logger.info("Tool results keys: %s", list(result["tool_results"].keys()))


# ---------------------------------------------------------------------------
# Multi-skill / Routing Tests
# ---------------------------------------------------------------------------

class TestA2ARouting:
    """Tests that verify the supervisor routes correctly to the right worker."""

    def test_knowledge_base_and_policy_query(self, client_node):
        """A query asking for FAQs and policy info should route to both workers."""
        state  = _make_state(
            f"Search the FAQs for referral requirements and also look up "
            f"deductible information for my {TEST_PLAN_TYPE} plan."
        )
        result = client_node(state)
        _assert_successful_response(result, "test_knowledge_base_and_policy_query")

        path     = result.get("execution_path", [])
        path_str = " ".join(str(s) for s in path).lower()
        assert "search_knowledge_base" in path_str or "search_policy_info" in path_str, \
            f"Expected search workers in execution path, got: {path}"

    def test_a2a_task_state_completed(self, client_node):
        """Successful tasks should have execution path ending with FINISH."""
        state  = _make_state(
            f"Search FAQs for: {TEST_SEARCH_QUERY}"
        )
        result = client_node(state)

        exec_path_list = result.get("execution_path")
        assert "search_services_supervisor -> FINISH (all steps done)" in exec_path_list, \
            f"Expected FINISH in execution path, got '{exec_path_list}'"

    def test_medical_codes_vs_policy_routing(self, client_node):
        """Supervisor should route medical description queries to search_medical_codes,
        not search_policy_info."""
        state  = _make_state(
            f"What is the CPT code for {TEST_MEDICAL_QUERY}?"
        )
        result = client_node(state)
        _assert_successful_response(result, "test_medical_codes_vs_policy_routing")

        path     = result.get("execution_path", [])
        path_str = " ".join(str(s) for s in path).lower()
        assert "search_medical_codes" in path_str, \
            f"Expected search_medical_codes worker for medical description query, got: {path}"

    def test_session_isolation(self, client_node):
        """Two requests with different session IDs should not interfere."""
        session_a = f"test-session-a-{uuid.uuid4().hex[:6]}"
        session_b = f"test-session-b-{uuid.uuid4().hex[:6]}"

        state_a = _make_state(f"Search FAQs for: {TEST_SEARCH_QUERY}", session_id=session_a)
        state_b = _make_state(f"Search FAQs for: {TEST_SEARCH_QUERY}", session_id=session_b)

        result_a = client_node(state_a)
        result_b = client_node(state_b)

        _assert_successful_response(result_a, "session_a")
        _assert_successful_response(result_b, "session_b")


# ---------------------------------------------------------------------------
# Error / Edge Case Tests
# ---------------------------------------------------------------------------

class TestA2AErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_query(self, client_node):
        """Empty query should return a graceful response, not crash."""
        state  = _make_state("")
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_source_filter(self, client_node):
        """An unrecognised source filter should be handled gracefully."""
        state  = _make_state(
            "Search the UNKNOWN_SOURCE for referral requirements"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_code_type(self, client_node):
        """An unrecognised code type should be handled gracefully."""
        state  = _make_state(
            f"Find UNKNOWNTYPE codes for {TEST_MEDICAL_QUERY}"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_invalid_plan_type(self, client_node):
        """An unrecognised plan type should be handled gracefully."""
        state  = _make_state(
            f"What is my deductible under a UNKNOWN plan?"
        )
        result = client_node(state)
        assert result is not None
        assert "messages" in result

    def test_result_has_no_unhandled_exception(self, client_node):
        """Any query should return a result dict, never raise an exception."""
        queries = [
            "What are the referral requirements for specialists?",
            f"Search FAQs for: {TEST_SEARCH_QUERY}",
            f"Find the CPT code for {TEST_MEDICAL_QUERY}",
            f"Search {TEST_CODE_TYPE} codes for colonoscopy",
            f"What is my deductible under a {TEST_PLAN_TYPE} plan?",
        ]
        for query in queries:
            state = _make_state(query)
            try:
                result = client_node(state)
                assert isinstance(result, dict), \
                    f"Expected dict, got {type(result)} for query: '{query}'"
            except Exception as e:
                pytest.fail(
                    f"A2AClientNode raised unexpected exception for '{query}': {e}"
                )
