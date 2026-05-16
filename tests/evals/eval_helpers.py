"""
Shared helper functions for CSIP AI evaluation tests.

These are plain functions (not pytest fixtures) that test modules
import directly. Fixtures remain in conftest.py where pytest
auto-discovers them.
"""
import json
import os
from typing import Any, Dict, List, Optional

import httpx
import pytest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("CSIP_API_BASE", "https://localhost/agentic/access")
VERIFY_SSL = os.getenv("CSIP_VERIFY_SSL", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------
def query_agent(
    client: httpx.Client,
    query: str,
    prior_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a query to the CSIP agent and return the structured response.

    Returns dict with keys:
        session_id, response, execution_path, tool_results, error_count
    """
    body = {"query": query}
    if prior_session_id:
        body["prior_session_id"] = prior_session_id

    response = client.post("/api/agent/query", json=body)
    assert response.status_code == 200, (
        f"Query failed: {response.status_code} {response.text[:300]}"
    )
    data = response.json()

    assert "session_id" in data, f"Missing session_id: {data.keys()}"
    assert "response" in data, f"Missing response: {data.keys()}"
    assert "tool_results" in data, f"Missing tool_results: {data.keys()}"
    return data


# ---------------------------------------------------------------------------
# Context Graph helpers
# ---------------------------------------------------------------------------
def fetch_cg_tree(client: httpx.Client, session_id: str) -> Dict[str, Any]:
    """Fetch the full CG execution tree for a session."""
    response = client.get(f"/api/cg/session/{session_id}")
    assert response.status_code == 200, (
        f"CG tree fetch failed: {response.status_code} {response.text[:200]}"
    )
    return response.json()


def find_decision_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk the CG tree recursively and return all AgentExecution nodes
    where isDecisionNode is true.
    """
    results = []
    _walk_tree(tree, results)
    return results


def _walk_tree(node: Dict[str, Any], results: List[Dict[str, Any]]):
    """Recursive tree walker for decision node extraction."""
    if not node:
        return
    props = node.get("props", {})
    if props.get("isDecisionNode") is True:
        results.append(props)
    for child in node.get("children", []):
        _walk_tree(child, results)


def get_teams_from_path(execution_path: List[str]) -> set:
    """Extract team names from an execution path."""
    teams = set()
    for node in execution_path:
        if node.startswith("a2a_"):
            teams.add(node[4:])
    return teams


def extract_tool_results_text(tool_results: Dict[str, Any]) -> str:
    """
    Flatten tool_results dict into a single text string suitable
    for use as retrieval_context in DeepEval test cases.
    """
    parts = []
    for tool_name, result in tool_results.items():
        if isinstance(result, dict):
            output = result.get("output", result.get("tool_raw_output", ""))
            if output:
                parts.append(f"[{tool_name}]: {output}")
        elif isinstance(result, str):
            parts.append(f"[{tool_name}]: {result}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Experience store helpers
# ---------------------------------------------------------------------------
def seed_experience(
    client: httpx.Client,
    session_id: str,
    query_text: str,
    plan_json: str,
    team_assignments: str,
) -> bool:
    """
    Seed an experience into the production Chroma collection.
    """
    try:
        from databases.chroma_experience_store import get_experience_store
        store = get_experience_store()
        return store.store_experience(
            session_id=session_id,
            query_text=query_text,
            plan_json=plan_json,
            team_assignments=team_assignments,
            result_summary="Test experience for eval",
            rating="correct",
        )
    except ImportError:
        pytest.skip("ChromaExperienceStore not available outside CSIP container")
        return False


def remove_experience(session_id: str) -> bool:
    """Remove a test experience from the production Chroma collection."""
    try:
        from databases.chroma_experience_store import get_experience_store
        return get_experience_store().remove_experience(session_id)
    except ImportError:
        return False
