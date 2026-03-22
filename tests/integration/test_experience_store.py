"""
Integration tests for the Chroma Experience Store.

These tests require a running Chroma instance (the same one used by the
CSIP Docker stack).  Run with::

    pytest tests/integration/test_experience_store.py -v

The tests use a unique collection name to avoid contaminating the
production experience store.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_COLLECTION = "test_successful_experiences"


@pytest.fixture
def experience_store():
    """
    Create an ExperienceStore instance pointing at a test collection.
    Cleans up the test collection after the test.
    """
    from databases.chroma_experience_store import ChromaExperienceStore
    from databases.connections import get_chroma

    store = ChromaExperienceStore(chroma_conn=get_chroma())
    # Override collection name for test isolation
    store.COLLECTION_NAME = TEST_COLLECTION

    yield store

    # Cleanup: delete the test collection
    try:
        client = get_chroma().connect()
        client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


@pytest.fixture
def sample_plan():
    """A minimal plan JSON for testing."""
    return json.dumps({
        "goals": [{"id": "goal_1", "description": "Check claim status", "priority": 1}],
        "steps": [{"step_id": "step_1", "goal_id": "goal_1",
                    "agent": "claims_services_team", "order": 1}],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperienceStoreBasics:
    """Test store, retrieve, and remove operations."""

    def test_store_and_retrieve(self, experience_store, sample_plan):
        """Store an experience, then retrieve it by similar query."""
        success = experience_store.store_experience(
            session_id="test-session-001",
            query_text="What is the status of claim CLM-123456?",
            plan_json=sample_plan,
            team_assignments="claims_services_team",
            result_summary="Claim is under review",
            rating="correct",
        )
        assert success is True

        # Retrieve with a similar query
        result = experience_store.retrieve_similar_experiences(
            query_text="Check the status of claim CLM-789012",
            top_k=3,
        )
        assert result != ""
        assert "claims_services_team" in result
        assert "goal" in result.lower()

    def test_empty_collection_returns_empty_string(self, experience_store):
        """An empty collection should return gracefully."""
        result = experience_store.retrieve_similar_experiences(
            query_text="What is my eligibility?",
            top_k=3,
        )
        assert result == ""

    def test_remove_experience(self, experience_store, sample_plan):
        """Remove should delete the experience so it's no longer retrievable."""
        experience_store.store_experience(
            session_id="test-session-remove",
            query_text="Remove me please",
            plan_json=sample_plan,
            team_assignments="claims_services_team",
        )

        # Verify it's there
        assert experience_store.get_collection_size() >= 1

        # Remove
        success = experience_store.remove_experience("test-session-remove")
        assert success is True

    def test_upsert_deduplication(self, experience_store, sample_plan):
        """Storing the same session_id twice should not create duplicates."""
        experience_store.store_experience(
            session_id="test-session-dedup",
            query_text="Original query",
            plan_json=sample_plan,
            team_assignments="claims_services_team",
        )
        initial_size = experience_store.get_collection_size()

        # Store again with updated query
        experience_store.store_experience(
            session_id="test-session-dedup",
            query_text="Updated query",
            plan_json=sample_plan,
            team_assignments="claims_services_team",
        )
        assert experience_store.get_collection_size() == initial_size

    def test_collection_size(self, experience_store, sample_plan):
        """get_collection_size should reflect stored documents."""
        assert experience_store.get_collection_size() == 0

        experience_store.store_experience(
            session_id="test-size-1",
            query_text="Query one",
            plan_json=sample_plan,
            team_assignments="member_services_team",
        )
        experience_store.store_experience(
            session_id="test-size-2",
            query_text="Query two",
            plan_json=sample_plan,
            team_assignments="claims_services_team",
        )
        assert experience_store.get_collection_size() == 2


class TestExperienceRetrieval:
    """Test retrieval quality and formatting."""

    def test_retrieval_returns_formatted_examples(self, experience_store, sample_plan):
        """Retrieved experiences should be formatted for prompt injection."""
        experience_store.store_experience(
            session_id="test-format-001",
            query_text="What is the payment info for claim CLM-555?",
            plan_json=sample_plan,
            team_assignments="claims_services_team",
        )

        result = experience_store.retrieve_similar_experiences(
            query_text="Get payment details for claim CLM-666",
            top_k=1,
        )
        # Should contain example numbering
        assert "Example 1:" in result
        # Should contain the query
        assert "CLM-555" in result
        # Should contain team info
        assert "claims_services_team" in result

    def test_top_k_limits_results(self, experience_store, sample_plan):
        """top_k should limit the number of returned experiences."""
        for i in range(5):
            experience_store.store_experience(
                session_id=f"test-topk-{i}",
                query_text=f"Query about claims number {i}",
                plan_json=sample_plan,
                team_assignments="claims_services_team",
            )

        result = experience_store.retrieve_similar_experiences(
            query_text="Query about claims",
            top_k=2,
        )
        # Should have at most 2 examples
        assert result.count("Example") <= 2


class TestGracefulDegradation:
    """Test that failures don't crash the system."""

    def test_retrieve_with_connection_error(self):
        """Retrieval should return empty string on connection failure."""
        from databases.chroma_experience_store import ChromaExperienceStore

        mock_conn = MagicMock()
        mock_conn.get_or_create_collection.side_effect = Exception("Connection refused")

        store = ChromaExperienceStore(chroma_conn=mock_conn)
        result = store.retrieve_similar_experiences("test query")
        assert result == ""

    def test_store_with_connection_error(self):
        """Store should return False on connection failure."""
        from databases.chroma_experience_store import ChromaExperienceStore

        mock_conn = MagicMock()
        mock_conn.get_or_create_collection.side_effect = Exception("Connection refused")

        store = ChromaExperienceStore(chroma_conn=mock_conn)
        result = store.store_experience(
            session_id="fail", query_text="fail",
            plan_json="{}", team_assignments="",
        )
        assert result is False
