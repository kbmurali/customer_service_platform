"""
Seed Experience Collections — bootstraps the ``successful_experiences``
Chroma collection with manually curated examples for cold start.

Run once during initial setup (after Chroma is healthy)::

    python3 data/seed_experience_collections.py

The seed experiences provide few-shot planning examples from day one,
before any real CSR feedback has been collected.  As genuine feedback
accumulates, the seed experiences are gradually outnumbered by real
production experiences.

Idempotent — skips existing records by session_id.
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Add project root to path so imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data — 25 curated experiences covering all 5 teams
# ---------------------------------------------------------------------------

SEED_EXPERIENCES = [
    # ── Member Services (5) ───────────────────────────────────────────
    {
        "session_id": "seed-member-001",
        "query": "What is the eligibility status for member M-12345?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check member eligibility", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1}],
        },
        "teams": "member_services_team",
    },
    {
        "session_id": "seed-member-002",
        "query": "Look up member M-67890 and show me their dependents",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Retrieve member details and dependents", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1}],
        },
        "teams": "member_services_team",
    },
    {
        "session_id": "seed-member-003",
        "query": "What is the coverage for member M-11111 under their current policy?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Retrieve member coverage details", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1}],
        },
        "teams": "member_services_team",
    },
    {
        "session_id": "seed-member-004",
        "query": "What are the deductible and copay amounts for member M-22222?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Retrieve member cost-sharing details", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1}],
        },
        "teams": "member_services_team",
    },
    {
        "session_id": "seed-member-005",
        "query": "Update the contact phone number for member M-33333",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Update member contact information", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1}],
        },
        "teams": "member_services_team",
    },

    # ── Claims Services (5) ───────────────────────────────────────────
    {
        "session_id": "seed-claims-001",
        "query": "What is the status of claim CLM-123456?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check claim processing status", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "claims_services_team", "order": 1}],
        },
        "teams": "claims_services_team",
    },
    {
        "session_id": "seed-claims-002",
        "query": "What are the payment details for claim CLM-789012?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Retrieve claim payment information", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "claims_services_team", "order": 1}],
        },
        "teams": "claims_services_team",
    },
    {
        "session_id": "seed-claims-003",
        "query": "Why was claim CLM-456789 denied?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Retrieve claim details including denial reason", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "claims_services_team", "order": 1}],
        },
        "teams": "claims_services_team",
    },
    {
        "session_id": "seed-claims-004",
        "query": "Get the status and payment info for claim CLM-111222",
        "plan": {
            "goals": [
                {"id": "goal_1", "description": "Check claim status", "priority": 1},
                {"id": "goal_2", "description": "Retrieve claim payment details", "priority": 2},
            ],
            "steps": [
                {"step_id": "step_1", "goal_id": "goal_1", "agent": "claims_services_team", "order": 1},
                {"step_id": "step_2", "goal_id": "goal_2", "agent": "claims_services_team", "order": 2},
            ],
        },
        "teams": "claims_services_team",
    },
    {
        "session_id": "seed-claims-005",
        "query": "Update the status of claim CLM-333444 to approved with reason 'documentation complete'",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Update claim status to approved", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "claims_services_team", "order": 1}],
        },
        "teams": "claims_services_team",
    },

    # ── PA Services (5) ───────────────────────────────────────────────
    {
        "session_id": "seed-pa-001",
        "query": "Does CPT 29881 require prior authorization under a PPO plan?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check PA requirements for CPT code under PPO", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "pa_services_team", "order": 1}],
        },
        "teams": "pa_services_team",
    },
    {
        "session_id": "seed-pa-002",
        "query": "What is the status of prior authorization PA-2024-001?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check PA status", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "pa_services_team", "order": 1}],
        },
        "teams": "pa_services_team",
    },
    {
        "session_id": "seed-pa-003",
        "query": "Approve prior authorization PA-2024-005 for member M-44444",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Approve pending prior authorization", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "pa_services_team", "order": 1}],
        },
        "teams": "pa_services_team",
    },
    {
        "session_id": "seed-pa-004",
        "query": "What prior authorizations are pending for member M-55555?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "List pending PAs for member", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "pa_services_team", "order": 1}],
        },
        "teams": "pa_services_team",
    },
    {
        "session_id": "seed-pa-005",
        "query": "Does knee replacement surgery need prior auth under an HMO plan?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check PA requirement for procedure under HMO", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "pa_services_team", "order": 1}],
        },
        "teams": "pa_services_team",
    },

    # ── Provider Services (5) ─────────────────────────────────────────
    {
        "session_id": "seed-provider-001",
        "query": "Is Dr. Chen (NPI 1234567890) in-network for member M-12345?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check provider network status", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "provider_services_team", "order": 1}],
        },
        "teams": "provider_services_team",
    },
    {
        "session_id": "seed-provider-002",
        "query": "Find in-network cardiologists near ZIP 60601",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Search for in-network cardiologists by location", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "provider_services_team", "order": 1}],
        },
        "teams": "provider_services_team",
    },
    {
        "session_id": "seed-provider-003",
        "query": "What are the details for provider with NPI 9876543210?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Retrieve provider details by NPI", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "provider_services_team", "order": 1}],
        },
        "teams": "provider_services_team",
    },
    {
        "session_id": "seed-provider-004",
        "query": "Has provider NPI 1111222233 serviced any claims under policy POL-GOLD-001?",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Check provider claim history under policy", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "provider_services_team", "order": 1}],
        },
        "teams": "provider_services_team",
    },
    {
        "session_id": "seed-provider-005",
        "query": "Find orthopedic surgeons accepting new patients near ZIP 90210",
        "plan": {
            "goals": [{"id": "goal_1", "description": "Search for orthopedic providers by location", "priority": 1}],
            "steps": [{"step_id": "step_1", "goal_id": "goal_1", "agent": "provider_services_team", "order": 1}],
        },
        "teams": "provider_services_team",
    },

    # ── Multi-Team Queries (5) ────────────────────────────────────────
    {
        "session_id": "seed-multi-001",
        "query": "Is Dr. Chen in-network for this member, and why was their claim denied?",
        "plan": {
            "goals": [
                {"id": "goal_1", "description": "Check provider network status", "priority": 1},
                {"id": "goal_2", "description": "Retrieve claim denial reason", "priority": 2},
            ],
            "steps": [
                {"step_id": "step_1", "goal_id": "goal_1", "agent": "provider_services_team", "order": 1},
                {"step_id": "step_2", "goal_id": "goal_2", "agent": "claims_services_team", "order": 1},
            ],
        },
        "teams": "provider_services_team,claims_services_team",
    },
    {
        "session_id": "seed-multi-002",
        "query": "Does CPT 29881 need prior auth under a PPO, and what is the procedure code for knee replacement?",
        "plan": {
            "goals": [
                {"id": "goal_1", "description": "Check PA requirements for CPT code", "priority": 1},
                {"id": "goal_2", "description": "Look up procedure code for knee replacement", "priority": 2},
            ],
            "steps": [
                {"step_id": "step_1", "goal_id": "goal_1", "agent": "pa_services_team", "order": 1},
                {"step_id": "step_2", "goal_id": "goal_2", "agent": "search_services_team", "order": 1},
            ],
        },
        "teams": "pa_services_team,search_services_team",
    },
    {
        "session_id": "seed-multi-003",
        "query": "Check eligibility for member M-12345 and get the status of their claim CLM-123456",
        "plan": {
            "goals": [
                {"id": "goal_1", "description": "Check member eligibility", "priority": 1},
                {"id": "goal_2", "description": "Check claim status", "priority": 2},
            ],
            "steps": [
                {"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1},
                {"step_id": "step_2", "goal_id": "goal_2", "agent": "claims_services_team", "order": 2},
            ],
        },
        "teams": "member_services_team,claims_services_team",
    },
    {
        "session_id": "seed-multi-004",
        "query": "What does the PPO policy say about out-of-network mental health coverage, and find in-network psychiatrists near 60601?",
        "plan": {
            "goals": [
                {"id": "goal_1", "description": "Look up PPO mental health coverage policy", "priority": 1},
                {"id": "goal_2", "description": "Search for in-network psychiatrists by location", "priority": 2},
            ],
            "steps": [
                {"step_id": "step_1", "goal_id": "goal_1", "agent": "search_services_team", "order": 1},
                {"step_id": "step_2", "goal_id": "goal_2", "agent": "provider_services_team", "order": 1},
            ],
        },
        "teams": "search_services_team,provider_services_team",
    },
    {
        "session_id": "seed-multi-005",
        "query": "Check member M-99999 eligibility, then tell me if their PA for CPT 27447 was approved",
        "plan": {
            "goals": [
                {"id": "goal_1", "description": "Check member eligibility", "priority": 1},
                {"id": "goal_2", "description": "Check PA status for CPT code", "priority": 2},
            ],
            "steps": [
                {"step_id": "step_1", "goal_id": "goal_1", "agent": "member_services_team", "order": 1},
                {"step_id": "step_2", "goal_id": "goal_2", "agent": "pa_services_team", "order": 2},
            ],
        },
        "teams": "member_services_team,pa_services_team",
    },
]


def seed_experiences() -> int:
    """
    Populate the experience store with curated seed data.

    Returns the number of new experiences stored (skips existing ones).
    """
    from databases.chroma_experience_store import get_experience_store

    store = get_experience_store()
    stored = 0

    for exp in SEED_EXPERIENCES:
        session_id = exp["session_id"]

        success = store.store_experience(
            session_id=session_id,
            query_text=exp["query"],
            plan_json=json.dumps(exp["plan"]),
            team_assignments=exp["teams"],
            result_summary="Seed experience — curated example",
            rating="correct",
        )
        if success:
            stored += 1

    logger.info("Seeded %d / %d experiences", stored, len(SEED_EXPERIENCES))
    return stored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv())
    except ImportError:
        pass

    count = seed_experiences()
    from databases.chroma_experience_store import ChromaExperienceStore
    print(f"Seeded {count} experiences into Chroma collection "
          f"'{ChromaExperienceStore.COLLECTION_NAME}'")
    sys.exit(0)
