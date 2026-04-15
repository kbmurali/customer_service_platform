#!/usr/bin/env python3
"""
Seed LangFuse with the current hardcoded prompts from CSIP source files.
Run once after initial deployment (or after adding new prompts) to
register each prompt in LangFuse under the 'production' label.
Subsequent updates should be made through the LangFuse UI or API so
that versioning, rollback, and A/B testing are available.
Usage::
    python data/seed_langfuse_prompts.py
Environment variables required:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST   (defaults to http://langfuse:3000)
The script is idempotent: running it again updates the prompt text if it
has changed, preserving the existing version history.

NOTE: This script extracts prompt strings by parsing source files with
Python's ast module rather than importing the modules directly. This
avoids triggering transitive dependency chains (langchain, pymysql,
neo4j, redis, etc.) that may not be installed in the environment where
the script is run.
"""
from __future__ import annotations
import ast
import logging
import os
import sys
from pathlib import Path

# Add project root to path so imports work when run from any directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST-based prompt extraction (no imports, no dependency chains)
# ---------------------------------------------------------------------------

def _extract_string_constant(filepath: Path, variable_name: str):
    """
    Parse a Python source file with the ast module and extract a
    module-level string constant by variable name.

    Handles:
      - Simple string assignment: PROMPT = "..."
      - Parenthesized multi-line strings: PROMPT = ("..." "...")
      - Triple-quoted strings

    Returns None if the file doesn't exist or the variable is not found.
    """
    if not filepath.exists():
        return None
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as exc:
        logger.warning("Failed to parse %s: %s", filepath, exc)
        return None

    for node in ast.iter_child_nodes(tree):
        # Match: VARIABLE = "string" or VARIABLE = ("str1" "str2")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    try:
                        value = ast.literal_eval(node.value)
                        if isinstance(value, str):
                            return value
                    except (ValueError, TypeError):
                        pass
    return None


# ---------------------------------------------------------------------------
# Prompt registry — maps LangFuse prompt name -> (source file, variable name)
# ---------------------------------------------------------------------------

_AGENTS = Path("agents")
_TEAMS = _AGENTS / "teams"

PROMPT_REGISTRY = [
    # -- Central Supervisor ------------------------------------------------
    ("csip-central-planning-prompt",
     _AGENTS / "central_supervisor.py", "PLANNING_SYSTEM_PROMPT"),
    ("csip-central-routing-prompt",
     _AGENTS / "central_supervisor.py", "SUPERVISOR_SYSTEM_PROMPT"),
    ("csip-consolidation-system-prompt",
     _AGENTS / "central_supervisor.py", "CONSOLIDATION_SYSTEM_PROMPT"),
    ("csip-consolidation-user-prompt",
     _AGENTS / "central_supervisor.py", "CONSOLIDATION_USER_PROMPT"),

    # -- Claims Services ---------------------------------------------------
    ("csip-claims-planning-prompt",
     _TEAMS / "claims_services" / "supervisor" / "claims_services_supervisor.py",
     "_PLANNING_PROMPT_TEXT"),
    ("csip-claims-routing-prompt",
     _TEAMS / "claims_services" / "supervisor" / "claims_services_supervisor.py",
     "_ROUTING_PROMPT_TEXT"),
    ("csip-claims-lookup-worker-prompt",
     _TEAMS / "claims_services" / "supervisor" / "claim_lookup_worker.py",
     "WORKER_PROMPT"),
    ("csip-claims-status-worker-prompt",
     _TEAMS / "claims_services" / "supervisor" / "claim_status_worker.py",
     "WORKER_PROMPT"),
    ("csip-claims-payment-worker-prompt",
     _TEAMS / "claims_services" / "supervisor" / "claim_payment_info_worker.py",
     "WORKER_PROMPT"),
    ("csip-claims-update-status-worker-prompt",
     _TEAMS / "claims_services" / "supervisor" / "update_claim_status_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-claims-worker-prompt",
     _TEAMS / "claims_services" / "supervisor" / "member_claims_worker.py",
     "WORKER_PROMPT"),
    ("csip-claims-adjudication-worker-prompt",
     _TEAMS / "claims_services" / "supervisor" / "claim_adjudication_worker.py",
     "WORKER_PROMPT"),

    # -- Member Services ---------------------------------------------------
    ("csip-member-planning-prompt",
     _TEAMS / "member_services" / "supervisor" / "member_services_supervisor.py",
     "_PLANNING_PROMPT_TEXT"),
    ("csip-member-routing-prompt",
     _TEAMS / "member_services" / "supervisor" / "member_services_supervisor.py",
     "_ROUTING_PROMPT_TEXT"),
    ("csip-member-lookup-worker-prompt",
     _TEAMS / "member_services" / "supervisor" / "member_lookup_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-eligibility-worker-prompt",
     _TEAMS / "member_services" / "supervisor" / "check_eligibility_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-coverage-worker-prompt",
     _TEAMS / "member_services" / "supervisor" / "coverage_lookup_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-update-worker-prompt",
     _TEAMS / "member_services" / "supervisor" / "update_member_info_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-policy-lookup-worker-prompt",
     _TEAMS / "member_services" / "supervisor" / "member_policy_lookup_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-treatment-history-worker-prompt",
     _TEAMS / "member_services" / "supervisor" / "treatment_history_worker.py",
     "WORKER_PROMPT"),

    # -- PA Services -------------------------------------------------------
    ("csip-pa-planning-prompt",
     _TEAMS / "pa_services" / "supervisor" / "pa_services_supervisor.py",
     "_PLANNING_PROMPT_TEXT"),
    ("csip-pa-routing-prompt",
     _TEAMS / "pa_services" / "supervisor" / "pa_services_supervisor.py",
     "_ROUTING_PROMPT_TEXT"),
    ("csip-pa-lookup-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "pa_lookup_worker.py",
     "WORKER_PROMPT"),
    ("csip-pa-status-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "pa_status_worker.py",
     "WORKER_PROMPT"),
    ("csip-pa-requirements-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "pa_requirements_worker.py",
     "WORKER_PROMPT"),
    ("csip-pa-approve-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "approve_prior_auth_worker.py",
     "WORKER_PROMPT"),
    ("csip-pa-deny-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "deny_prior_auth_worker.py",
     "WORKER_PROMPT"),
    ("csip-member-prior-auth-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "member_prior_auth_worker.py",
     "WORKER_PROMPT"),
    ("csip-pa-recommendation-worker-prompt",
     _TEAMS / "pa_services" / "supervisor" / "pa_recommendation_worker.py",
     "WORKER_PROMPT"),

    # -- Provider Services -------------------------------------------------
    ("csip-provider-planning-prompt",
     _TEAMS / "provider_services" / "supervisor" / "provider_services_supervisor.py",
     "_PLANNING_PROMPT_TEXT"),
    ("csip-provider-routing-prompt",
     _TEAMS / "provider_services" / "supervisor" / "provider_services_supervisor.py",
     "_ROUTING_PROMPT_TEXT"),
    ("csip-provider-lookup-worker-prompt",
     _TEAMS / "provider_services" / "supervisor" / "provider_lookup_worker.py",
     "WORKER_PROMPT"),
    ("csip-provider-network-worker-prompt",
     _TEAMS / "provider_services" / "supervisor" / "provider_network_check_worker.py",
     "WORKER_PROMPT"),
    ("csip-provider-search-worker-prompt",
     _TEAMS / "provider_services" / "supervisor" / "provider_search_by_specialty_worker.py",
     "WORKER_PROMPT"),

    # -- Search Services ---------------------------------------------------
    ("csip-search-planning-prompt",
     _TEAMS / "search_services" / "supervisor" / "search_services_supervisor.py",
     "_PLANNING_PROMPT_TEXT"),
    ("csip-search-routing-prompt",
     _TEAMS / "search_services" / "supervisor" / "search_services_supervisor.py",
     "_ROUTING_PROMPT_TEXT"),
    ("csip-search-knowledge-worker-prompt",
     _TEAMS / "search_services" / "supervisor" / "search_knowledge_base_worker.py",
     "WORKER_PROMPT"),
    ("csip-search-medical-codes-worker-prompt",
     _TEAMS / "search_services" / "supervisor" / "search_medical_codes_worker.py",
     "WORKER_PROMPT"),
    ("csip-search-policy-worker-prompt",
     _TEAMS / "search_services" / "supervisor" / "search_policy_info_worker.py",
     "WORKER_PROMPT"),
]


# ---------------------------------------------------------------------------
# LangFuse client and seeding
# ---------------------------------------------------------------------------

def _get_langfuse_client():
    from langfuse import Langfuse
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host       = os.getenv("LANGFUSE_HOST", "http://langfuse:3000")
    if not public_key or not secret_key:
        raise EnvironmentError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set"
        )
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def _seed_prompt(lf, name, prompt, label="production"):
    """Create or update a prompt in LangFuse."""
    try:
        lf.create_prompt(
            name=name,
            prompt=prompt,
            labels=[label],
            config={"type": "csip_system_prompt"},
        )
        logger.info("Seeded: %-45s  [label=%s]", name, label)
    except Exception as exc:
        logger.error("Failed to seed '%s': %s", name, exc)


def main():
    # Load environment variables
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv())
    except ImportError:
        pass

    lf = _get_langfuse_client()
    logger.info("Connected to LangFuse — seeding prompts...")

    seeded = 0
    skipped = 0
    not_found = 0

    for prompt_name, rel_path, variable_name in PROMPT_REGISTRY:
        filepath = _PROJECT_ROOT / rel_path
        prompt_text = _extract_string_constant(filepath, variable_name)

        if prompt_text is None:
            if not filepath.exists():
                logger.warning(
                    "Skipped: %-45s  (file not found: %s)", prompt_name, rel_path
                )
                not_found += 1
            else:
                logger.warning(
                    "Skipped: %-45s  (variable '%s' not found in %s)",
                    prompt_name, variable_name, rel_path,
                )
                skipped += 1
            continue

        _seed_prompt(lf, prompt_name, prompt_text)
        seeded += 1

    lf.flush()
    logger.info(
        "Seeding complete: %d seeded, %d variable-not-found, %d file-not-found.",
        seeded, skipped, not_found,
    )
    if not_found > 0:
        logger.info(
            "Files not found are expected if running from a partial source tree "
            "(e.g., a single team's container). To seed all prompts, run from "
            "the full project root."
        )


if __name__ == "__main__":
    main()
