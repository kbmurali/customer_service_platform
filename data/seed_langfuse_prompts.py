#!/usr/bin/env python3
"""
Seed LangFuse with the current hardcoded prompts from CSIP source files.

Run once after initial deployment (or after adding new prompts) to
register each prompt in LangFuse under the 'production' label.
Subsequent updates should be made through the LangFuse UI or API so
that versioning, rollback, and A/B testing are available.

Usage::

    python scripts/seed_langfuse_prompts.py

Environment variables required:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST   (defaults to http://langfuse:3000)

The script is idempotent: running it again updates the prompt text if it
has changed, preserving the existing version history.
"""

from __future__ import annotations

import logging
import os
import sys

# Add project root to path so imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def _seed_prompt(lf, name: str, prompt: str, label: str = "production") -> None:
    """Create or update a prompt in LangFuse."""
    try:
        lf.create_prompt(
            name=name,
            prompt=prompt,
            labels=[label],
            config={"type": "csip_system_prompt"},
        )
        logger.info("Seeded: %s  [label=%s]", name, label)
    except Exception as exc:
        logger.error("Failed to seed '%s': %s", name, exc)


def main() -> None:
    lf = _get_langfuse_client()
    logger.info("Connected to LangFuse — seeding prompts...")

    # ── Central Supervisor ──────────────────────────────────────────────
    from agents.central_supervisor import PLANNING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
    _seed_prompt(lf, "csip-central-planning-prompt",  PLANNING_SYSTEM_PROMPT)
    _seed_prompt(lf, "csip-central-routing-prompt",   SUPERVISOR_SYSTEM_PROMPT)

    # ── Claims Services ─────────────────────────────────────────────────
    from agents.teams.claims_services.claims_services_supervisor import (
        _PLANNING_PROMPT_TEXT as claims_planning,
        _ROUTING_PROMPT_TEXT  as claims_routing,
    )
    _seed_prompt(lf, "csip-claims-planning-prompt", claims_planning)
    _seed_prompt(lf, "csip-claims-routing-prompt",  claims_routing)

    from agents.teams.claims_services.supervisor.claim_lookup_worker      import WORKER_PROMPT as clm_lookup_p
    from agents.teams.claims_services.supervisor.claim_status_worker      import WORKER_PROMPT as clm_status_p
    from agents.teams.claims_services.supervisor.claim_payment_info_worker import WORKER_PROMPT as clm_payment_p
    from agents.teams.claims_services.supervisor.update_claim_status_worker import WORKER_PROMPT as clm_update_p
    _seed_prompt(lf, "csip-claims-lookup-worker-prompt",        clm_lookup_p)
    _seed_prompt(lf, "csip-claims-status-worker-prompt",        clm_status_p)
    _seed_prompt(lf, "csip-claims-payment-worker-prompt",       clm_payment_p)
    _seed_prompt(lf, "csip-claims-update-status-worker-prompt", clm_update_p)

    # ── Member Services ─────────────────────────────────────────────────
    from agents.teams.member_services.member_services_supervisor import (
        _PLANNING_PROMPT_TEXT as ms_planning,
        _ROUTING_PROMPT_TEXT  as ms_routing,
    )
    _seed_prompt(lf, "csip-member-planning-prompt", ms_planning)
    _seed_prompt(lf, "csip-member-routing-prompt",  ms_routing)

    from agents.teams.member_services.supervisor.member_lookup_worker     import WORKER_PROMPT as ml_p
    from agents.teams.member_services.supervisor.check_eligibility_worker import WORKER_PROMPT as el_p
    from agents.teams.member_services.supervisor.coverage_lookup_worker   import WORKER_PROMPT as cl_p
    from agents.teams.member_services.supervisor.update_member_info_worker import WORKER_PROMPT as um_p
    _seed_prompt(lf, "csip-member-lookup-worker-prompt",      ml_p)
    _seed_prompt(lf, "csip-member-eligibility-worker-prompt", el_p)
    _seed_prompt(lf, "csip-member-coverage-worker-prompt",    cl_p)
    _seed_prompt(lf, "csip-member-update-worker-prompt",      um_p)

    # ── PA Services ─────────────────────────────────────────────────────
    from agents.teams.pa_services.pa_services_supervisor import (
        _PLANNING_PROMPT_TEXT as pa_planning,
        _ROUTING_PROMPT_TEXT  as pa_routing,
    )
    _seed_prompt(lf, "csip-pa-planning-prompt", pa_planning)
    _seed_prompt(lf, "csip-pa-routing-prompt",  pa_routing)

    from agents.teams.pa_services.supervisor.pa_lookup_worker         import WORKER_PROMPT as pal_p
    from agents.teams.pa_services.supervisor.pa_status_worker         import WORKER_PROMPT as pas_p
    from agents.teams.pa_services.supervisor.pa_requirements_worker   import WORKER_PROMPT as par_p
    from agents.teams.pa_services.supervisor.approve_prior_auth_worker import WORKER_PROMPT as paa_p
    from agents.teams.pa_services.supervisor.deny_prior_auth_worker    import WORKER_PROMPT as pad_p
    _seed_prompt(lf, "csip-pa-lookup-worker-prompt",       pal_p)
    _seed_prompt(lf, "csip-pa-status-worker-prompt",       pas_p)
    _seed_prompt(lf, "csip-pa-requirements-worker-prompt", par_p)
    _seed_prompt(lf, "csip-pa-approve-worker-prompt",      paa_p)
    _seed_prompt(lf, "csip-pa-deny-worker-prompt",         pad_p)

    # ── Provider Services ───────────────────────────────────────────────
    from agents.teams.provider_services.provider_services_supervisor import (
        _PLANNING_PROMPT_TEXT as prov_planning,
        _ROUTING_PROMPT_TEXT  as prov_routing,
    )
    _seed_prompt(lf, "csip-provider-planning-prompt", prov_planning)
    _seed_prompt(lf, "csip-provider-routing-prompt",  prov_routing)

    from agents.teams.provider_services.supervisor.provider_lookup_worker              import WORKER_PROMPT as pvl_p
    from agents.teams.provider_services.supervisor.provider_network_check_worker       import WORKER_PROMPT as pvn_p
    from agents.teams.provider_services.supervisor.provider_search_by_specialty_worker import WORKER_PROMPT as pvs_p
    _seed_prompt(lf, "csip-provider-lookup-worker-prompt",   pvl_p)
    _seed_prompt(lf, "csip-provider-network-worker-prompt",  pvn_p)
    _seed_prompt(lf, "csip-provider-search-worker-prompt",   pvs_p)


    # ── Search Services ─────────────────────────────────────────────────
    from agents.teams.search_services.search_services_supervisor import (
        _PLANNING_PROMPT_TEXT as search_planning,
        _ROUTING_PROMPT_TEXT  as search_routing,
    )
    _seed_prompt(lf, "csip-search-planning-prompt", search_planning)
    _seed_prompt(lf, "csip-search-routing-prompt",  search_routing)

    from agents.teams.search_services.supervisor.search_knowledge_base_worker import WORKER_PROMPT as skb_p
    from agents.teams.search_services.supervisor.search_medical_codes_worker  import WORKER_PROMPT as smc_p
    from agents.teams.search_services.supervisor.search_policy_info_worker    import WORKER_PROMPT as spi_p
    _seed_prompt(lf, "csip-search-knowledge-worker-prompt",    skb_p)
    _seed_prompt(lf, "csip-search-medical-codes-worker-prompt", smc_p)
    _seed_prompt(lf, "csip-search-policy-worker-prompt",        spi_p)

    lf.flush()
    logger.info("All prompts seeded successfully.")


if __name__ == "__main__":
    main()
