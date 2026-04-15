#!/usr/bin/env python3
"""
Purge all CSIP prompts from LangFuse and re-seed from current source.

Usage:
    # From inside ANY CSIP container (e.g. agentic-access-api):
    docker exec -it <container> python3 data/purge_and_reseed_langfuse_prompts.py

    # Or from the host with env vars set:
    LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-... LANGFUSE_HOST=http://localhost:3000 \
        python3 data/purge_and_reseed_langfuse_prompts.py

This script:
  1. Lists all prompts in LangFuse matching the "csip-" prefix
  2. Deletes each one (all versions)
  3. Re-runs the seed logic to create fresh v1 prompts from current source

This is necessary after updating hardcoded prompt constants (e.g. adding
new rules to planning prompts) because LangFuse prompt versioning serves
the stored version, not the Python module constant.
"""

from __future__ import annotations

import logging
import os
import sys
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# All CSIP prompt names registered by seed_langfuse_prompts.py
CSIP_PROMPT_NAMES = [
    # Central
    "csip-central-planning-prompt",
    "csip-central-routing-prompt",
    # Claims
    "csip-claims-planning-prompt",
    "csip-claims-routing-prompt",
    "csip-claims-lookup-worker-prompt",
    "csip-claims-status-worker-prompt",
    "csip-claims-payment-worker-prompt",
    "csip-claims-update-status-worker-prompt",
    "csip-claims-adjudication-worker-prompt",
    # Member
    "csip-member-planning-prompt",
    "csip-member-routing-prompt",
    "csip-member-lookup-worker-prompt",
    "csip-member-eligibility-worker-prompt",
    "csip-member-coverage-worker-prompt",
    "csip-member-update-worker-prompt",
    # PA
    "csip-pa-planning-prompt",
    "csip-pa-routing-prompt",
    "csip-pa-lookup-worker-prompt",
    "csip-pa-status-worker-prompt",
    "csip-pa-requirements-worker-prompt",
    "csip-pa-approve-worker-prompt",
    "csip-pa-deny-worker-prompt",
    "csip-pa-recommendation-worker-prompt",
    # Provider
    "csip-provider-planning-prompt",
    "csip-provider-routing-prompt",
    "csip-provider-lookup-worker-prompt",
    "csip-provider-network-worker-prompt",
    "csip-provider-search-worker-prompt",
    # Search
    "csip-search-planning-prompt",
    "csip-search-routing-prompt",
    "csip-search-knowledge-worker-prompt",
    "csip-search-medical-codes-worker-prompt",
    "csip-search-policy-worker-prompt",
]


def _read_secret(env_var: str, secret_path: str) -> str:
    """Read from env var first, then fall back to Docker Swarm secret file."""
    val = os.getenv(env_var, "")
    if val:
        return val
    try:
        with open(secret_path) as f:
            return f.read().strip()
    except Exception:
        return ""


def purge_prompts():
    """Delete all CSIP prompts from LangFuse via the public API."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host       = os.getenv("LANGFUSE_HOST", "http://langfuse:3000")

    if not public_key or not secret_key:
        logger.error("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set")
        sys.exit(1)

    logger.info("Purging %d CSIP prompts from LangFuse at %s ...", len(CSIP_PROMPT_NAMES), host)
    deleted = 0
    not_found = 0
    errors = 0

    with httpx.Client(timeout=10.0) as client:
        for name in CSIP_PROMPT_NAMES:
            try:
                resp = client.delete(
                    f"{host}/api/public/v2/prompts/{name}",
                    auth=(public_key, secret_key),
                )
                if resp.status_code in (200, 204):
                    logger.info("  Deleted: %s", name)
                    deleted += 1
                elif resp.status_code == 404:
                    logger.info("  Not found (skip): %s", name)
                    not_found += 1
                else:
                    logger.warning("  Unexpected %d for %s: %s", resp.status_code, name, resp.text[:200])
                    errors += 1
            except Exception as exc:
                logger.error("  Error deleting %s: %s", name, exc)
                errors += 1

    logger.info("Purge complete: %d deleted, %d not found, %d errors", deleted, not_found, errors)
    return errors == 0


def reseed_prompts():
    """Re-seed all prompts from current Python source constants."""
    logger.info("Re-seeding prompts from current source...")
    try:
        from seed_langfuse_prompts import main as seed_main
        seed_main()
    except ImportError:
        # Try alternate import path
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
            from seed_langfuse_prompts import main as seed_main
            seed_main()
        except ImportError as exc:
            logger.error("Cannot import seed_langfuse_prompts: %s", exc)
            logger.info("Falling back to inline seeding...")
            _inline_seed()


def _inline_seed():
    """Inline seed as fallback if seed_langfuse_prompts.py can't be imported."""
    from langfuse import Langfuse

    public_key = _read_secret("LANGFUSE_PUBLIC_KEY", "/run/secrets/LANGFUSE_PUBLIC_KEY")
    secret_key = _read_secret("LANGFUSE_SECRET_KEY", "/run/secrets/LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://langfuse:3000")

    lf = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    seeded = 0

    def _seed(name, prompt):
        nonlocal seeded
        try:
            lf.create_prompt(name=name, prompt=prompt, labels=["production"],
                             config={"type": "csip_system_prompt"})
            logger.info("  Seeded: %s", name)
            seeded += 1
        except Exception as exc:
            logger.error("  Failed: %s — %s", name, exc)

    # Central
    try:
        from agents.central_supervisor import PLANNING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
        _seed("csip-central-planning-prompt", PLANNING_SYSTEM_PROMPT)
        _seed("csip-central-routing-prompt", SUPERVISOR_SYSTEM_PROMPT)
    except ImportError as e:
        logger.warning("Central prompts skipped: %s", e)

    # Claims
    try:
        from agents.teams.claims_services.claims_services_supervisor import (
            _PLANNING_PROMPT_TEXT as cp, _ROUTING_PROMPT_TEXT as cr)
        _seed("csip-claims-planning-prompt", cp)
        _seed("csip-claims-routing-prompt", cr)
    except (ImportError, AttributeError) as e:
        logger.warning("Claims supervisor prompts skipped: %s", e)

    # Member
    try:
        from agents.teams.member_services.member_services_supervisor import (
            _PLANNING_PROMPT_TEXT as mp, _ROUTING_PROMPT_TEXT as mr)
        _seed("csip-member-planning-prompt", mp)
        _seed("csip-member-routing-prompt", mr)
    except (ImportError, AttributeError) as e:
        logger.warning("Member supervisor prompts skipped: %s", e)

    # PA
    try:
        from agents.teams.pa_services.pa_services_supervisor import (
            _PLANNING_PROMPT_TEXT as pp, _ROUTING_PROMPT_TEXT as pr)
        _seed("csip-pa-planning-prompt", pp)
        _seed("csip-pa-routing-prompt", pr)
    except (ImportError, AttributeError) as e:
        logger.warning("PA supervisor prompts skipped: %s", e)

    # Provider
    try:
        from agents.teams.provider_services.provider_services_supervisor import (
            _PLANNING_PROMPT_TEXT as pvp, _ROUTING_PROMPT_TEXT as pvr)
        _seed("csip-provider-planning-prompt", pvp)
        _seed("csip-provider-routing-prompt", pvr)
    except (ImportError, AttributeError) as e:
        logger.warning("Provider supervisor prompts skipped: %s", e)

    # Search
    try:
        from agents.teams.search_services.search_services_supervisor import (
            _PLANNING_PROMPT_TEXT as sp, _ROUTING_PROMPT_TEXT as sr)
        _seed("csip-search-planning-prompt", sp)
        _seed("csip-search-routing-prompt", sr)
    except (ImportError, AttributeError) as e:
        logger.warning("Search supervisor prompts skipped: %s", e)

    lf.flush()
    logger.info("Inline seeding complete: %d prompts seeded", seeded)


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv())
    except ImportError:
        pass
    
    logger.info("=" * 60)
    logger.info("CSIP LangFuse Prompt Purge & Re-seed")
    logger.info("=" * 60)

    success = purge_prompts()

    logger.info("")
    logger.info("-" * 60)

    reseed_prompts()

    logger.info("")
    logger.info("=" * 60)
    logger.info("IMPORTANT: Restart all CSIP containers after re-seeding!")
    logger.info("Supervisors cache prompts at __init__ time via @lru_cache.")
    logger.info("  docker service update --force health_insurance_agentic-access-api")
    logger.info("  docker service update --force health_insurance_claims-services-a2a-server")
    logger.info("  docker service update --force health_insurance_member-services-a2a-server")
    logger.info("  docker service update --force health_insurance_pa-services-a2a-server")
    logger.info("  docker service update --force health_insurance_provider-services-a2a-server")
    logger.info("  docker service update --force health_insurance_search-services-a2a-server")
    logger.info("=" * 60)
