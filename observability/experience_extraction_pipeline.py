"""
Experience Extraction Pipeline — populates the Chroma experience store
from CSR-validated successful sessions.

Runs on a configurable schedule (invoked by evaluation_pipeline.py) and
extracts query→plan pairs from sessions rated ``'correct'``.  Each
extracted experience is stored in the ``successful_experiences`` Chroma
collection where the Central Supervisor retrieves it as a few-shot
planning example.

Flow:
  1. Query MySQL for ``correct``-rated sessions not yet processed.
  2. For each qualifying session, fetch the CG plan (goals, steps, teams).
  3. Extract the original query from the CG Session node.
  4. Store the experience in Chroma.
  5. Mark the session as processed in ``experience_extraction_log``.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics — imported from the central registry to avoid
# duplicate registration errors.
# ---------------------------------------------------------------------------
try:
    from observability.prometheus_metrics import (
        experiences_extracted_total,
        experience_store_size,
    )
    _PROM = True
except ImportError:
    _PROM = False


def _extract_first_human_query(conv_raw: str) -> str:
    """
    Extract the first human message content from conversationMessages.

    conversationMessages is stored as a JSON array of message objects:
        [{"type": "human", "data": {"content": "user query..."}}, ...]

    Returns the content string (truncated to 1000 chars to stay well
    within embedding model token limits), or empty string on failure.
    """
    if not conv_raw or not isinstance(conv_raw, str):
        return ""
    try:
        messages = json.loads(conv_raw)
        if not isinstance(messages, list):
            return ""
        for msg in messages:
            if msg.get("type") == "human":
                content = msg.get("data", {}).get("content", "")
                if content:
                    return content[:1000]
        return ""
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Fallback: if it's plain text (not JSON), take first 500 chars
        return conv_raw[:500]


def run_extraction(redis_client=None) -> int:
    """
    Extract experiences from unprocessed ``correct``-rated sessions.

    Returns the number of new experiences stored in this cycle.

    This function is intentionally defensive — failures in one session
    do not prevent processing of subsequent sessions.
    """
    extracted = 0

    try:
        from config.settings import get_settings
        settings = get_settings()

        if not settings.EXPERIENCE_STORE_ENABLED:
            logger.debug("ExperienceExtraction: disabled by EXPERIENCE_STORE_ENABLED=False")
            return 0

        from databases.feedback_data_access import get_feedback_data_access
        from databases.chroma_experience_store import get_experience_store
        from databases.context_graph_data_access import get_cg_data_access

        feedback_da = get_feedback_data_access()
        store = get_experience_store()
        cg = get_cg_data_access()

        # ── Step 1: Get unprocessed correct-rated sessions ────────────
        unprocessed = feedback_da.get_unprocessed_correct_sessions(
            limit=50,
        )

        if not unprocessed:
            logger.debug("ExperienceExtraction: no unprocessed correct sessions")
            _update_store_size(store)
            return 0

        logger.info(
            "ExperienceExtraction: found %d unprocessed correct sessions",
            len(unprocessed),
        )

        for row in unprocessed:
            session_id = row.get("session_id", "")
            if not session_id:
                continue

            try:
                # ── Step 2: Fetch the CG plan ─────────────────────────
                plan = cg.get_active_plan(session_id)
                if not plan:
                    logger.debug(
                        "ExperienceExtraction: no plan found for session=%s, skipping",
                        session_id,
                    )
                    feedback_da.mark_session_extracted(session_id, "skipped_no_plan")
                    continue

                goals = plan.get("goals", [])
                steps = plan.get("steps", [])

                if not goals or not steps:
                    feedback_da.mark_session_extracted(session_id, "skipped_empty_plan")
                    continue

                # ── Step 3: Extract the original query ────────────────
                session_info = cg.get_session(session_id)
                query_text = ""

                if session_info:
                    # conversationMessages is a JSON array of message
                    # objects. Parse it and extract the first human
                    # message's content — NOT the raw JSON string,
                    # which can exceed embedding model token limits.
                    conv = session_info.get("conversationMessages", "")
                    if conv:
                        query_text = _extract_first_human_query(conv)

                if not query_text:
                    # Fallback: use the first step's action field
                    for step in steps:
                        q = step.get("action", "") or step.get("query", "")
                        if q:
                            query_text = q[:1000]
                            break

                if not query_text:
                    feedback_da.mark_session_extracted(session_id, "skipped_no_query")
                    continue

                # ── Step 4: Build team assignments string ─────────────
                teams = set()
                for step in steps:
                    agent = step.get("agent", "") or step.get("worker", "")
                    if agent:
                        teams.add(agent)
                team_assignments = ",".join(sorted(teams))

                # ── Step 5: Build plan JSON for storage ───────────────
                plan_for_storage = {
                    "goals": [
                        {"id": g.get("id", ""), "description": g.get("description", ""),
                         "priority": g.get("priority", 1)}
                        for g in goals
                    ],
                    "steps": [
                        {"step_id": s.get("step_id", ""), "goal_id": s.get("goal_id", ""),
                         "agent": s.get("agent", "") or s.get("worker", ""),
                         "order": s.get("order", 1)}
                        for s in steps
                    ],
                }
                plan_json = json.dumps(plan_for_storage)

                # ── Step 6: Store in Chroma ───────────────────────────
                success = store.store_experience(
                    session_id=session_id,
                    query_text=query_text,
                    plan_json=plan_json,
                    team_assignments=team_assignments,
                    result_summary="",  # Could be enriched with tool results
                    rating="correct",
                )

                if success:
                    feedback_da.mark_session_extracted(
                        session_id, store.COLLECTION_NAME,
                    )
                    extracted += 1
                    if _PROM:
                        experiences_extracted_total.inc()

            except Exception as sess_exc:
                logger.error(
                    "ExperienceExtraction: failed for session=%s: %s",
                    session_id, sess_exc,
                )
                continue

        _update_store_size(store)
        logger.info("ExperienceExtraction: extracted %d new experiences", extracted)

    except Exception as exc:
        logger.error("ExperienceExtraction.run_extraction failed: %s", exc)

    return extracted


def _update_store_size(store) -> None:
    """Update the Prometheus gauge with the current collection size."""
    if _PROM:
        try:
            experience_store_size.set(store.get_collection_size())
        except Exception:
            pass
