"""
Chroma Experience Store — manages the ``successful_experiences`` collection.

Stores validated successful query→plan pairs extracted from high-rated
CSR sessions.  These experiences are retrieved at planning time and
injected as few-shot examples into the Central Supervisor's
PLANNING_SYSTEM_PROMPT, enabling the planner to learn from past
successes without fine-tuning.

The collection uses the same Chroma instance, embedding model, and
connection singleton as the existing policy/FAQ collections — no new
infrastructure required.

Collection schema (per document)::

    id:        session_id (deduplicated — one experience per session)
    document:  original user query text (embedded for similarity search)
    metadata:
        session_id:       str   — CSIP session ID
        plan_json:        str   — JSON string of the central plan (goals + steps)
        team_assignments: str   — comma-separated team names from the plan
        result_summary:   str   — compressed summary of tool results
        rating:           str   — feedback rating ('correct')
        stored_at:        str   — ISO-8601 timestamp
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ChromaExperienceStore:
    """
    Thin wrapper around a single Chroma collection that stores and
    retrieves successful query→plan experiences.
    """

    COLLECTION_NAME = "successful_experiences"

    def __init__(self, chroma_conn=None):
        """
        Args:
            chroma_conn: A ``ChromaConnection`` instance.  If *None*,
                         the module-level singleton from connections.py
                         is used.
        """
        if chroma_conn is None:
            from databases.connections import get_chroma
            chroma_conn = get_chroma()
        self._conn = chroma_conn

    # ------------------------------------------------------------------
    # Collection access (lazy — created on first call)
    # ------------------------------------------------------------------

    def _get_collection(self):
        """Return the experience collection, creating it if necessary."""
        return self._conn.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Successful query-plan pairs for experience-augmented planning"},
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def store_experience(
        self,
        session_id: str,
        query_text: str,
        plan_json: str,
        team_assignments: str,
        result_summary: str = "",
        rating: str = "correct",
    ) -> bool:
        """
        Store a successful experience record.

        The *query_text* is embedded for similarity search.  All other
        fields are stored as Chroma metadata.

        Args:
            session_id:       Unique CSIP session ID (used as document ID
                              for deduplication).
            query_text:       The original user query.
            plan_json:        JSON string of the central plan produced for
                              this query.
            team_assignments: Comma-separated team names (e.g.
                              ``"claims_services_team,pa_services_team"``).
            result_summary:   Compressed summary of tool results.
            rating:           Feedback rating that qualified this session.

        Returns:
            True on success, False on failure.
        """
        try:
            collection = self._get_collection()
            collection.upsert(
                ids=[session_id],
                documents=[query_text],
                metadatas=[{
                    "session_id": session_id,
                    "plan_json": plan_json[:4000],           # Chroma metadata limit
                    "team_assignments": team_assignments,
                    "result_summary": result_summary[:2000],
                    "rating": rating,
                    "stored_at": datetime.now(timezone.utc).isoformat(),
                }],
            )
            logger.info(
                "ExperienceStore: stored experience session=%s teams=%s",
                session_id, team_assignments,
            )
            return True
        except Exception as exc:
            logger.error("ExperienceStore.store_experience failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def retrieve_similar_experiences(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve the *top_k* most similar past successful plans.

        Returns a formatted text block ready for injection into the
        planning prompt.  If the collection is empty or the query fails,
        returns an empty string (graceful degradation).
        """
        try:
            collection = self._get_collection()
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return ""

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if results.get("distances") else [None] * len(docs)

            lines: List[str] = []
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
                plan_json = meta.get("plan_json", "{}")
                teams = meta.get("team_assignments", "unknown")
                try:
                    plan = json.loads(plan_json)
                    goal_count = len(plan.get("goals", []))
                    step_count = len(plan.get("steps", []))
                    # Build a concise representation of the plan
                    steps_desc = []
                    for step in plan.get("steps", []):
                        agent = step.get("agent", "unknown")
                        steps_desc.append(f"    step {step.get('order', '?')}: {agent}")
                    steps_text = "\n".join(steps_desc) if steps_desc else "    (no steps)"
                except (json.JSONDecodeError, TypeError):
                    goal_count = "?"
                    step_count = "?"
                    steps_text = "    (plan not parseable)"

                lines.append(
                    f"Example {i}:\n"
                    f"  Query: \"{doc}\"\n"
                    f"  Successful Plan: {goal_count} goal(s), {step_count} step(s)\n"
                    f"  Teams: {teams}\n"
                    f"  Steps:\n{steps_text}"
                )

            return "\n\n".join(lines)

        except Exception as exc:
            logger.warning("ExperienceStore.retrieve_similar_experiences failed (non-fatal): %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def remove_experience(self, session_id: str) -> bool:
        """
        Remove an experience record by session_id.

        Used by supervisors to remove experiences that were later found
        to be incorrect.
        """
        try:
            collection = self._get_collection()
            collection.delete(ids=[session_id])
            logger.info("ExperienceStore: removed experience session=%s", session_id)
            return True
        except Exception as exc:
            logger.error("ExperienceStore.remove_experience failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_collection_size(self) -> int:
        """Return the number of documents in the experience collection."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_experience_store: Optional[ChromaExperienceStore] = None


def get_experience_store() -> ChromaExperienceStore:
    """Return (and lazily create) the module-level ExperienceStore singleton."""
    global _experience_store
    if _experience_store is None:
        _experience_store = ChromaExperienceStore()
    return _experience_store
